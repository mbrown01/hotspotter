"""Edge-aware graph attention network for per-residue hot-spot classification.

WHY ATTENTION, AND WHY EDGE-AWARE
    A hot spot is usually made by ONE specific interaction — a single salt bridge reaching
    across the interface — not by the average character of a residue's neighborhood. A GCN
    averages over neighbors, which is exactly the wrong inductive bias: it would dilute the
    one contact that matters into the six that don't.

    Attention lets each residue learn which neighbor to listen to. But plain attention scores
    only on node features, and in this graph the decisive information lives on the EDGES:
    whether a contact is a salt bridge or an incidental hydrophobic brush, how far apart the
    partners sit, and whether the edge reaches across the interface or runs along the same
    chain. ``GATv2Conv(edge_dim=...)`` folds ``edge_attr`` into the attention computation, so
    "pay attention to the neighbor I salt-bridge with" is directly expressible.

    GATv2 rather than GAT: the original GAT computes *static* attention — the ranking of
    neighbors is the same for every query node, a limitation Brody et al. (2022) showed is
    real. GATv2 makes attention dynamic for the cost of reordering one matrix multiply.

SHAPES
    x          [n_nodes, 26]  the pinned Phase-1 node features
    edge_index [2, n_edges]   both directions stored
    edge_attr  [n_edges, 7]   5 interaction-type flags, distance, is_cross_interface
    out        [n_nodes]      ONE logit per node, for BCEWithLogitsLoss

    Every node gets a prediction; the training loop applies the loss only where
    ``label_mask`` is True, because ~82% of nodes were never measured.

SIZING, DELIBERATELY SMALL
    There are 1,540 labeled nodes across 172 graphs. That is a small supervised set, and the
    matched XGBoost baseline sits at PR-AUC 0.502 +/- 0.081. A large model would memorize
    rather than generalize, and the fold-to-fold noise would hide it. Defaults are therefore
    modest (2 layers, 64 hidden, 4 heads) and every knob is exposed so capacity can be raised
    deliberately rather than by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from hotspotter.ml.graph_dataset import EDGE_ATTR_NAMES, NODE_FEATURES

N_NODE_FEATURES = len(NODE_FEATURES)   # 26
N_EDGE_FEATURES = len(EDGE_ATTR_NAMES)  # 7


@dataclass
class GNNConfig:
    """Hyperparameters, in one place so a run can be logged and reproduced."""

    in_channels: int = N_NODE_FEATURES
    edge_dim: int = N_EDGE_FEATURES
    hidden_channels: int = 64
    n_layers: int = 2
    heads: int = 4
    dropout: float = 0.3
    attn_dropout: float = 0.1
    head_hidden: int = 32
    residual: bool = True
    wide: bool = False
    """Wide & Deep: feed the standardized raw features straight to the classifier head.

    FIX #1 for the inductive-bias ceiling. Across four configurations the GAT sat ~0.03
    BELOW gradient boosting on identical features and identical folds, which says message
    passing was diluting sharp per-residue signal rather than adding to it: a lone salt
    bridge gets averaged against six uninformative neighbors on the way to the head.

    With ``wide=True`` the head sees ``[deep_embedding (hidden) || x_norm (26)]``. The raw
    features reach the classifier without passing through any attention layer, so the model
    can fall back on the tabular signal and use message passing only where it ADDS
    something. This is the Cheng et al. (2016) wide-and-deep idea: memorization (wide) and
    generalization (deep) on separate paths.
    """
    drop_features: tuple[str, ...] = ()
    """Node feature names to exclude before the model sees them.

    FIX #2. A feature audit over all 8,525 nodes found three problems that cost a small MLP
    head more than they cost a tree ensemble:

      dead      ``n_disulfides`` is identically 0 across the entire alanine dataset — no
                cross-interface disulfide occurs — so it is pure wasted input width.
      constant  ``is_interface_sasa`` is 1.0 for 97.1% of nodes, carrying almost no signal.
      redundant ``has_salt_bridge`` is exactly ``n_salt_bridges > 0``; ``is_aromatic``,
                ``is_charged`` and ``is_polar`` restate information already present in
                ``n_aromatic``, ``charge`` and ``hydropathy``.

    XGBoost is indifferent to redundant columns — it simply never splits on them. A dense
    head is not: every input gets a weight, and correlated inputs split the gradient
    between them. This matters more now that Wide & Deep feeds raw features straight into
    that head.
    """

    extra_feature_names: tuple[str, ...] = ()
    """Names of feature columns appended to x BEYOND the 26 pinned NODE_FEATURES.

    FIX #3 uses this for one column: the nested out-of-fold XGBoost probability. Declaring
    the name here keeps the model's notion of its own input layout explicit, so a stacked
    run cannot be silently confused with an unstacked one.
    """

    clip_bfactor: float | None = None
    """Clip the ``bfactor`` column at this value before standardization.

    B-factors reach 502.6 in this dataset against a mean of 45.3. Values above ~100 A^2
    indicate poorly-ordered regions and are not meaningfully comparable. Z-scoring does not
    fix a long right tail — it just gives one residue a z-score of 11 — so the tail is
    clipped first.
    """

    norm: str = "layer"
    """Normalization between layers: "layer" (default) or "batch".

    LayerNorm normalizes each node across its own feature vector, so it is independent of
    how many nodes happen to share a batch. That matters here: interfaces range from about
    20 to 150 residues, so BatchNorm's running statistics are computed over a node
    population whose composition swings with whichever graphs land together — and at
    evaluation time the running averages may not match the graphs being scored at all.
    """

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class FeatureStandardizer(nn.Module):
    """Z-score node and edge features using statistics from the TRAINING fold only.

    This is not cosmetic. Raw node features span wildly different scales — ``sasa_unbound``
    runs to the hundreds of square angstroms while ``charge`` is in [-1, 1] — and an
    unnormalized input makes the large-magnitude columns dominate the first projection
    regardless of how informative they are.

    Statistics are buffers, not parameters: they are fitted once on the training fold, move
    with ``.to(device)``, and are saved in the state dict. Fitting them on all data would
    leak test-fold information into training, which on a 1,540-node dataset is exactly the
    kind of small, invisible leak that inflates a result.
    """

    def __init__(self, n_node: int = N_NODE_FEATURES, n_edge: int = N_EDGE_FEATURES):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(n_node))
        self.register_buffer("x_std", torch.ones(n_node))
        self.register_buffer("e_mean", torch.zeros(n_edge))
        self.register_buffer("e_std", torch.ones(n_edge))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, graphs) -> "FeatureStandardizer":
        """Compute means/stds over a list of Data objects (the training fold)."""
        x = torch.cat([g.x for g in graphs], dim=0)
        e = torch.cat([g.edge_attr for g in graphs], dim=0)
        self.x_mean.copy_(x.mean(0))
        self.e_mean.copy_(e.mean(0))
        # clamp: a constant column (std 0) would produce inf/nan on division
        self.x_std.copy_(x.std(0).clamp_min(1e-6))
        self.e_std.copy_(e.std(0).clamp_min(1e-6))
        self.fitted.fill_(True)
        return self

    def forward(self, x, edge_attr):
        return (x - self.x_mean) / self.x_std, (edge_attr - self.e_mean) / self.e_std


class HotSpotGAT(nn.Module):
    """GATv2 node classifier over an interface graph.

    Layout::

        standardize -> encoder(Linear+BN+ReLU+Dropout)
                    -> n_layers x [ GATv2Conv(edge_dim=7) -> BN -> ELU -> Dropout (+residual) ]
                    -> head(Linear -> ReLU -> Dropout -> Linear) -> 1 logit per node

    The encoder projects 26 features to ``hidden_channels`` before any message passing, so
    every attention layer operates at one width and residual connections line up.
    """

    def __init__(self, config: GNNConfig | None = None, **overrides):
        super().__init__()
        cfg = config or GNNConfig(**overrides)
        if cfg.hidden_channels % cfg.heads != 0:
            raise ValueError(
                f"hidden_channels ({cfg.hidden_channels}) must be divisible by heads "
                f"({cfg.heads}) so concatenated head outputs come back to hidden width"
            )
        if cfg.norm not in ("layer", "batch"):
            raise ValueError(f"norm must be 'layer' or 'batch', got {cfg.norm!r}")
        self.config = cfg

        # --- feature selection / clipping, applied before anything else sees x ----------
        all_features = tuple(NODE_FEATURES) + tuple(cfg.extra_feature_names)
        unknown = set(cfg.drop_features) - set(all_features)
        if unknown:
            raise ValueError(f"drop_features names not in the input layout: {sorted(unknown)}")
        keep = [i for i, f in enumerate(all_features) if f not in set(cfg.drop_features)]
        if not keep:
            raise ValueError("drop_features would remove every node feature")
        self.kept_features = tuple(all_features[i] for i in keep)
        self.register_buffer("keep_idx", torch.tensor(keep, dtype=torch.long))
        # position of bfactor in the RAW column order (clipping happens pre-selection)
        bf = all_features.index("bfactor") if "bfactor" in all_features else -1
        self.register_buffer("bfactor_idx", torch.tensor(bf, dtype=torch.long))

        in_dim = len(keep)
        self.in_dim = in_dim
        self.standardizer = FeatureStandardizer(in_dim, cfg.edge_dim)

        def make_norm(width: int) -> nn.Module:
            return (nn.LayerNorm(width) if cfg.norm == "layer"
                    else nn.BatchNorm1d(width))

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_channels),
            make_norm(cfg.hidden_channels),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        per_head = cfg.hidden_channels // cfg.heads
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=cfg.hidden_channels,
                    out_channels=per_head,
                    heads=cfg.heads,
                    concat=True,              # heads * per_head == hidden_channels
                    dropout=cfg.attn_dropout,  # dropout ON THE ATTENTION COEFFICIENTS
                    edge_dim=cfg.edge_dim,     # <-- edge features enter the attention score
                    add_self_loops=True,       # a residue keeps access to its own features
                )
            )
            self.norms.append(make_norm(cfg.hidden_channels))

        self.dropout = nn.Dropout(cfg.dropout)
        # Wide path: the head also sees the standardized raw features, untouched by any
        # attention layer. 32 (deep) + 26 (wide) = 58 inputs at the default width.
        head_in = cfg.hidden_channels + (in_dim if cfg.wide else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    def prepare_x(self, x: torch.Tensor) -> torch.Tensor:
        """Clip the bfactor tail, then keep only the selected feature columns.

        Applied identically when fitting the standardizer and at every forward pass, so
        normalization statistics always describe the tensor the model actually consumes.
        """
        if self.config.clip_bfactor is not None and int(self.bfactor_idx) >= 0:
            x = x.clone()
            j = int(self.bfactor_idx)
            x[:, j] = x[:, j].clamp(max=self.config.clip_bfactor)
        return x.index_select(1, self.keep_idx)

    @torch.no_grad()
    def fit_standardizer(self, graphs) -> "HotSpotGAT":
        """Fit normalization on the TRAINING fold, after clipping and column selection."""
        xs = torch.cat([self.prepare_x(g.x) for g in graphs], dim=0)
        es = torch.cat([g.edge_attr for g in graphs], dim=0)
        self.standardizer.x_mean.copy_(xs.mean(0))
        self.standardizer.x_std.copy_(xs.std(0).clamp_min(1e-6))
        self.standardizer.e_mean.copy_(es.mean(0))
        self.standardizer.e_std.copy_(es.std(0).clamp_min(1e-6))
        self.standardizer.fitted.fill_(True)
        return self

    def forward(self, x, edge_index, edge_attr, batch=None) -> torch.Tensor:
        """Return raw logits, shape [n_nodes]. No sigmoid — BCEWithLogitsLoss applies it."""
        x_norm, edge_attr = self.standardizer(self.prepare_x(x), edge_attr)
        h = self.encoder(x_norm)

        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index, edge_attr=edge_attr)
            out = norm(out)
            out = F.elu(out)
            out = self.dropout(out)
            # Residual keeps a residue's own features reachable at the output even if
            # attention routes everything to its neighbors -- and helps the isolated nodes
            # (0.5% of the dataset) that receive no messages at all.
            h = h + out if self.config.residual else out

        if self.config.wide:
            h = torch.cat([h, x_norm], dim=-1)
        return self.head(h).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, data) -> torch.Tensor:
        """Sigmoid probabilities for every node in a Data/Batch."""
        self.eval()
        return torch.sigmoid(
            self(data.x, data.edge_index, data.edge_attr, getattr(data, "batch", None))
        )

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        c = self.config
        return (
            f"HotSpotGAT({c.n_layers} x GATv2Conv, hidden={c.hidden_channels}, "
            f"heads={c.heads}, edge_dim={c.edge_dim}, dropout={c.dropout}, "
            f"norm={c.norm}, residual={c.residual}, wide={c.wide}, "
            f"in_dim={self.in_dim}"
            f"{', clip_bf=' + str(c.clip_bfactor) if c.clip_bfactor else ''}) "
            f"-> {self.n_parameters():,} parameters"
        )


def masked_bce_loss(logits, y, label_mask, pos_weight=None):
    """BCEWithLogits over LABELED nodes only.

    Most nodes were never measured. Their ``y`` is nan, so they must be excluded before the
    loss sees them — training an unmeasured residue toward 0 would assert "confirmed not a
    hot spot", which the data does not say.

    ``pos_weight`` is the GNN's equivalent of XGBoost's ``scale_pos_weight``: with a 23%
    positive rate, an unweighted model does well by predicting "not a hot spot" everywhere.
    """
    if label_mask.sum() == 0:
        return logits.sum() * 0.0      # keeps the graph connected for autograd
    sel_logits = logits[label_mask]
    sel_y = y[label_mask].float()
    return F.binary_cross_entropy_with_logits(sel_logits, sel_y, pos_weight=pos_weight)
