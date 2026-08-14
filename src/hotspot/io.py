"""Structure input: download, parse, and lightly clean protein complex structures.

Handles both experimental (RCSB PDB) and predicted (AlphaFold DB) structures, in both
PDB and mmCIF formats. Everything downstream works on a Biopython ``Structure`` object
plus a small ``ResidueId`` helper so we can talk about "chain A, residue 95" unambiguously.
"""

from __future__ import annotations

import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from Bio.PDB import MMCIFParser, PDBParser, Select
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue

from hotspot.constants import STANDARD_RESIDUES, THREE_TO_ONE

# Where downloaded structures land (gitignored). Resolved relative to the repo root.
DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
# AlphaFold DB: model files are keyed by UniProt accession, e.g. P69905 -> ...-model_v4.cif
ALPHAFOLD_CIF_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.cif"
ALPHAFOLD_PAE_URL = (
    "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-predicted_aligned_error_v4.json"
)


@dataclass(frozen=True)
class ResidueId:
    """A stable, hashable identity for one residue: which chain, which position.

    ``resseq`` is the author residue number (what a biologist calls "residue 95"),
    ``icode`` is the insertion code (usually blank; non-empty in e.g. antibody numbering).
    """

    chain: str
    resseq: int
    icode: str = " "
    resname: str = ""

    @property
    def one_letter(self) -> str:
        return THREE_TO_ONE.get(self.resname, "X")

    @property
    def label(self) -> str:
        """Human-readable, e.g. 'A/ARG95' or 'D/TYR96'."""
        ins = self.icode.strip()
        return f"{self.chain}/{self.resname}{self.resseq}{ins}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.label


def residue_id(residue: Residue) -> ResidueId:
    """Build a :class:`ResidueId` from a Biopython residue."""
    _, resseq, icode = residue.get_id()
    chain = residue.get_parent().id
    return ResidueId(chain=chain, resseq=resseq, icode=icode, resname=residue.get_resname())


def is_amino_acid(residue: Residue) -> bool:
    """True for the 20 standard amino acids (excludes water, ions, ligands, HETATMs).

    We check the hetero flag *and* the residue name so that modified residues and ligands
    don't leak into the interface analysis.
    """
    hetflag, _, _ = residue.get_id()
    return hetflag == " " and residue.get_resname() in STANDARD_RESIDUES


# --------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------
def fetch_pdb(pdb_id: str, dest_dir: Path | None = None, fmt: str = "pdb") -> Path:
    """Download a structure from the RCSB PDB. Cached: skips download if already present.

    Parameters
    ----------
    pdb_id : 4-character PDB id, e.g. "1BRS".
    fmt    : "pdb" or "cif".
    """
    pdb_id = pdb_id.lower().strip()
    dest_dir = Path(dest_dir) if dest_dir else DATA_RAW
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = "pdb" if fmt == "pdb" else "cif"
    out = dest_dir / f"{pdb_id}.{ext}"
    if out.exists() and out.stat().st_size > 0:
        return out
    url = (RCSB_PDB_URL if fmt == "pdb" else RCSB_CIF_URL).format(pdb_id=pdb_id.upper())
    _download(url, out)
    return out


def fetch_alphafold(uniprot: str, dest_dir: Path | None = None) -> Path:
    """Download an AlphaFold DB model (mmCIF) by UniProt accession, e.g. 'P69905'.

    Note: AlphaFold DB entries are *single-chain* monomer predictions. Use this for
    monomer confidence features; for complexes you'll typically have a ColabFold multimer
    prediction file of your own.
    """
    uniprot = uniprot.upper().strip()
    dest_dir = Path(dest_dir) if dest_dir else DATA_RAW
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"AF-{uniprot}-F1.cif"
    if out.exists() and out.stat().st_size > 0:
        return out
    _download(ALPHAFOLD_CIF_URL.format(uniprot=uniprot), out)
    return out


_TRUST_INJECTED = False


def _ensure_trust() -> None:
    """Make Python's SSL trust the OS certificate store (once per process).

    On corporate networks (e.g. an intercepting proxy) the system installs its own root
    CA into the Windows/macOS trust store, but Python's ``requests`` ships its own bundle
    and doesn't see it -> downloads fail with CERTIFICATE_VERIFY_FAILED. ``truststore``
    bridges Python's ssl to the OS store and fixes this transparently. It's optional: if
    it isn't installed we just proceed (works fine on non-intercepted networks).
    """
    global _TRUST_INJECTED
    if _TRUST_INJECTED:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # pragma: no cover - best effort
        pass
    _TRUST_INJECTED = True


def _download(url: str, out: Path) -> None:
    _ensure_trust()
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out.write_bytes(resp.content)


def _maybe_gunzip(path: Path) -> Path:
    if path.suffix == ".gz":
        target = path.with_suffix("")
        with gzip.open(path, "rb") as fin, open(target, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return target
    return path


# --------------------------------------------------------------------------------------
# Parse
# --------------------------------------------------------------------------------------
def load_structure(path: str | Path, structure_id: str = "complex") -> Structure:
    """Parse a local .pdb / .cif (optionally .gz) file into a Biopython Structure.

    Uses the first model only (NMR ensembles and multi-model predictions have >1); the
    interface analysis is defined on a single coordinate set.
    """
    path = _maybe_gunzip(Path(path))
    suffix = path.suffix.lower()
    if suffix == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)  # QUIET silences noisy but harmless warnings
    structure = parser.get_structure(structure_id, str(path))
    return structure


def get_model(structure: Structure):
    """Return the first model of a structure (handles multi-model files)."""
    return next(structure.get_models())


def get_chain_ids(structure: Structure) -> list[str]:
    return [chain.id for chain in get_model(structure).get_chains()]


def iter_amino_acids(structure_or_model, chains: Iterable[str] | None = None):
    """Yield (chain_id, residue) for standard amino acids, optionally filtered by chain."""
    model = structure_or_model
    if isinstance(structure_or_model, Structure):
        model = get_model(structure_or_model)
    wanted = set(chains) if chains else None
    for chain in model.get_chains():
        if wanted is not None and chain.id not in wanted:
            continue
        for residue in chain.get_residues():
            if is_amino_acid(residue):
                yield chain.id, residue


class _AminoAcidSelect(Select):
    """Biopython Select that keeps only standard amino acids in the given chains."""

    def __init__(self, chains: Iterable[str] | None = None):
        self.chains = set(chains) if chains else None

    def accept_chain(self, chain):  # noqa: N802 (Biopython API)
        return self.chains is None or chain.id in self.chains

    def accept_residue(self, residue):  # noqa: N802
        return is_amino_acid(residue)
