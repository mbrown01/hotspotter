"""Amino-acid reference data and geometric cutoffs used across the pipeline.

Everything a feature module needs to *classify atoms and residues* lives here, in one
place, so the chemistry is auditable and easy to sanity-check with a domain expert.

BIOLOGY NOTE (read me):
    A residue = one amino acid in the chain. Each has a fixed *backbone* (the repeating
    N-Cα-C=O spine, atoms named N, CA, C, O) and a variable *side chain* (everything
    else, the part that gives each amino acid its personality: charge, size, greasiness,
    aromatic ring, etc.). Almost all interface *chemistry* is side-chain chemistry, so
    most of the atom sets below name specific side-chain atoms.

    Atom naming follows the PDB convention: the Greek-letter walk out from the backbone.
    CB = beta carbon (first side-chain atom), then CG (gamma), CD (delta), CE (epsilon),
    CZ (zeta), etc. Branches get a number: CD1/CD2, NH1/NH2. So "Arg NH1/NH2" are the two
    terminal nitrogens of arginine's guanidinium group — the cationic tip that forms salt
    bridges (this is Arg95 in the ARL15-CNNM2 story).
"""

from __future__ import annotations

# --- Three-letter <-> one-letter amino-acid codes -------------------------------------
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}
STANDARD_RESIDUES = frozenset(THREE_TO_ONE)

# --- Formal side-chain charge at physiological pH (~7.4) ------------------------------
# +1 cationic, -1 anionic, 0 neutral. HIS is ~+0.1 (mostly neutral) at pH 7.4, so we
# treat it as neutral for charge but still allow it as a weak salt-bridge / cation-pi
# partner elsewhere. See docs/biology/02_interaction_chemistry.md.
FORMAL_CHARGE = {
    "ARG": +1, "LYS": +1,
    "ASP": -1, "GLU": -1,
    "HIS": 0,   # borderline; flagged separately as a possible cationic partner
}

# Atoms that carry the charge, by residue. Salt bridges form between these specific
# atoms, not residue centroids -> we measure atom-atom distances.
CATIONIC_ATOMS = {
    "ARG": {"NH1", "NH2", "NE"},   # guanidinium
    "LYS": {"NZ"},                 # ammonium
    "HIS": {"ND1", "NE2"},         # imidazole (weak / pH-dependent)
}
ANIONIC_ATOMS = {
    "ASP": {"OD1", "OD2"},         # carboxylate
    "GLU": {"OE1", "OE2"},         # carboxylate
}

# --- Hydrogen-bond donor / acceptor heavy atoms ---------------------------------------
# We usually lack explicit hydrogens in X-ray structures, so we detect H-bonds by the
# donor(heavy)...acceptor(heavy) distance and, when H atoms exist, a D-H...A angle.
# Backbone: N donates, O accepts. Side-chain sets below.
HBOND_DONORS = {
    "ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}, "ASN": {"ND2"}, "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "TRP": {"NE1"}, "CYS": {"SG"},
}
HBOND_ACCEPTORS = {
    "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}, "ASN": {"OD1"}, "GLN": {"OE1"},
    "HIS": {"ND1", "NE2"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "MET": {"SD"}, "CYS": {"SG"},
}
BACKBONE_DONOR_ATOMS = {"N"}       # amide N-H
BACKBONE_ACCEPTOR_ATOMS = {"O"}    # carbonyl O

# --- Hydrophobic side-chain carbons ---------------------------------------------------
# "Greasy" residues; hydrophobic contact = close apolar carbon-carbon packing that lets
# these residues bury away from water together. We list nonpolar side-chain carbons only
# (exclude the polar carbonyl backbone carbon).
HYDROPHOBIC_RESIDUES = frozenset({"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"})
HYDROPHOBIC_CARBONS = {
    "ALA": {"CB"},
    "VAL": {"CB", "CG1", "CG2"},
    "LEU": {"CB", "CG", "CD1", "CD2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"},
    "MET": {"CB", "CG", "CE"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "PRO": {"CB", "CG", "CD"},
}

# --- Aromatic ring atoms (for pi-stacking / aromatic contacts) ------------------------
# Six-membered (or indole) ring atoms whose centroid + normal define the ring plane.
AROMATIC_RESIDUES = frozenset({"PHE", "TYR", "TRP", "HIS"})
AROMATIC_RING_ATOMS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],  # six-membered benzene ring
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],          # five-membered imidazole
}

# --- Geometric cutoffs (angstroms / degrees) ------------------------------------------
# These are the knobs a structural biologist will want to sanity-check. Defaults follow
# common literature conventions; every one is documented in the features glossary.
class Cutoffs:
    INTERFACE_HEAVY_ATOM = 5.0     # residue is "at interface" if any heavy atom within this of the other chain
    SALT_BRIDGE = 4.0              # cationic N ... anionic O centroid/atom distance (Barlow & Thornton ~4 A)
    HBOND_DISTANCE = 3.5           # donor(heavy) ... acceptor(heavy)
    HBOND_ANGLE_MIN = 120.0        # D-H...A angle, only checked when H atoms are present
    HYDROPHOBIC_CONTACT = 4.5      # apolar C ... apolar C
    AROMATIC_CENTROID = 6.0        # ring centroid ... ring centroid
    DISULFIDE = 2.5               # CYS SG ... CYS SG (true bonds ~2.05 A)
    PACKING_RADIUS = 10.0          # heavy-atom count within this radius = local packing density
    SASA_INTERFACE_MIN = 1.0       # residue counts as interface if it buries at least this many A^2


# --- Maximum solvent-accessible surface area per residue (A^2) -------------------------
# Used to turn absolute SASA into *relative* SASA (RSA = SASA / max). Theoretical Gly-X-Gly
# tripeptide values from Tien et al. 2013 (PLoS ONE) — the modern standard.
MAX_ASA_TIEN = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

# --- Kyte-Doolittle hydropathy (higher = more hydrophobic) ----------------------------
KYTE_DOOLITTLE = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}

# --- Side-chain volume (A^3), Zamyatnin 1972 ------------------------------------------
RESIDUE_VOLUME = {
    "ALA": 88.6, "ARG": 173.4, "ASN": 114.1, "ASP": 111.1, "CYS": 108.5,
    "GLN": 143.8, "GLU": 138.4, "GLY": 60.1, "HIS": 153.2, "ILE": 166.7,
    "LEU": 166.7, "LYS": 168.6, "MET": 162.9, "PHE": 189.9, "PRO": 112.7,
    "SER": 89.0, "THR": 116.1, "TRP": 227.8, "TYR": 193.6, "VAL": 140.0,
}

# --- Flexibility propensity (Vihinen et al. 1994 normalized B-factor scale) -----------
# Higher = intrinsically more flexible/mobile backbone. A cheap proxy in the absence of MD.
FLEXIBILITY = {
    "ALA": 0.984, "ARG": 1.008, "ASN": 1.048, "ASP": 1.068, "CYS": 0.906,
    "GLN": 1.037, "GLU": 1.094, "GLY": 1.031, "HIS": 0.950, "ILE": 0.927,
    "LEU": 0.935, "LYS": 1.102, "MET": 0.952, "PHE": 0.915, "PRO": 1.049,
    "SER": 1.046, "THR": 0.997, "TRP": 0.904, "TYR": 0.929, "VAL": 0.931,
}
