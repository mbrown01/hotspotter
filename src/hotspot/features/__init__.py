"""Per-residue feature groups. Each module exposes a function that annotates interface
residues with features, keyed by :class:`hotspot.io.ResidueId`.

Feature groups:
    chemistry     salt bridges, H-bonds, hydrophobic, aromatic, disulfide (the Arg95 signal)
    sasa          buried surface area, dSASA, relative SASA (the burial signal)
    topology      central vs peripheral, contact counts, local packing (the O-ring signal)
    identity      amino-acid physicochemical properties (lookup tables)
    confidence    pLDDT / PAE (predicted) and B-factor (experimental) flexibility proxies
    conservation  evolutionary conservation from an MSA / ConSurf (planned; highest signal
                  after chemistry)
"""
