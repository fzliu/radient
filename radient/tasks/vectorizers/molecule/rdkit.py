__all__ = [
    "RDKitMoleculeVectorizer"
]

from typing import Any, List

import numpy as np

from radient.utils.lazy_import import LazyImport
from radient.vector import Vector
from radient.tasks.vectorizers.molecule._base import MoleculeVectorizer

Chem = LazyImport("rdkit.Chem")
AllChem = LazyImport("rdkit.Chem.AllChem")


class RDKitMoleculeVectorizer(MoleculeVectorizer):
    """Generates binary fingerprints for molecules.
    """

    def __init__(self, fingerprint_type: str = "topological", **kwargs):
        super().__init__()
        self._fingerprint_type = fingerprint_type
        if fingerprint_type == "topological":
            self._fpgen = AllChem.GetRDKitFPGenerator(**kwargs)
        elif fingerprint_type == "morgan":
            self._fpgen = AllChem.GetMorganGenerator(**kwargs)

    def _vectorize(self, molecule: str, **kwargs) -> Vector:
        if isinstance(molecule, str):
            molecule = Chem.MolFromSmiles(molecule)
        fp = self._fpgen.GetFingerprint(molecule)
        # Use dtype=bool to avoid having to bit-pack into `uint8`.
        vector = np.array(fp.ToList(), dtype=bool)
        return vector.view(Vector)

    @property
    def fingerprint_type(self) -> str:
        return self._fingerprint_type
