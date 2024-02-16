__all__ = [
    "RDKitMoleculeVectorizer"
]

from typing import Any, List

from radient.base import Vector
from radient.util import LazyImport
from radient.molecule.base import MoleculeVectorizer

Chem = LazyImport("rdkit", attribute="Chem")
AllChem = LazyImport("rdkit.Chem", attribute="AllChem")


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

    def vectorize(self, molecules: List[Any]) -> List[Vector]:
        vectors = []
        for n, mol in enumerate(molecules):
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
            fp = self._fpgen.GetFingerprint(mol)
            # Use dtype=bool to avoid having to bit-pack into `uint8`.
            vector = np.array(fp.ToList(), dtype=bool)
            vectors.append(vector.view(Vector))
        return vectors

    @property
    def fingerprint_type(self) -> str:
        return self._fingerprint_type
