__all__ = [
    "MoleculeVectorizer"
]

from abc import abstractmethod
from typing import Any, List

from radient.base import Vector, Vectorizer
from radient.util import fully_qualified_name, LazyImport

Chem = LazyImport("rdkit", attribute="Chem")


class MoleculeVectorizer(Vectorizer):

    @abstractmethod
    def __init__(self):
        super().__init__()

    @classmethod
    def standardize_input(cls, molecule: Any) -> str:
        full_name = fully_qualified_name(molecule)
        if full_name == "str":
            return molecule
        elif full_name == "rdkit.Chem.rdchem.Mol":
            # TODO: don't do this step for `RDKitMoleculeVectorizer`
            return Chem.MolToSmiles(molecule)
        else:
            raise TypeError
