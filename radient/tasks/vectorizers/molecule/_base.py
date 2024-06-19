__all__ = [
    "MoleculeVectorizer"
]

from abc import abstractmethod
from typing import Any, List

from radient.vector import Vector
from radient.utils import fully_qualified_name
from radient.utils.lazy_import import LazyImport
from radient.tasks.vectorizers._base import Vectorizer

Chem = LazyImport("rdkit.Chem")


class MoleculeVectorizer(Vectorizer):

    @abstractmethod
    def __init__(self):
        super().__init__()

    def _preprocess(self, molecule: Any, **kwargs) -> str:
        full_name = fully_qualified_name(molecule)
        if full_name == "builtins.str":
            return molecule
        elif full_name == "rdkit.Chem.rdchem.Mol":
            # TODO: don't do this step for `RDKitMoleculeVectorizer`
            return Chem.MolToSmiles(molecule)
        else:
            raise TypeError
