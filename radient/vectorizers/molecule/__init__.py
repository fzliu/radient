__all__ = [
    "RDKitMoleculeVectorizer"
    "molecule_vectorizer"
]

from typing import Optional

from radient.vectorizers.molecule.base import MoleculeVectorizer
from radient.vectorizers.molecule.rdkit import RDKitMoleculeVectorizer


def molecule_vectorizer(method: str = "rdkit", **kwargs) -> MoleculeVectorizer:
    """Creates a text vectorizer specified by `method`.
    """

    # Return a reasonable default vectorizer in the event that the user does
    # not specify one.
    if method in ("rdkit",):
        return RDKitMoleculeVectorizer(**kwargs)
    else:
        raise NotImplementedError
