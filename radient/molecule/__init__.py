__all__ = [
    "RDKitMoleculeVectorizer",
    "molecule_vectorizer"
]

from typing import Optional

from radient.molecule.base import MoleculeVectorizer
from radient.molecule.rdkit import RDKitMoleculeVectorizer


def molecule_vectorizer(method: Optional[str] = None, **kwargs) -> MoleculeVectorizer:
    """Creates a text vectorizer specified by `method`.
    """

    if method == None:
        # Return a reasonable default vectorizer in the event that the user does not
        # specify one.
        return RDKitMoleculeVectorizer()
    elif method == "rdkit":
        return RDKitMoleculeVectorizer(**kwargs)
    else:
        raise NotImplementedError
