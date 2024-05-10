## Drug discovery

AI-powered drug discovery often isn't a top-of-mind use case for embedding vectors, but it's actually a powerful application and goes to show how vector search can be used to store unstructured data more broadly.

By way of example, let we'll grab the dataset of FDA-approved drugs and their corresponding SMILES strings (SMILES is a way to describe molecular structure using a string of letters and symbols). We'll then vectorize all of these SMILES strings and search the results to see if we can discover alternatives to Ibuprofen (often sold as Advil or Motrin), an analgesic, anti-inflammatory drug.

```python
import csv

import numpy as np
import requests
import scipy as sp

from radient import molecule_vectorizer

# This dataset contains a list of all FDA-approved drugs and their
# corresponding SMILES strings.
r = requests.get("https://gist.githubusercontent.com/fzliu/8052bd4d609bc6260ab7e8c838d2f518/raw/f1c9efb816d6b8514c0a643323f7afa29372b1c4/fda_approved_structures.csv")
csv_data = csv.reader(r.text.splitlines(), delimiter=",")
mol_data = [{"name": d[0], "mol": d[1]} for d in csv_data]

# Create our vectorizer and compute vectors for all molecules. The query
# vector is generated from the SMILES string for Ibuprofen, a drug used to
# treat pain and inflammation.
vectorizer = molecule_vectorizer()
vec_data = vectorizer.vectorize([row["mol"] for row in mol_data])
query = vectorizer.vectorize("CC(C)CC1=CC=C(C=C1)C(C)C(O)=O")

# Let's find the "closest" drugs to Ibuprofen. We're using Jaccard distance
# since the default molecule vectorizer returns binary vectors.
dists = sp.spatial.distance.cdist(
    query[np.newaxis,...],
    vectors,
    metric="jaccard"
).squeeze()
top10 = [mol_data[i]["name"] for i in np.argsort(dists)[:10]]
print(top10)
```

    ['Dexibuprofen', 'Ibuprofen', 'Loxoprofen', 'Phenylacetic acid', 'Naproxen', 'Fenoprofen', 'Ketoprofen', 'Dexketoprofen', 'Mandelic acid', 'Oxeladin']

The similarity with many of these is clear: Loxoprofen, Phenylacetic acid, Naproxen, Fenoprofen, and Ketoprofen, are, like Ibuprofen, all analgesic, anti-inflammatory drugs. Surprisingly, Mandelic acid and Oxeladin are relevant too; some studies show that they also possess anti-inflammatory properties.