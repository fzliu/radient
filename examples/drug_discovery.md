## Drug discovery

Outside of biotech circles, AI-powered drug discovery isn't a well-known use case for embedding vectors. In reality, billion-scale vector search is frequently in this industry, and this particular application demonstrates how vectors can be used to represent non-traditional unstructured data.

By way of example, let's use the default `molecule_vectorizer` to generate embedding vectors for molecular structures. We'll first grab the dataset of FDA-approved drugs and their corresponding SMILES strings (SMILES is a way to describe molecular structure using a string of letters and symbols). We'll then vectorize all of these SMILES strings and search the results to see if we can discover alternatives to Ibuprofen (often sold as Advil or Motrin), an analgesic, anti-inflammatory drug.

We'll start with our imports:

```python
import csv

import numpy as np
import requests
import scipy as sp

from radient import molecule_vectorizer
```

From here, let's use `requests` and `csv` to download and parse the dataset, respectively:

```python
r = requests.get("https://gist.githubusercontent.com/fzliu/8052bd4d609bc6260ab7e8c838d2f518/raw/f1c9efb816d6b8514c0a643323f7afa29372b1c4/fda_approved_structures.csv")
csv_data = csv.reader(r.text.splitlines(), delimiter=",")
mol_data = [{"name": d[0], "mol": d[1]} for d in csv_data]
```

Now we'll create our vectorizer and compute vectors for all molecules. The query vector is generated from the SMILES string for Ibuprofen:

```python 
vectorizer = molecule_vectorizer()
vec_data = vectorizer.vectorize([row["mol"] for row in mol_data])
query = vectorizer.vectorize("CC(C)CC1=CC=C(C=C1)C(C)C(O)=O")
```

With that out of the way, let's find the "closest" drugs to Ibuprofen. We're using [Jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index) since the default molecule vectorizer returns binary vectors:

```python
dists = sp.spatial.distance.cdist(
    query[np.newaxis,...],
    vectors,
    metric="jaccard"
).squeeze()
top10 = [mol_data[i]["name"] for i in np.argsort(dists)[:10]]
print(top10)
```

    ['Dexibuprofen', 'Ibuprofen', 'Loxoprofen', 'Phenylacetic acid', 'Naproxen', 'Fenoprofen', 'Ketoprofen', 'Dexketoprofen', 'Mandelic acid', 'Oxeladin']

Ibuprofen's similarity with many of these drugs is clear: Loxoprofen, Phenylacetic acid, Naproxen, Fenoprofen, and Ketoprofen, are, like Ibuprofen, all analgesic, anti-inflammatory drugs. Surprisingly, Mandelic acid and Oxeladin are relevant too; some studies show that they also possess anti-inflammatory properties.

For convenience, here's the full script:

```python
import csv

import numpy as np
import requests
import scipy as sp

from radient import molecule_vectorizer

r = requests.get("https://gist.githubusercontent.com/fzliu/8052bd4d609bc6260ab7e8c838d2f518/raw/f1c9efb816d6b8514c0a643323f7afa29372b1c4/fda_approved_structures.csv")
csv_data = csv.reader(r.text.splitlines(), delimiter=",")
mol_data = [{"name": d[0], "mol": d[1]} for d in csv_data]

vectorizer = molecule_vectorizer()
vec_data = vectorizer.vectorize([row["mol"] for row in mol_data])
query = vectorizer.vectorize("CC(C)CC1=CC=C(C=C1)C(C)C(O)=O")

dists = sp.spatial.distance.cdist(
    query[np.newaxis,...],
    vectors,
    metric="jaccard"
).squeeze()
top10 = [mol_data[i]["name"] for i in np.argsort(dists)[:10]]
print(top10)
```