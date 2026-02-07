# Radient

Radient is a developer-friendly, lightweight library for generating and working with embeddings. Radient supports simple vectorization (i.e. turning data into vectors) as well as building retrieval systems (i.e. complex end-to-end workflows for embedding-based search).

```shell
$ pip install radient
```

If you find this project helpful or interesting, please consider giving it a star. :star:

### Getting started

Basic vectorization can be performed as follows:

```python
from radient import text_vectorizer
vz = text_vectorizer()
vz.vectorize("Hello, world!")
# Vector([-3.21440510e-02, -5.10351397e-02,  3.69579718e-02, ...])
```

The above snippet vectorizes the string `"Hello, world!"` using a default model, namely `bge-small-en-v1.5` from `sentence-transformers`. If your Python environment does not contain the `sentence-transformers` library, Radient will prompt you for it:

```python
vz = text_vectorizer()
# Vectorizer requires sentence-transformers. Install? [Y/n]
```

You can type "Y" to have Radient install it for you automatically.

Each vectorizer can take a `method` parameter along with optional keyword arguments which get passed directly to the underlying vectorization library. For example, we can pick Mixbread AI's `mxbai-embed-large-v1` model using the `sentence-transformers` library via:

```python
vz_mbai = text_vectorizer(method="sentence-transformers", model_name_or_path="mixedbread-ai/mxbai-embed-large-v1")
vz_mbai.vectorize("Hello, world!")
# Vector([ 0.01729078,  0.04468533,  0.00055427, ...])
```

### More than just text

With Radient, you're not limited to text. Audio, graphs, images, and molecules can be vectorized as well:

```python
from radient import (
    audio_vectorizer,
    graph_vectorizer,
    image_vectorizer,
    molecule_vectorizer,
)
avec = audio_vectorizer().vectorize(str(Path.home() / "audio.wav"))
gvec = graph_vectorizer().vectorize(nx.karate_club_graph())
ivec = image_vectorizer().vectorize(str(Path.home() / "image.jpg"))
mvec = molecule_vectorizer().vectorize("O=C=O")
```

For production use cases with large quantities of data, performance is key. Radient also provides an `accelerate` function to optimize some vectorizers on-the-fly:

```python
import numpy as np
vz = text_vectorizer()
vec0 = vz.vectorize("Hello, world!")
vz.accelerate()
vec1 = vz.vectorize("Hello, world!")
np.allclose(vec0, vec1)
# True
```

On a 2.3 GHz Quad-Core Intel Core i7, the original vectorizer returns in ~32ms, while the accelerated vectorizer returns in ~17ms.

### Building unstructured data ETL

Aside from running experiments, pure vectorization is not particularly useful. End-to-end search and retrieval systems leveraging embeddings often require four separate components:
1. A data __source__ where unstructured data is stored
2. One more more __transform__ modules that perform data conversions and pre-processing
3. A __vectorizer__ which turns the data into semantically rich embeddings
4. A __sink__ to persist the vectors once they have been computed

Radient provides a `Workflow` object specifically for building vector-centric ETL applications. With Workflows, you can combine any number of each of these components into a directed graph. For example, a workflow to continuously read text documents from Google Drive, vectorize them with [Voyage AI](https://www.voyageai.com/), and store the resulting vectors in MongoDB might look like:

```python
from radient import make_operator
from radient import Workflow

extract = make_operator("source", method="google-drive", task_params={"folder": "My Files"})
transform = make_operator("transform", method="read-text", task_params={})
vectorize = make_operator("vectorizer", method="voyage-ai", task_params={})
load = make_operator("sink", method="mongodb", task_params={"operation": "insert"})

wf = (
    Workflow()
    .add(extract, name="extract")
    .add(transform, name="transform")
    .add(vectorize, name="vectorize")
    .add(load, name="load")
)
```

You can use accelerated vectorizers and transforms in a Workflow by specifying `accelerate=True` for all supported operators.

### Supported vectorizer engines

Radient builds atop work from the broader ML community. Most open-source vectorizers come from other libraries:

- [Imagebind](https://imagebind.metademolab.com/)
- [Pytorch Image Models](https://huggingface.co/timm)
- [RDKit](https://rdkit.org)
- [Sentence Transformers](https://sbert.net)
- [scikit-learn](https://scikit-learn.org)
- [TorchAudio](https://pytorch.org/audio)

On-the-fly model acceleration is done via [ONNX](https://onnx.ai).

A massive thank you to all the creators and maintainers of these libraries.

### Coming soon&trade;

A couple of features slated for the near-term (hopefully):
1) Sparse vector, binary vector, and multi-vector support
2) Support for all relevant embedding models on Huggingface

LLM connectors _will not_ be a feature that Radient provides. Building context-aware systems around LLMs is a complex task, and not one that Radient intends to solve. Projects such as [Haystack](https://haystack.deepset.ai/) and [Llamaindex](https://www.llamaindex.ai/) are two of the many great options to consider if you're looking to extract maximum RAG performance.

Full write-up on Radient will come later, along with more sample applications, so stay tuned.

