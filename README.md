# Radient

Radient is a developer-friendly, lightweight library for vectorization, i.e. turning data into embeddings. Radient supports simple vectorization as well as complex vector-centric workflows.

```shell
$ pip install radient
```

### Why Radient?

In applications that leverage [RAG](https://zilliz.com/use-cases/llm-retrieval-augmented-generation), vector databases are commonly used as a way to retrieve relevant content that is relevant to the query. It's become so popular that "traditional" database vendors are rushing to support vector search. (Anybody see those [funky Singlestore ads](https://media.licdn.com/dms/image/D4E22AQE0uXihwNGBjQ/feedshare-shrink_2048_1536/0/1710685199486?e=2147483647&v=beta&t=t50JyZHIazYLQ_eVXbFtQpyhegiRiZEdxJjK0xBNLUo) on US-101?)

Although still predominantly used for text today, vectors will be used extensively across a variety of different modalities in the upcoming months. This evolution is being powered by two independent occurrences: 1) the shift from large language models to large _multimodal_ models (such as [GPT-4o](https://openai.com/index/hello-gpt-4o), [Reka](https://www.reka.ai), and [Fuyu](https://www.adept.ai/blog/adept-fuyu-heavy)), and 2) the rise in adoption for "traditional" tasks such as recommendation and semantic search. In short, vectors are going mainstream, and we need a way to vectorize _everything_, not just text.

If you find this project helpful or interesting, please consider giving it a star. :star:

### Getting started

Basic vectorization can be performed as follows:

```python
from radient import text_vectorizer
vz = text_vectorizer()
vz.vectorize("Hello, world!")  # Vector([-3.21440510e-02, -5.10351397e-02,  3.69579718e-02, ...])
```

The above snippet vectorizes the string `"Hello, world!"` using a default model, namely `bge-small-en-v1.5` from `sentence-transformers`. If your Python environment does not contain the `sentence-transformers` library, Radient will prompt you for it:

```python
vz = text_vectorizer()  # Vectorizer requires sentence-transformers. Install? [Y/n]
```

You can type "Y" to have Radient install it for you automatically.

Each vectorizer can take a `method` parameter along with optional keyword arguments which get passed directly to the underlying vectorization library. For example, we can pick Mixbread AI's `mxbai-embed-large-v1` model using the `sentence-transformers` library via:

```python
vz_mbai = text_vectorizer(method="sentence-transformers", model_name_or_path="mixedbread-ai/mxbai-embed-large-v1")
vz_mbai.vectorize("Hello, world!")  # Vector([ 0.01729078,  0.04468533,  0.00055427, ...])
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

A partial list of methods and optional kwargs supported by each modality can be found [here](https://github.com/fzliu/radient/blob/main/docs/supported_methods.md).

For production use cases with large quantities of data, performance is key. Radient also provides an `accelerate` function to optimize vectorizers on-the-fly:

```python
import numpy as np
vz = text_vectorizer()
vec0 = vz.vectorize("Hello, world!")
vz.accelerate()
vec1 = vz.vectorize("Hello, world!")
np.allclose(vec0, vec1)  # True
```

The original vectorizer returns in ~32ms, while the accelerated vectorizer returns in ~17ms.

### Building vector-centric workflows

Aside from running experiments, pure vectorization is not particularly useful. Real-world workloads often require a combination of four components:

- A data source (or a data reader) for extracting unstructured data,
- One or more transform modules such as video demuxing or OCR,
- A vectorizer or set of vectorizers, and
- A place to store the vectors once they have been computed.

Radient provides a `Workflow` object specifically for building vector-centric applications. With Workflows, you can combine any number of each of these components into a graph. For example, a workflow to read text

```python
from radient import Workflow, LocalRunner
from radient import source, transform, vectorizer, sink
wf = (
    Workflow()
    .add(LocalRunner(source, task_kwargs={"datasource": "google-drive"}))
    .add(LocalRunner(transform, task_kwargs={"method": "pdf-to-text"}))
    .add(LocalRunner(vectorizer, task_kwargs={"modality": "text", "method": "sentence-transformers"}))
    .add(LocalRunner(sink, task_kwargs={"datastore": "milvus"}))
)
```

With workflows, you can specify your own custom tasks as well.

You can use accelerated vectorizers and transforms in a Workflow by specifying `accelerate=True` for all supported tasks.

### Supported libraries

Radient builds atop work from the broader ML community. Most vectorizers come from other libraries:

- [Imagebind](https://imagebind.metademolab.com/)
- [Pytorch Image Models](https://huggingface.co/timm)
- [RDKit](https://rdkit.org)
- [Sentence Transformers](https://sbert.net)
- [scikit-learn](https://scikit-learn.org)
- [TorchAudio](https://pytorch.org/audio)

A massive thank you to all the creators and maintainers of these libraries.

### Coming soon&trade;

A couple of features slated for the near-term (hopefully):
1) Sparse vector, binary vector, and multi-vector support
2) Support for all relevant embedding models on Huggingface

LLM connectors _will not_ be a feature that Radient provides. Building context-aware systems around LLMs is a complex task, and not one that Radient intends to solve. Projects such as [Haystack](https://haystack.deepset.ai/) and [Llamaindex](https://www.llamaindex.ai/) are two of the many great options to consider if you're looking to extract maximum RAG performance.

Full write-up on Radient will come later, along with more sample applications, so stay tuned.

