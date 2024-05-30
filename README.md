# Radient

Radient is a developer-friendly, lightweight library for _vectorization_, i.e. turning data into embeddings. Think of Radient as a Fivetran for multimodal data.

```shell
$ pip install radient
```

### Why Radient?

In applications that leverage [RAG](https://zilliz.com/use-cases/llm-retrieval-augmented-generation), vector databases are commonly used as a way to retrieve relevant content that is relevant to the query. It's become so popular that "traditional" database vendors are rushing to support vector search. (Anybody see those [funky Singlestore ads](https://media.licdn.com/dms/image/D4E22AQE0uXihwNGBjQ/feedshare-shrink_2048_1536/0/1710685199486?e=2147483647&v=beta&t=t50JyZHIazYLQ_eVXbFtQpyhegiRiZEdxJjK0xBNLUo) on US-101?)

Although still predominantly used for text today, vectors will be used extensively across a variety of different modalities in the upcoming months. This evolution is being powered by two independent occurrences: 1) the shift from large language models to large _multimodal_ models (such as [GPT-4o](https://openai.com/index/hello-gpt-4o), [Reka](https://www.reka.ai), and [Fuyu](https://www.adept.ai/blog/adept-fuyu-heavy)), and 2) the rise in adoption for "traditional" tasks such as recommendation and semantic search. In short, vectors are going mainstream, and we need a way to vectorize _everything_, not just text.

### Getting started

Vectorization can be performed as follows:

```python
>>> from radient import text_vectorizer
>>> vectorizer = text_vectorizer()
>>> vectorizer.vectorize("Hello, world!")
Vector([-3.21440510e-02, -5.10351397e-02,  3.69579718e-02, ...])
```

The above snippet vectorizes the string `"Hello, world!"` using a default model, namely `bge-small-en-v1.5` from `sentence-transformers`. If your Python environment does not contain the `sentence-transformers` library, Radient prompt you for it:

```python
>>> vectorizer = text_vectorizer()
Vectorizer requires sentence-transformers. Install? [Y/n]
```

You can type "Y" to have Radient install it for you automatically.

### More than just text

With Radient, you're not limited to text. Audio, graphs, images, and molecules can be vectorized as well:

```python
>>> from pathlib import Path
>>> from radient import audio_vectorizer, graph_vectorizer, image_vectorizer, molecule_vectorizer
>>> audio_vectorizer().vectorize(str(Path.home() / "audio.wav"))
Vector([-5.26519306e-03, -4.55586426e-03,  1.79212391e-02, ...])
>>> graph_vectorizer().vectorize(nx.karate_club_graph())
[Vector([ 2.16479279e-01, -2.39208999e-02, -4.14670670e-02, ...]),
 Vector([ 2.29488305e-01, -2.78161774e-02, -3.32570679e-02, ...]),
 ...
 Vector([ 0.04171451,  0.19261454, -0.05810466,])]
>>> image_vectorizer().vectorize(str(Path.home() / "image.jpg"))
Vector([0.00898108, 0.02274677, 0.00100744, ...])
>>> molecule_vectorizer().vectorize("O=C=O")  # O=C=O == SMILES string for CO2
Vector([False, False, False, ...])
```

A partial list of methods and optional kwargs supported by each modality can be found [here](https://github.com/fzliu/radient/blob/main/docs/supported_methods.md).

### Customizing vectorizers

Each vectorizer can take a `method` parameter along with optional keyword arguments which get passed directly to the underlying vectorization library. For example, we can pick a specific model from the `sentence-transformers` library using:

```python
>>> vectorizer_mbai = text_vectorizer(method="sbert", model_name_or_path="mixedbread-ai/mxbai-embed-large-v1")
>>> vectorizer_mbai.vectorize("Hello, world!")
Vector([ 0.01729078,  0.04468533,  0.00055427, ...])
```

This will use Mixbread AI's `mxbai-embed-large-v1` model to perform vectorization.

### Sources and sinks

You can attach metadata to the resulting embeddings and store them in sinks. Radient currently supports [Milvus](https://milvus.io):

```python
>>> vector = vectorizer.vectorize("My name is Slim Shady")
>>> vector.putscalar("artist", "Eminem")  # {"artist": "Eminem"}
>>> vector.store(collection_name="_radient", field_name="vector")
{'insert_count': 1, 'ids': [449662764050547785]}
```

This will store the vector in a Milvus instance at `http://localhost:19530` by default; if the specified collection does not exist at this URI, it will create it (with dynamic schema turned on for flexibility). You can change the desired Milvus instance by specifying the `milvus_uri` parameter. This works with [Zilliz Cloud](https:/zilliz.com/cloud) instances too, e.g. `vector.store(milvus_uri="https://in01-dd7f98cd6b900f6.aws-us-west-2.vectordb.zillizcloud.com:19530")`.

### Radient in production

For production use cases with large quantities of data, performance is key. Radient provides an `accelerate` function to optimize some vectorizers on-the-fly:

```python
>>> vectorizer.vectorize("Hello, world!")  # runtime: ~32ms
Vector([-3.21440510e-02, -5.10351397e-02,  3.69579718e-02, ...])
>>> vectorizer.accelerate()
>>> vectorizer.vectorize("Hello, world!")  # runtime: ~17ms
Vector([-3.21440622e-02, -5.10351285e-02,  3.69579904e-02, ...])
```

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
- Sparse, binary, and multi-vector support
- Support all relevant embedding models on Huggingface, e.g. non-seq2seq models
- Data _sources_ from object storage, Google Drive, Box, etc
- Vector _sinks_ to Zilliz, Databricks, Confluent, etc
- Creating _flows_ to tie sources, operators, vectorizers, and sinks together

LLM connectors _will not_ be a feature that Radient provides. Building context-aware systems around LLMs is a complex task, and not one that Radient intends to solve. Projects such as [Llamaindex](https://www.llamaindex.ai/) and [Haystack](https://haystack.deepset.ai/) are two of the many great options to consider if you're looking to extract maximum RAG performance. 

Full write-up on Radient will come later, along with more sample applications, so stay tuned.

