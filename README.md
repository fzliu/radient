# Radient

Radient is a developer-friendly, lightweight library for _vectorization_, i.e. turning data into embeddings. Radient supports many data types, not just text.

```shell
$ pip install radient
```

### Why Radient?

In applications that leverage [RAG](https://zilliz.com/use-cases/llm-retrieval-augmented-generation), vector databases are commonly used as a way to retrieve relevant content that is relevant to the query. It's become so popular that "traditional" database vendors are rushing to support vector search. (Anybody see those [funky Singlestore ads](https://media.licdn.com/dms/image/D4E22AQE0uXihwNGBjQ/feedshare-shrink_2048_1536/0/1710685199486?e=2147483647&v=beta&t=t50JyZHIazYLQ_eVXbFtQpyhegiRiZEdxJjK0xBNLUo) on US-101?)

Although still predominantly used for text today, vectors will be used extensively across a variety of different modalities in the upcoming months. This evolution is being powered by two independent occurrences: 1) the shift from large language models to large _multimodal_ models (such as [Reka](https://www.reka.ai) and [Fuyu](https://www.adept.ai/blog/adept-fuyu-heavy)), and 2) the rise in adoption for "traditional" tasks such as recommendation and semantic search. In short, vectors are going mainstream, and we need a way to vectorize _everything_, not just text.

### Getting started

Vectorization can be performed as follows:

```python
>>> from radient import text_vectorizer
>>> vectorizer = text_vectorizer()
>>> vectorizer.vectorize("Hello, world!")
Vector([-3.21440510e-02, -5.10351397e-02,  3.69579718e-02, ...
```

You're not limited to text modalities. Audio, graphs, images, and molecules can be vectorized as well:

```python
>>> from pathlib import Path
>>> from radient import audio_vectorizer, molecule_vectorizer
>>> audio_vectorizer().vectorize(str(Path.home() / "audio.wav"))
Vector([-5.26519306e-03, -4.55586426e-03,  1.79212391e-02, ...
>>> molecule_vectorizer().vectorize("O=C=O")  # O=C=O == SMILES string for CO2
Vector([False, False, False, ...
```

You can attach metadata to the resulting embeddings and store them in sinks. Radient currently supports [Milvus](https://milvus.io):

```python
>>> vector = vectorizer.vectorize("My name is Slim Shady")
>>> vector.add_key_value("artist", "Eminem")  # {"artist": "Eminem"}
>>> vector.store()
```

For production use cases with large quantities of data, performance is key. Radient provides an `accelerate` function to optimize some vectorizers on-the-fly:

```python
>>> vectorizer.vectorize("Hello, world!")  # runtime: ~32ms
Vector([-3.21440510e-02, -5.10351397e-02,  3.69579718e-02, ...
>>> vectorizer.accelerate()
>>> vectorizer.vectorize("Hello, world!")  # runtime: ~17ms
Vector([-3.21440622e-02, -5.10351285e-02,  3.69579904e-02, ...
```

Full write-up on Radient will come later, along with some sample applications.

### Supported libraries

Radient builds atop work from the broader ML community. Most vectorizers come from other libraries:

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

