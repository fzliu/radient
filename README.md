# Radient

Radient turns _many data types_ - not just text - into vectors for similarity search.

## Overview 

Powered by the popularity of ChatGPT and other large autoregressive language models, we've seen a huge surge in interest for vector search in 2023. In applications that leverage [RAG](https://zilliz.com/use-cases/llm-retrieval-augmented-generation), vector search is commonly used as a way to retrieve relevant document chucks and other short-form text. The applicability of text-based RAG has led to widespread adoption, from single-person tech startups to Fortune 500 fintech companies.

Although primarily used for text today, vector search will be used extensively across a variety of different modalities in the near future. This evolution is being powered by two independent occurrences: 1) the shift from large language models to large _multimodal_ models, and 2) the rise in adoption for "traditional" tasks such as recommendation and semantic search. This library aims to close the gap by providing an easy way to vectorize a variety of different types of unstructured data.

Radient currently includes models that vectorize the following modalities: audio, graph, image, molecule, text, and video. In some cases, the same vectorization method can be used across multiple modalities. This is known as _multimodal vectorization_.

### Getting started

Vectorization can be performed as follows:

```python
>>> from radient import text_vectorizer
>>> vectorizer = text_vectorizer()
>>> vectorizer.vectorize(["Hello, world!"])
[Vector([-3.21440510e-02, -5.10351397e-02,  3.69579718e-02,
...
```

The return class `Vector` is a subclass of `np.ndarray`, so you can easily perform mathematical operations such as cosine similarity:

```python
>>> import numpy as np
>>> vectors = vectorizer.vectorize(["Hello, world!", "My name is Slim Shady."])
>>> np.dot(vectors[0], vectors[1])
0.51627934
```

Some vectorizers can be accelerated:

```python
>>> vectorizer.vectorize(["Hello, world!"])  # runtime: ~32ms
[Vector([-3.21440510e-02, -5.10351397e-02,  3.69579718e-02,
>>> vectorizer.accelerate()
>>> vectorizer.vectorize(["Hello, world!"])  # runtime: ~17ms
[Vector([-3.21440622e-02, -5.10351285e-02,  3.69579904e-02,
...
```

For neural network vectorizers, this is typically done by exporting the model to ONNX, which essentically converts the model into a directed graph and performs a series of optimizations (such as constant folding).


### Coming soon &trade;

A couple of features slated for the near-term (hopefully):
- A _preprocessing module_ that transforms the input data prior to vectorization
- Data _extractors_ from S3, Google Drive, Dropbox, etc and _readers_ for full end-to-end anydata ETL
- _Vectorization-as-a-service_ using [BentoML](https://github.com/bentoml/BentoML) or some other similar framework
