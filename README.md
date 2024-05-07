# Radient

Radient is a developer-friendly, lightweight library for _vectorization_, i.e. turning data into embeddings. Radient supports many data types, not just text.

```shell
$ pip install radient
```

### Getting started

Vectorization can be performed as follows:

```python
>>> from radient import text_vectorizer
>>> vectorizer = text_vectorizer()
>>> vectorizer.vectorize(["Hello, world!"])
[Vector([-3.21440510e-02, -5.10351397e-02,  3.69579718e-02,
...
```

You're not limited to text modalities. Audio, graphs, images, and molecules can be vectorized as well:

```python
>>> from radient import audio_vectorizer, molecule_vectorizer
>>> av = audio_vectorizer()
```

The resulting embeddings can be stored in sinks. Radient currently supports [Milvus](https://milvus.io):

```python
```

For production use cases with large quantities of data, performance is key. Radient provides an `accelerate` function to optimize some vectorizers on-the-fly:

```python
>>> vectorizer.vectorize(["Hello, world!"])  # runtime: ~32ms
[Vector([-3.21440510e-02, -5.10351397e-02,  3.69579718e-02, ...
>>> vectorizer.accelerate()
>>> vectorizer.vectorize(["Hello, world!"])  # runtime: ~17ms
[Vector([-3.21440622e-02, -5.10351285e-02,  3.69579904e-02, ...
```

Check out the full [write-up on Radient]() for more information along with some sample applications.

### Supported libraries

Radient builds atop work from the broader ML community. Many vectorizers come from other libraries:

- Pytorch Image Models
- RDKit
- Sentence Transformers
- Scikit Learn
- Torchaudio

A massive thank you to all the creators and maintainers of these libraries.

### Coming soon&trade;

A couple of features slated for the near-term (hopefully):
- Support relevant embedding models on Huggingface (e.g. non-seq2seq models)
- A _preprocessing module_ that transforms the input data prior to vectorization
- Data _sources_ from S3/GCS/Blob, Google Drive, Box, etc and _readers_ for full end-to-end anydata ETL

