# Radient

Radient is a developer-friendly, lightweight library for _vectorization_, i.e. turning data into embeddings. Radient supports simple vectorization as well as complex vector-centric workflows.

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
vz.vectorize("Hello, world!")
```

    Vector([-3.21440510e-02, -5.10351397e-02,  3.69579718e-02, ...])

The above snippet vectorizes the string `"Hello, world!"` using a default model, namely `bge-small-en-v1.5` from `sentence-transformers`. If your Python environment does not contain the `sentence-transformers` library, Radient will prompt you for it:

```python
vz = text_vectorizer()
```

    Vectorizer requires sentence-transformers. Install? [Y/n]

You can type "Y" to have Radient install it for you automatically.

Each vectorizer can take a `method` parameter along with optional keyword arguments which get passed directly to the underlying vectorization library. For example, we can pick Mixbread AI's `mxbai-embed-large-v1` model using the `sentence-transformers` library via:

```python
vz_mbai = text_vectorizer(method="sentence-transformers", model_name_or_path="mixedbread-ai/mxbai-embed-large-v1")
vz_mbai.vectorize("Hello, world!")
```

    Vector([ 0.01729078,  0.04468533,  0.00055427, ...])


### More than just text

With Radient, you're not limited to text. Audio, graphs, images, and molecules can be vectorized as well:

```python
from radient import audio_vectorizer
audio_vectorizer().vectorize(str(Path.home() / "audio.wav"))
```

    Vector([-5.26519306e-03, -4.55586426e-03,  1.79212391e-02, ...])  


```python
from radient import graph_vectorizer
graph_vectorizer().vectorize(nx.karate_club_graph())
```

    [Vector([ 2.16479279e-01, -2.39208999e-02, -4.14670670e-02, ...]),
     Vector([ 2.29488305e-01, -2.78161774e-02, -3.32570679e-02, ...]),
     ...
     Vector([ 0.04171451,  0.19261454, -0.05810466,])]


```python
from radient import image_vectorizer
image_vectorizer().vectorize(str(Path.home() / "image.jpg"))
```

    Vector([0.00898108, 0.02274677, 0.00100744, ...])


```python
from radient import molecule_vectorizer
molecule_vectorizer().vectorize("O=C=O")  # O=C=O == SMILES string for CO2
```

    Vector([False, False, False, ...])


A partial list of methods and optional kwargs supported by each modality can be found [here](https://github.com/fzliu/radient/blob/main/docs/supported_methods.md).

For production use cases with large quantities of data, performance is key. Radient also provides an `accelerate` function to optimize vectorizers on-the-fly:

```python
vz = text_vectorizer()
vz.vectorize("Hello, world!")  # runtime: ~32ms
```

    Vector([-3.21440510e-02, -5.10351397e-02,  3.69579718e-02, ...])


```python
vz.accelerate()
vz.vectorize("Hello, world!")  # runtime: ~17ms
```

    Vector([-3.21440622e-02, -5.10351285e-02,  3.69579904e-02, ...])


### Building vector-centric workflows

Aside from running experiments, pure vectorization is not particularly useful. Real-world workloads often require a combination of four components:

- A data source (or a data reader) for extracting unstructured data,
- One or more transform modules such as video demuxing or OCR,
- A vectorizer or set of vectorizers, and
- A place to store the vectors once they have been computed.

Radient provides a `Workflow` object specifically for building vector-centric applications. With Workflows, you can combine any number of each of these components into a graph. For example, to store a video as audio and frame/image vectors in Milvus, we can use the following Workflow:

```python
from radient import Workflow, LocalRunner
from radient import transform, vectorizer, sink
wf = (
    Workflow()
    .add(LocalRunner(transform, task_kwargs={"method": "video_demux"}))
    .add(LocalRunner(vectorizer, task_kwargs={"modality": "multimodal", "method": "imagebind"}))
    .add(LocalRunner(sink, task_kwargs={"datastore": "milvus"}, flatten_inputs=True))
)
wf("/path/to/video.mp4")
```

Let's decompose the example above:

1) `Workflow`: All workflows consist of a series of tasks using the `add` method, which automatically assigns the previously added task as a dependency.
2) `LocalRunner`: All tasks must be wrapped in a `Runner`, which defines how the tasks are executed. Currently, only local and lazy/on-demand local runners are supported, but more runners will be added in the future (e.g. remote, GPU/TPU, etc).
3) `transform`: The first task we add is an _unstructured data transform task_ which performs video demultiplexing. This built-in task splits the video into audio snippets and image frames with a default window of 2s.
4) `vectorizer`: The second task _vectorizes_ data output by the video demux task using FAIR's ImageBind model. We use ImageBind in this example because it handles multiple data modalities (audio, images, and text).
5) `sink`: The third and final task reads the resulting vectors and _stores them in the specified datastore_ (Milvus). The `flatten_inputs` argument tells the runner to unroll dicts and lists of vectors before sending them into Milvus.

With this, we can now execute the workflow. This example uses the video at `/path/to/video.mp4`. Once execution is finished, you can verify that 

```python
from radient import text_vectorizer
vz = text_vectorizer(method="imagebind")
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

