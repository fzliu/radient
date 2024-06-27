## Multimodal RAG

We've seen an influx of powerful multimodal capabilities in many LLMs, notably [GPT-4o](https://openai.com/index/hello-gpt-4o) and [Gemini](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024). Moving forward, most of the modalities won't be "searchable" in the traditional sense - using human-labelled tags or descriptions to retrieve relevant video or audio is not a scalable solution for multimodal RAG. We need to use dense vectors as semantic representations for _all modalities of data_.

In this example, we'll vectorize audio, text, and images into the same embedding space with [ImageBind](https://imagebind.metademolab.com/), store the vectors in [Milvus](https://milvus.io), retrieve all relevant data given a query, and input multimodal data as context into Meta's [Chameleon](https://ai.meta.com/blog/meta-fair-research-new-releases/) large multimodal model (LMM).

We'll start by specifying our imports and downloading a video we'd like to perform RAG over. For this example, let's use the 2024 Google I/O Pre-Show:

```shell
yt-dlp "https://www.youtube.com/watch?v=wwk1QIDswcQ" -o ~/google_io_preshow.mp4
```

We'll generate vectors for frames and audio snippets at two second intervals (using the `video-demux` transform), vectorize them (using the `imagebind` multimodalvectorizer), and store the results in our vector database (using the `milvus` datastore). Radient provides a way to build these with just a few lines of code:

```python
from radient import Workflow, LazyLocalRunner
from radient import transform, vectorizer, sink
insert_wf = (Workflow()
    .add(LazyLocalRunner(source, task_kwargs={"datasource": "local"}))
    .add(LazyLocalRunner(transform, task_kwargs={"method": "video-demux"}))
    .add(LazyLocalRunner(vectorizer, task_kwargs={"modality": "multimodal", "method": "imagebind"}))
    .add(LazyLocalRunner(sink, task_kwargs={"datastore": "milvus", "method": "insert"}))
)
```

We can then run the workflow on the video file:

```python
insert_wf("google_io_preshow.mp4")
```

If all goes well, you should see an output that looks something like this:

```
[[{'insert_count': 644, 'ids': [450597351514963968, 450597351514963969, ...
```

All the data we need is now in our vector database; given some query text, we can now search nearest neighbors in multiple modalities:

```python
search_wf = (Workflow()
    .add(LazyLocalRunner(vectorizer, task_kwargs={"modality": "text", "method": "imagebind"}))
    .add(LazyLocalRunner(sink, task_kwargs={"datastore": "milvus", "method": "search"}))
)
search_wf("")
```

Passing this into Chameleon as a part of the prompt, we get:

```python

```

And that's it! We've successfully built a multimodal RAG system in just a few lines of code.

We're using Imagebind in this example because it seems to one of the more powerful multimodal embedding models circa June 2024. [AudioCLIP](https://arxiv.org/abs/2106.13043) and [UForm](https://github.com/unum-cloud/uform) are two other another options that may be interesting to play around with. Either way, we'll see an influx of multimodal embedding models to pair with these LMMs - I'm experimenting with a couple of training strategies myself and hope to show something soon&trade;.

```python

```

Keep in mind that Radient is meant to be an ETL framework for complex, multimodal data (i.e. vector-centric), _not_ an end-to-end RAG solution; projects such as [Llamaindex](https://www.llamaindex.ai/) and [Haystack](https://haystack.deepset.ai/) are two of the many great options to consider if you're looking to extract maximum RAG performance. 

For convenience, here's the full script:

```python
```

