## Multimodal RAG

We've seen an influx of powerful multimodal capabilities in many LLMs, notably [GPT-4o](https://openai.com/index/hello-gpt-4o) and [Gemini](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024). Moving forward, most of the modalities won't be "searchable" in the traditional sense - using human-labelled tags or descriptions to retrieve relevant video or audio is not a scalable solution for multimodal RAG. We need to use dense vectors as semantic representations for _all modalities of data_. If you'd like to follow along but aren't 100% familiar with RAG just yet, LlamaIndex provides an excellent yet concise [RAG overview](https://docs.llamaindex.ai/en/stable/getting_started/concepts/).

In this example, we'll vectorize audio, text, and images into the same embedding space with [ImageBind](https://imagebind.metademolab.com/), store the vectors in [Milvus](https://milvus.io) Lite, retrieve all relevant data given a query, and input multimodal data as context into Meta's [Chameleon](https://ai.meta.com/blog/meta-fair-research-new-releases/) 7B, large multimodal model (LMM).

We'll start by specifying our imports and downloading a video we'd like to perform RAG over. For this example, let's use the 2024 Google I/O Pre-Show:

```shell
pip install -U yt-dlp
yt-dlp "https://www.youtube.com/watch?v=wwk1QIDswcQ" -o ~/google_io_preshow.mp4
```

```python
from pathlib import Path
from radient import make_operator
from radient import Workflow
```

Turning this video into vectors is a multistep process that involves: 1) splitting the video into a combination of audio and visual snippets, 2) vectorizing all snippets into the same embedding space, and 3) storing these into our vector database. Radient provides a `Workflow` object to repeatably run these steps:

```python
path = str(Path.home()/"google_io_preshow.mp4")

# The `read` operator reads all files from the directory or file specified by `path`.
# The `demux` operator splits the video into audio and visual snippets at 5.0 second intervals.
# The `vectorize` operator embeds all audio snippets and frames into a common embedding space using ImageBind.
# The `store` operator stores the vectors into Milvus. If you don't specify a URI, it will use local mode by default.
read = make_operator(optype="source", method="local", task_params={"path": path})
demux = make_operator(optype="transform", method="video-demux", task_params={"interval": 5.0})
vectorize = make_operator(optype="vectorizer", method="imagebind", modality="multimodal", task_params={})
store = make_operator(optype="sink", method="milvus", task_params={"operation": "insert"})

# All of these operators are then combined into an end-to-end workflow.
insert_wf = (Workflow()
    .add(read, name="read")
    .add(demux, name="demux")
    .add(vectorize, name="vectorize")
    .add(store, name="store")
)
```

We can then run the workflow to process our video file:

```python
insert_wf()
```

If all goes well, you should see an output that looks something like this:

```
[[{'insert_count': 258, 'ids': [450857205535866880, 450857205535866881, ...
  {'insert_count': 258, 'ids': [450857205569946116, 450857205569946117, ...]]
```

This is the result of the two `insert` operations into Milvus - one for the audio vectors, and one for the image vectors.

All the data we need is now in our vector database; given some query text, we can now search nearest neighbors in multiple modalities. Searches can be done with a workflow as well:

```python
vectorize = make_operator("vectorizer", "imagebind", modality="text")
search = make_operator("sink", "milvus", task_params={"operation": "search", "output_fields": None})

search_wf = (Workflow()
    .add(vectorize, name="vectorize")
    .add(search, name="search")
)
```

The output of this workflow are the top ten results for each query. We can test this by passing a text prompt into it:

```python
prompt = "What was unusual about the coffee mug?"
search_wf(data=prompt)
```

The output should look something like this:

```
[[[{'id': 450857205535866888, 'distance': 0.27359023690223694, 'entity': {}},
   {'id': 450857205535866886, 'distance': 0.26841503381729126, 'entity': {}},
   ...]]]
```

Chameleon unfortunately accepts only text and image inputs, so we'll need to pass a few extra parameters - namely, a [top-k limit](https://milvus.io/docs/single-vector-search.md#Basic-search), [output fields](https://milvus.io/docs/single-vector-search.md#Basic-search), and a [metadata filter](https://milvus.io/docs/single-vector-search.md#Filtered-search) - before we can pass the results into Chameleon's context window. These variables are passed directly to the `search` task, which forwards them to Milvus as keyword arguments:

```python
search_vars = {
    "limit": 1,  # top-k limit
    "output_fields": ["*"],  # output fields
    "filter": 'modality like "image"',  # metadata filter
}
results = search_wf(
    extra_vars={"search": search_vars},
    data=prompts
)
results
```

The results are now exactly what we need:

```
[[[{'id': 450857205535866888,
    'distance': 0.27359023690223694,
    'entity': {'data': '/your/home/.radient/data/video_demux/b53ebb6f-6e8e-476c-8b10-7888932c9a81/frame_0008.png',
     'modality': 'image'}}]]]
```

We've now completed the indexing and retrieval portion of our multimodal RAG system; the final step is to pass the results into Chameleon. The Github repository provided by FAIR unfortunately requires that you have a fairly beefy GPU, so you may want to run this in [Google Colab](https://colab.research.google.com/) or a completely separate Python environment:

```shell
pip install -U git+https://github.com/facebookresearch/chameleon.git@main
git clone https://huggingface.co/eastwind/meta-chameleon-7b
```

```python
from chameleon.inference.chameleon import ChameleonInferenceModel
model = ChameleonInferenceModel(
    "./meta-chameleon-7b/models/7b/",
    "./meta-chameleon-7b/tokenizer/text_tokenizer.json",
    "./meta-chameleon-7b/tokenizer/vqgan.yaml",
    "./meta-chameleon-7b/tokenizer/vqgan.ckpt",
)
tokens = model.generate(
    prompt_ui=[
        {"type": "image", "value": f"file:{results[0][0][0]["entity"]["data"]}"},
        {"type": "text", "value": prompt},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ]
)
print(model.decode_text(tokens)[0])
```

Which returns something like this (YMMV, depending on the temperature that you set):

```
The unusual aspect of this coffee mug is that it is designed to resemble the logo of a well-known tech company, Apple, but with a few modifications to give it a more playful and whimsical look. The mug features a stylized apple logo, but with a bright, bold color scheme and a curved shape that gives it a more organic and playful appearance. Additionally, the mug may have additional features or embellishments, such as a colorful pattern or a fun design element, that make it stand out from a traditional coffee mug.
```

And that's it! We've successfully built a multimodal RAG system in just a few lines of code. Although we used only one video in this example, this framework is extensible to any number of videos.

For convenience, here's the full script:

```shell
pip install -U yt-dlp
pip install -U git+https://github.com/facebookresearch/chameleon.git@main
git clone https://huggingface.co/eastwind/meta-chameleon-7b
yt-dlp "https://www.youtube.com/watch?v=wwk1QIDswcQ" -o ~/google_io_preshow.mp4
```

```python
from pathlib import Path

from radient import make_operator
from radient import Workflow
from chameleon.inference.chameleon import ChameleonInferenceModel


#
# Add multimodal (visual + audio) data into our vector database.
#

path = str(Path.home()/"google_io_preshow.mp4")

read = make_operator(optype="source", method="local", task_params={"path": path})
demux = make_operator(optype="transform", method="video-demux", task_params={"interval": 5.0})
vectorize = make_operator(optype="vectorizer", method="imagebind", modality="multimodal", task_params={})
store = make_operator(optype="sink", method="milvus", task_params={"operation": "insert"})

insert_wf = (Workflow()
    .add(read, name="read")
    .add(demux, name="demux")
    .add(vectorize, name="vectorize")
    .add(store, name="store")
)

#
# With data ingestion complete, we can now create a workflow for searches.
#

vectorize = make_operator(optype="vectorizer", method="imagebind", modality="text", task_params={})
search = make_operator(optype="sink", method="milvus", task_params={"operation": "search", "output_fields": None})

search_wf = (Workflow()
    .add(vectorize, name="vectorize")
    .add(search, name="search")
)
search_vars = {
    "limit": 1,
    "output_fields": ["*"],
    "filter": 'modality like "image"',
}

#
# Perform the search and send the results to Chameleon.
#

results = search_wf(
    extra_vars={"search": search_vars},
    data=prompts
)
model = ChameleonInferenceModel(
    "./meta-chameleon-7b/models/7b/",
    "./meta-chameleon-7b/tokenizer/text_tokenizer.json",
    "./meta-chameleon-7b/tokenizer/vqgan.yaml",
    "./meta-chameleon-7b/tokenizer/vqgan.ckpt",
)
tokens = model.generate(
    prompt_ui=[
        {"type": "text", "value": prompt},
        {"type": "image", "value": f"file:{results[0][0][0]["entity"]["data"]}"},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ]
)
print(model.decode_text(tokens)[0])
```

---

A few notes and other parting words:

1. We're using Imagebind in this example because it seems to one of the more powerful multimodal embedding models circa July 2024. [AudioCLIP](https://arxiv.org/abs/2106.13043) and [UForm](https://github.com/unum-cloud/uform) are two other another options that may be interesting to play around with, although UForm doesn't support audio just yet. Either way, we'll see an influx of multimodal embedding models to pair with LMMs - I'm experimenting with a couple of training strategies myself and hope to show something soon&trade;.

2. Although it doesn't support audio modalities yet, Meta's Chameleon is still a solid option for multimodal RAG. The model is trained on a large and diverse mulitimodal (text and image) dataset, and it's been shown to perform well on many simpler tasks despite its small size. I'll update this example once there's a solid LMM that supports audio along with image and text modalities.

3. As an unstructured data ETL framework, Radient is only meant to solve the _retrieval_ portion of retrieval-augmented generation rather than be an end-to-end RAG solution. [Llamaindex](https://www.llamaindex.ai/) and [Haystack](https://haystack.deepset.ai/) are two of the many great open source options to consider if you're looking to extract maximum RAG performance.
