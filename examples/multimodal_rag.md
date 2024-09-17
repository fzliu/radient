## Multimodal RAG (with Meta Chameleon 7B)

We've seen an influx of powerful multimodal capabilities in many LLMs, notably [GPT-4o](https://openai.com/index/hello-gpt-4o) and [Gemini](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024). Moving forward, most of the modalities won't be "searchable" in the traditional sense - using human-labelled tags or descriptions to retrieve relevant video or audio is not a scalable solution for multimodal RAG. We need to use dense vectors as semantic representations for _all modalities of data_.

In this example, we'll vectorize audio, text, and images into the same embedding space with [ImageBind](https://imagebind.metademolab.com/), store the vectors in [Milvus Lite](https://milvus.io/docs/milvus_lite.md), retrieve all relevant data given a query, and input multimodal data as context into [Chameleon](https://ai.meta.com/blog/meta-fair-research-new-releases/)-7B (vision/language model).

If you'd like to follow along but aren't 100% familiar with RAG just yet, LlamaIndex provides an excellent yet concise [RAG overview](https://docs.llamaindex.ai/en/stable/getting_started/concepts/).

<div align="center">
  <img src="https://gist.githubusercontent.com/fzliu/9011b2bcac8e1d8a0689aa339c135b37/raw/ff0d4b15c244abaea05cc422d062cc02daa7e95a/multimodal_rag_diagram.svg">
  <p style="text-align:center"><sub>Multimodal RAG using Radient.</sub></p>
</div>

We'll start by specifying our imports. We'll use `radient` to build a video vectorization workflow and `transformers` to run Chameleon:

```shell
pip install -U radient
pip install -U transformers
```

```python
from radient import make_operator
from radient import Workflow
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
from PIL import Image
```

We're going to use the 2024 Google I/O Pre-Show as the video for this example (linked via the image below). Prior to taking the stage, musician Marc Rebillet climbed out of a human-sized coffee mug plcaed to the side of the stage, and began using Google's MusicFX DJ to create AI-generated beats and tunes as a part of his performance. The video is a great example of a rich, multimodal piece of unstructured data which we can use to perform multimodal RAG:

<div align="center">
  <a href="https://www.youtube.com/watch?v=wwk1QIDswcQ"><img src="https://img.youtube.com/vi/wwk1QIDswcQ/0.jpg"></a>
</div>

Turning this video into vectors is a multistep process that involves: 1) splitting the video into a combination of audio and visual snippets, 2) vectorizing all snippets into the same embedding space, and 3) storing these into our vector database. Radient provides a `Workflow` object to repeatably run these steps:

```python
# The `read` operator grabs a video or playlist from Youtube and stores it locally.
# The `demux` operator splits the video into audio and visual snippets at 5.0 second intervals.
# The `vectorize` operator embeds all audio snippets and frames into a common embedding space using ImageBind.
# The `store` operator stores the vectors into Milvus. If you don't specify a URI, it will use local mode by default.
read = make_operator(task_name="source", task_type="youtube", task_params={"url": "https://www.youtube.com/watch?v=wwk1QIDswcQ"})
demux = make_operator(task_name="transform", task_type="video-demux", task_params={"method": "ffmpeg", "interval": 5.0})
vectorize = make_operator(task_name="vectorizer", task_type="multimodal", task_params={"method": "imagebind"})
store = make_operator(task_name="sink", task_type="milvus", task_params={"operation": "insert"})

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
vectorize = make_operator("vectorizer", "text", task_params={"method": "imagebind"})
search = make_operator("sink", "milvus", task_params={"operation": "search", "output_fields": None})

search_wf = (Workflow()
    .add(vectorize, name="vectorize")
    .add(search, name="search")
)
```

The output of this workflow are the top ten results for each query. We can test this by passing a text prompt into it:

```python
prompt = "What was weird about the coffee mug?"
search_wf(data=prompt)
```

The output should look something like this:

```
[[[{'id': 450857205535866888, 'distance': 0.27359023690223694, 'entity': {}},
   {'id': 450857205535866886, 'distance': 0.26841503381729126, 'entity': {}},
   ...]]]
```

We'll need to pass a few extra parameters - namely, a [top-k limit](https://milvus.io/docs/single-vector-search.md#Basic-search) and [output fields](https://milvus.io/docs/single-vector-search.md#Basic-search) - before we can pass the results into Chameleon's context window. These variables are passed directly to the `search` task, which forwards them to Milvus as keyword arguments:

```python
search_vars = {
    "limit": 1,  # top-k limit
    "output_fields": ["*"]  # output fields
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
    'entity': {'data': '/your/home/.radient/data/video_demux/b53ebb6f-6e8e-476c-8b10-7888932c9a81/frame_0006.png',
     'modality': 'image'}}]]]
```

Here's what the data stored in the returned entity (`frame_0006.png`) looks like:

<div align="center">
  <img src="https://lh3.googleusercontent.com/d/1rT4OdPoWZgHoXmxb1YZie1v_928UBXhy">
  <p style="text-align:center"><sub>Most relevant context retrieved with the prompt "<strong>What was weird about the coffee mug?</strong>"</sub></p>
</div>

We've now completed the indexing and retrieval portion of our multimodal RAG system; the final step is to pass the results into Chameleon. We can do this by loading the tokenizer and model, then generating text based on the prompt and image:

```python
processor = ChameleonProcessor.from_pretrained("nopperl/chameleon-7b-hf")
model = ChameleonForConditionalGeneration.from_pretrained("nopperl/chameleon-7b-hf", torch_dtype=torch.bfloat16, device_map="cpu")

image = Image.open(results[0][0][0]["entity"]["data"])
prompt = f"{prompt}<image>"

inputs = processor(prompt, image, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
generated_text = processor.batch_decode(out, skip_special_tokens=False)[0]
print(generated_text)
```

Which returns something like this (YMMV, depending on the temperature that you set):

```
The coffee mug was weirder because of the person in the image.
```

And that's it! We've successfully built a multimodal RAG system in just a few lines of code. Although we used only one video in this example, this framework is extensible to any number of videos.

This example is available on [Google Colab](https://colab.research.google.com/drive/1Z13NffkMpGjipBSExhsxQuqo28gL9VpF). For convenience, here's the full script:

```shell
pip install -U radient
pip install -U transformers
```

```python
from radient import make_operator
from radient import Workflow
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
from PIL import Image

#
# Add multimodal (visual + audio) data into our vector database.
#

read = make_operator(optype="source", method="youtube", task_params={"url": "https://www.youtube.com/watch?v=wwk1QIDswcQ"})
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

processor = ChameleonProcessor.from_pretrained("nopperl/chameleon-7b-hf")
model = ChameleonForConditionalGeneration.from_pretrained("nopperl/chameleon-7b-hf", device_map="cpu")

image = Image.open(results[0][0][0]["entity"]["data"])
prompt = f"{prompt}<image>"

inputs = processor(prompt, image, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
generated_text = processor.batch_decode(out, skip_special_tokens=False)[0]

#
# Print our result.
#
print(generated_text)
```

---

A few notes and other parting words:

1. We're using Imagebind in this example because it seems to one of the more powerful multimodal embedding models circa July 2024. [AudioCLIP](https://arxiv.org/abs/2106.13043) and [UForm](https://github.com/unum-cloud/uform) are two other another options that may be interesting to play around with, although UForm doesn't support audio just yet. Either way, we'll see an influx of multimodal embedding models to pair with LMMs - I'm experimenting with a couple of training strategies myself and hope to show something soon&trade;.

2. Although it doesn't support audio modalities yet, Meta's Chameleon is still a solid option for multimodal RAG. The model is trained on a large and diverse mulitimodal (text and image) dataset, and it's been shown to perform well on many simpler tasks despite its small size. I'll update this example once there's a solid LMM that supports audio along with image and text modalities.

3. As an unstructured data ETL framework, Radient is only meant to solve the _retrieval_ portion of retrieval-augmented generation rather than be an end-to-end RAG solution. [Llamaindex](https://www.llamaindex.ai/) and [Haystack](https://haystack.deepset.ai/) are two of the many great open source options to consider if you're looking to extract maximum RAG performance.
