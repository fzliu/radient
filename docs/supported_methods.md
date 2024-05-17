## Supported methods

Below is a series of tables that lists out supported method + kwarg pairs for each modality of data in Radient.

__Audio__ [^1]

| `method` | `model_name` | Description |
| --- | --- | --- |
| `torchaudio` | `WAV2VEC2_BASE` | Wav2vec 2.0 model ("base" architecture), pre-trained on 960 hours of unlabeled audio from LibriSpeech dataset [Panayotov et al., 2015] (the combination of "train-clean-100", "train-clean-360", and "train-other-500"), not fine-tuned. |
| `torchaudio` | `WAV2VEC2_LARGE` | Wav2vec 2.0 model ("large" architecture), pre-trained on 960 hours of unlabeled audio from LibriSpeech dataset [Panayotov et al., 2015] (the combination of "train-clean-100", "train-clean-360", and "train-other-500"), not fine-tuned. |
| `torchaudio` | `WAV2VEC2_LARGE_LV60K` | Wav2vec 2.0 model ("large-lv60k" architecture), pre-trained on 60,000 hours of unlabeled audio from Libri-Light dataset [Kahn et al., 2020], not fine-tuned. |
| `torchaudio` | `WAV2VEC2_XLSR53` | Wav2vec 2.0 model ("base" architecture), pre-trained on 56,000 hours of unlabeled audio from multiple datasets ( Multilingual LibriSpeech [Pratap et al., 2020], CommonVoice [Ardila et al., 2020] and BABEL [Gales et al., 2014]), not fine-tuned. |
| `torchaudio` | `WAV2VEC2_XLSR_300M` | XLS-R model with 300 million parameters, pre-trained on 436,000 hours of unlabeled audio from multiple datasets ( Multilingual LibriSpeech [Pratap et al., 2020], CommonVoice [Ardila et al., 2020], VoxLingua107 [Valk and Alumäe, 2021], BABEL [Gales et al., 2014], and VoxPopuli [Wang et al., 2021]) in 128 languages, not fine-tuned. |
| `torchaudio` | `WAV2VEC2_XLSR_1B` | XLS-R model with 1 billion parameters, pre-trained on 436,000 hours of unlabeled audio from multiple datasets ( Multilingual LibriSpeech [Pratap et al., 2020], CommonVoice [Ardila et al., 2020], VoxLingua107 [Valk and Alumäe, 2021], BABEL [Gales et al., 2014], and VoxPopuli [Wang et al., 2021]) in 128 languages, not fine-tuned. |
| `torchaudio` | `WAV2VEC2_XLSR_2B` | XLS-R model with 2 billion parameters, pre-trained on 436,000 hours of unlabeled audio from multiple datasets ( Multilingual LibriSpeech [Pratap et al., 2020], CommonVoice [Ardila et al., 2020], VoxLingua107 [Valk and Alumäe, 2021], BABEL [Gales et al., 2014], and VoxPopuli [Wang et al., 2021]) in 128 languages, not fine-tuned. |
| `torchaudio` | `HUBERT_BASE` | HuBERT model ("base" architecture), pre-trained on 960 hours of unlabeled audio from LibriSpeech dataset [Panayotov et al., 2015] (the combination of "train-clean-100", "train-clean-360", and "train-other-500"), not fine-tuned. |
| `torchaudio` | `HUBERT_LARGE` | HuBERT model ("large" architecture), pre-trained on 60,000 hours of unlabeled audio from Libri-Light dataset [Kahn et al., 2020], not fine-tuned. |
| `torchaudio` | `HUBERT_XLARGE` | HuBERT model ("extra large" architecture), pre-trained on 60,000 hours of unlabeled audio from Libri-Light dataset [Kahn et al., 2020], not fine-tuned. |
| `torchaudio` | `WAVLM_BASE` | WavLM Base model ("base" architecture), pre-trained on 960 hours of unlabeled audio from LibriSpeech dataset [Panayotov et al., 2015], not fine-tuned. |
| `torchaudio` | `WAVLM_BASE_PLUS` | WavLM Base+ model ("base" architecture), pre-trained on 60,000 hours of Libri-Light dataset [Kahn et al., 2020], 10,000 hours of GigaSpeech [Chen et al., 2021], and 24,000 hours of VoxPopuli [Wang et al., 2021], not fine-tuned. |
| `torchaudio` | `WAVLM_LARGE` |  WavLM Large model ("large" architecture), pre-trained on 60,000 hours of Libri-Light dataset [Kahn et al., 2020], 10,000 hours of GigaSpeech [Chen et al., 2021], and 24,000 hours of VoxPopuli [Wang et al., 2021], not fine-tuned. |

__Graph__

| `method` | `dimension` | Description |
| --- | --- | --- |
| `fastrp` | any positive integer | The FastRP (Fast Random Projection) algorithm is an efficient method for node embedding in graphs, utilizing random projections to reduce dimensionality while approximately preserving pairwise distances among nodes. |

__Image__

| `method` | `model_name` | Description |
| --- | --- | --- |
| `timm` | any model in `timm.list_models(pretrained=True)` |  |

__Molecule__

| `method` | `fingerprint_type` | Description |
| --- | --- | --- |
| `rdkit` | `topological` | Topological fingerprints represent molecules by encoding the presence or absence of particular substructures and patterns of connectivity within the molecule, focusing on the molecule's structural topology without considering the three-dimensional layout. |
| `rdkit` | `morgan` | Morgan fingerprints characterize the molecular structure based on the connectivity of atoms within a defined radius around each atom, capturing the local chemical environment in a more detailed way than simple topological features. |

__Text__

| `method` | `model_name_or_path` | Description |
| --- | --- | --- |
| `sentence-transformers` | any pretrained [Sentence Transformers model](https://huggingface.co/models?library=sentence-transformers) | |

---

[^1]: [Torchaudio documentation](https://pytorch.org/audio/stable/pipelines.html)
