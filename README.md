# Human V Machine Language Classifier
## EN 705.641 NLP Final Project
### Jose Montalvo Ferreiro, Muhammad Khan, Joe Mfonfu, Madihah Shaik

[Editable Link Presentation Slides](https://docs.google.com/presentation/d/1B-RdgZzLafXHr7mEVduqDNgqa0cGlSDPfSTU-bc7OqM/edit?usp=sharing)

## About Project
In this project, we trained a feedforward neural network to distinguish between human-written and machine-genrated text by using a hybrid approach that integrated semantic and stylic features of the input text. This approach is similar to the techniques discussed in [\[1\]](https://openreview.net/pdf?id=cWiEN1plhJ) and [\[2\]](https://arxiv.org/pdf/2505.14608). The semantic features are gotten from embeddings generated using a [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) model that we further finetuned for the task of human-machine text classification using the [Human Vs. LLM Text Corpus](https://www.kaggle.com/datasets/starblasters8/human-vs-llm-text-corpus?resource=download) dataset. The style features were extracted as embeddings in our data preprocessing notebook using the [StyleDistance](https://huggingface.co/StyleDistance/styledistance) model that was trained to separate semantics from style for applications like authorship detection. After concatenating the embeddings from the finetuned BERT model and those from StyleDistance, we used this as input for training a zero-shot 3-layer feed-forward neural network (or muli-layer perceptron) that can delienate human and machine text with a >96% accuracy.

high-level aim of the package
demo how to use key features
graphic of architecture diagram

description of api, pytest
