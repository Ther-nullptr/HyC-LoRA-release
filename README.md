# HyC-LoRA: Memory Efficient LoRA Fine-tuning with Hybrid Activation Compression

[[paper](xxxxxxxx)]

## Abstract

Large language models (LLMs) are widely used in applications like conversation and text summarization. With
the demand for model customization and privacy, lightweight fine-tuning methods for large models have begun to
receive widespread attention. Low-Rank Adaption (LoRA) is one of the most widely used fine-tuning algorithms,
which significantly reduces the tunable weights and associated optimizer memory when transferring pre-trained
LLMs to downstream tasks. However, past works lacked attention to the overhead of buffered activations in
low-rank adaption, leading to suboptimal system memory usage.

To reduce buffered activation memory consumption and further enable the on-device memory efficient fine-tuning
system, we propose HyC-LoRA, a variant of the LoRA training method using a hybrid compression framework
enabling almost 2-bit buffered activation quantization in all operators. HyC-LoRA observes that the temporarily
buffered activation for back-propagation dominates the memory consumption in the LoRA fine-tuning process,
and those in non-linear modules act as dominant memory consumers, whose quantization is more challenging.
Based on this, HyC-LoRA proposes a hybrid compression mechanism with two tiers: (1) Intra-operator hybrid
compression: HyC-LoRA detects extreme outliers in buffered activation and mitigates the quantization error
by structured outlier storage; (2) Inter-operator hybrid compression: HyC-LoRA utilizes the LoRA adapter
to achieve compensation for quantization errors and selective recomputation, through inter-operator reordering
and fusion. Finally, HyC-LoRA implements a buffered activation compression system and integrates it with the
existing machine learning framework to complete the last mile of lightweight storage for fine-tuning algorithms.
Evaluations with multiple LLMs such as Llama series, in widely-used downstream tasks show the proposed
HyC-LoRA framework achieves up to 3.97× end-to-end memory reduction compared to baseline, with negligible
accuracy degradation.

![pic](figures/main-intra-inter.png)

## Installation

### Algorithm Implementation

1. Install the required packages

```bash
$ git clone https://github.com/<TODO>
$ cd <TODO>
$ pip install -r requirements.txt
```

2. Run experiments on large LLMs

GSM8K (llama/mistral):

```bash
$ bash run_gsm8k.sh
```

Wikitext-2 (llama/mistral):

```bash
$ bash run_wikitext2.sh
```

Arithmetic Datasets (llama/mistral):

```bash
$ bash download_datasets.sh # prepare the datasets
$ bash run_multitask.sh
```

GLUE (RoBERTa):

```bash
$ bash run_glue.sh
```

Long Sequence (llama):

download the data from []()

```bash
$ bash run_longseq.sh
```

The core parameters can be adjusted in these script:

```bash
model_name=llama-2-7b-hf # model name
model_dir=<your model dir> # model directory

use_hyclora=True # Whether to use the specified training code
layer_type=intra_inter # Type of HyCLoRA layer [baseline(no compression), intra(optimization in single operator), intra_inter(optimization between operators), intra_inter_full_fuse(equivalent to intra_inter at the algorithmic level, but with kernel fusion for access-intensive operations such as lora, activation function, hadamard product, etc. during forward and backward passes)]
iteration_threshold=5 # Number of calibration iterations
softmax_outlier_ratio=0.05 # Outlier ratio for softmax attention map (play no role when use flash_attn)
layernorm_outlier_ratio=0.005 # Outlier channel ratio for layernorm/rmsnorm
q_bit=2 # quantization bit for buffered activation
```

## TODO

- [x]  GSM8K, Wikitext-2, Arithmetic Datasets, GLUE, and Long Sequence
- [x]  GLUE
- [x]  Long Sequence
- [ ]  Vision Transformer
- [ ]  Tutorials

## Acknowledgement

Our code is built upon the following projects:

* [LoftQ](https://github.com/yxli2123/LoftQ)
* [ApiQ](https://github.com/BaohaoLiao/ApiQ)
* [LongLoRA](https://github.com/dvlab-research/LongLoRA)
* [unsloth](https://github.com/unslothai/unsloth)

We thank the authors for their open-sourced code.

## Citation

TODO