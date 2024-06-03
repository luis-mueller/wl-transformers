# Aligning Transformers with Weisfeiler-Leman

[![pytorch](https://img.shields.io/badge/PyTorch_2.1.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![pyg](https://img.shields.io/badge/PyG_2.4+-3C2179?logo=pyg&logoColor=#3C2179)](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3.2-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

Code to reproduce the results in our ICML 2024 paper.

## Install
We recommend to use the package manager [`conda`](https://docs.conda.io/en/latest/). Once installed run
```bash
conda create -n wl-transformers python=3.10
conda activate wl-transformers
```
Install the repository with all dependencies via
```bash
pip install -e .
```

## Configuration
We use [`hydra`](https://hydra.cc) for configuring experiments. See [here](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) for a tutorial on the `hydra` override syntax.

> NOTE: By default, logging with `wandb` is disabled. To enable it set `wandb_project` in the command line. Optionally, set `wandb_entity` and `wandb_name` to configure your entity and run name, respectively.

## Data and Checkpoints
We store all experiment-related data under a root directory `root`, which is the working directory by default and will be created if it does not already exist. There, if not already present, we create two additional folders: `data` and `ckpt` for the datasets and checkpoints, respectively. You may configure `root` via the command line or by changing the corresponding experiment config.

## PCQM4Mv2
For the pre-training on PCQM4Mv2, run
```bash
python pre-training/pcqm4m.py pe_type=[LPE|SPE] checkpoint=name
```
where `pe_type` specifies the PE to use and `name` specifies the name of a model checkpoint that will be created under `{root}/ckpt/{name}.pt`. If the checkpoint already exists, training will continue from that checkpoint. If `checkpoint` is not specified, checkpointing is disabled.

To pre-train on multiple GPUs, run
```bash
torchrun --standalone --nproc_per_node=num_gpus pre-training/pcqm4m.py pe_type=[LPE|SPE] checkpoint=name
```
where `num_gpus` is the number of GPUs. The per-device batch size will be automatically set to `batch_size /  {num_gpus}`.

## Node classification
For training on node classification datasets, run
```bash
python pre-training/node.py pe_type=[LPE|SPE] dataset_name=CS
```
and
```bash
python pre-training/node.py pe_type=[LPE|SPE] dataset_name=Photo degree_dim=16
```
for training on `CS` and `Photo`, respectively.

## Molecular regression
To run fine-tuning on the Alchemy dataset, run
```bash
python fine-tuning/alchemy.py pe_type=[LPE|SPE] checkpoint=name
```
where `name` specifies the name of a model checkpoint that exists under `{root}/ckpt/{name}.pt`. If `checkpoint` is not specified, the model will be trained from scratch. Set `order=3` for *order transfer* to $(3,1)\textrm{-}\textsf{WL}$.

## Molecular classification
To run the fine-tuning on one of the molecular classification datasets, run
```bash
python fine-tuning/ogbg.py backbone=backbone dataset_name=dataset_name pe_type=[LPE|SPE] checkpoint=name
```
where `backbone` specifies whether to use a `transformer` or a `gnn`, `name` specifies the name of a model checkpoint that exists under `{root}/ckpt/{name}.pt` and `dataset_name` is one of `ogbg-molbbbp`, `ogbg-molbace`, `ogbg-molclintox`, `ogbg-moltox21`, `ogbg-moltoxcast`.

## Expressivity benchmark
To run the experiments on the BREC dataset, run
```bash
python expressivity/brec_benchmark.py backbone=backbone pe_type=[LPE|SPE]
```

## Documentation of config parameters
The following table documents the most common config parameters and explains their use.

|Parameter|Description|
|-|-|
|`root`|Path to the root directory. Data will be downloaded, pre-processed and permanently stored there. By default Checkpoints are written to `root/ckpts` if `checkpoint` is set (see below).|
|`checkpoint`|Set to enable checkpointing. Checkpoints are created if a new best validation score is reached. Validation performance is measured after `val_after` batches (see below).|
|`seed`|Random seed used for the pre-training.|
|`lr`|Maximum learning rate after warm-up.|
|`min_lr`|Minimum learning rate after learning rate decay.|
|`weight_decay`|Weight decay used with `AdamW`.|
|`gradient_norm`|Maximum norm for gradient clipping.|
|`batch_size`|Batch size. When trained with multiple GPUs this value is divided by the number of GPUs before being passed to the data loader (leave this unchanged if you use more GPUs).|
|`num_steps`|Total number of batches seen during training (depending on the dataset we may use `num_epochs` instead, see below).|
|`num_warmup_steps`|Number of batches seen before reaching `lr`.|
|`num_epochs`|Total number of epochs during training (depending on the dataset we may use `num_steps` instead, see above).|
|`num_workers`|Number of parallel workers in the data loader. Hint: Start this with `0` and increase until training becomes slower.|
|`log_after`|Number of batches before logging a windowed average of training performance. Only used if `wandb_project` is enabled (see below).|
|`val_after`|Number of batches before evaluating validation performance, possibly checkpointing and logging validation performance. Regardless of whether `wandb_project` is enabled, validation performance is always logged to the command line.|
|`large_graph`|If set to `True`, uses smallest and largest eigenvalues in equal parts instead of smallest eigenvalues, as proposed for large graphs by [Kim et al. 2022](https://arxiv.org/abs/2207.02505).|
|`cosine_lr_schedule`|Whether to use the cosine learning rate scheduler. If set to `False` uses constant learning rate.|
|`wandb_project`|W&B project. If set, `wandb` logging is enabled. Otherwise performance is logged to the command line.|
|`wandb_entity`|W&B entity (optional)|
|`wandb_name`|W&B name (optional)|

The following table documents all model-specific parameters and explains their use. For an example see `pre-training/configs/model/pcqm4m.yaml`. You can copy `pre-training/configs/model/pcqm4m.yaml` and e.g., change to `pre-training/configs/model/my-model.yaml`. Then invoke your model-specific parameters by running

```bash
python pre-training/pcqm4m.py model=my-model ...
```

> NOTE: Yes, it actually works that easily 🙌 Anyway, here are the most important parameters:

|Parameter|Description|
|-|-|
|`order`|Order $k$ of the $(k,1)$-GT implemented in this repository.|
|`pe_type`|Whether to use `LPE` or `SPE` as node-level PEs.|
|`num_vecs`|Number of eigenvalues/-vectors used for LPE/SPE.|
|`normalized_laplacian`|Whether to compute the eigenvalues/-vectors from the normalized graph Laplacian.|
|`normalize_vecs`|Whether to normalize the eigenvectors to unit-norm.|
|`max_degree`|The maximum degree to consider (needs to be fixed as we embed the degree with `torch.nn.Embedding`).|
|`embed_dim`|General embedding dimension of the transformer.|
|`eigen_dim`|Embedding dimension of LPE/SPE.|
|`inner_dim`|Inner dimension of LPE/SPE. For LPE this refers to the number of eigenvectors used, for SPE this can be set arbitrarily.|
|`degree_dim`|Embedding dimension of degree encodings.|
|`num_layers`|Number of transformer layers.|
|`num_heads`|Number of transformer heads.|
|`attention_dropout`|Dropout during attention computation.|
|`dropout`|Dropout after attention computation.|
|`linear_attention`|When set to `True` uses `Performer` attention instead of self-attention.|
|`bias`|Enable biases in all `torch.nn.Linear` modules.|
|`stochastic_depth`|Enable stochastic depth as described in [Kim et al. 2022](https://arxiv.org/abs/2207.02505).|
|`token_ln`|Apply layer-norm to the input tokens.|

Here, we additionally detail parameters exclusive to SPE (see [Huang et al., 2023](https://arxiv.org/abs/2310.02579) for details on the definition of SPE):
|Parameter|Description|
|-|-|
|`num_phi_layers`|Number of $\phi$ MLP layers.|
|`num_rho_layers`|Number of $\rho$ GNN layers.|
|`phi_dim`|Hidden dimension of $\phi$ MLPs.|
|`spe_lower_rank`|SPE low-rank $m$ according to Equation (6) in our paper.|

