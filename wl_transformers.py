import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, add_self_loops, scatter
from torch_geometric.nn import global_mean_pool, global_add_pool, GINEConv, GINConv
from performer_pytorch import SelfAttention
from torchvision.ops import stochastic_depth
import math
import os
from loguru import logger
from prettytable import PrettyTable
import inspect
import tqdm


def transform_dataset(dataset, transform):
    data_list = []
    for data in tqdm.tqdm(dataset, miniters=len(dataset) / 50):
        data_list.append(transform(data))
    data_list = list(filter(None, data_list))
    dataset._indices = None
    dataset._data_list = data_list
    dataset._data, dataset.slices = dataset.collate(data_list)
    return dataset


def configure_optimizers(
    model: torch.nn.Module,
    weight_decay,
    learning_rate,
    betas,
    device_type,
):
    """Adapted from https://github.com/karpathy/nanoGPT"""
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [
        p
        for n, p in param_dict.items()
        if p.dim() >= 2  # and (n.startswith("tokenizer.") or n.startswith("mlp."))
    ]
    nodecay_params = [
        p
        for n, p in param_dict.items()
        if p.dim() < 2  # and (n.startswith("tokenizer.") or n.startswith("mlp."))
    ]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    logger.info(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()

    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    logger.info(f"using fused AdamW: {use_fused}")

    return optimizer


def count_parameters(model: torch.nn.Module):
    """Source: https://stackoverflow.com/a/62508086"""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    logger.info(f"\n{str(table)}")
    return total_params


def ensure_root_folder(root, master_process=True):
    if not os.path.exists(root) and master_process:
        logger.info(f"Creating root directory {root}")
        os.makedirs(root)

    if not os.path.exists(data_dir := f"{root}/data") and master_process:
        logger.info(f"Creating data directory {data_dir}")
        os.makedirs(data_dir)

    if not os.path.exists(ckpt_dir := f"{root}/ckpt") and master_process:
        logger.info(f"Creating ckpt directory {ckpt_dir}")
        os.makedirs(ckpt_dir)

    return data_dir, ckpt_dir


def save_checkpoint(checkpoint_file, step, model, optimizer, scaler, best_val_score):
    logger.info(f"Creating and saving checkpoint to {checkpoint_file}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "step": step,
            "best_val_score": best_val_score,
        },
        checkpoint_file,
    )


def prepare_checkpoint_for_fine_tuning(
    model, checkpoint, keep_encoders, keep_tokenizer
):
    fine_tuning_state_dict = model.state_dict()
    model_state_dict = checkpoint["model_state_dict"]

    model_state_dict_keys = list(model_state_dict.keys())
    for state_dict_key in model_state_dict_keys:
        if (
            not keep_encoders
            and (
                state_dict_key.startswith("tokenizer.atom_encoder")
                or state_dict_key.startswith("tokenizer.bond_encoder")
            )
        ) or (not keep_tokenizer and state_dict_key.startswith("tokenizer.")):
            del model_state_dict[state_dict_key]

        if state_dict_key.startswith("mlp."):
            if state_dict_key in fine_tuning_state_dict:
                model_state_dict[state_dict_key] = fine_tuning_state_dict[
                    state_dict_key
                ]
            else:
                del model_state_dict[state_dict_key]

    for state_dict_key in fine_tuning_state_dict.keys():
        if (
            not keep_encoders
            and (
                state_dict_key.startswith("tokenizer.atom_encoder")
                or state_dict_key.startswith("tokenizer.bond_encoder")
            )
        ) or (not keep_tokenizer and state_dict_key.startswith("tokenizer.")):
            model_state_dict[state_dict_key] = fine_tuning_state_dict[state_dict_key]

    checkpoint["model_state_dict"] = model_state_dict
    return checkpoint


def load_checkpoint(checkpoint_file, model, keep_encoders=False, keep_tokenizer=True):
    if os.path.exists(checkpoint_file):
        logger.info(f"Loading pre-trained checkpoint from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        checkpoint = prepare_checkpoint_for_fine_tuning(
            model, checkpoint, keep_encoders, keep_tokenizer
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            "Model weights loaded for fine-tuning (initialized tokenizer and mlp) 💫"
        )
    else:
        raise ValueError(f"Could not find checkpoint {checkpoint_file}")


def continue_from_checkpoint(checkpoint_file, model, optimizer, scaler, device_id=None):
    if os.path.exists(checkpoint_file):
        logger.info(f"Loading pre-trained checkpoint from {checkpoint_file}")
        load_args = (
            dict(map_location=f"cuda:{device_id}") if torch.cuda.is_available() else {}
        )
        checkpoint = torch.load(checkpoint_file, **load_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        return checkpoint["step"] + 1, checkpoint["best_val_score"]
    else:
        logger.info(
            f"Could not find checkpoint {checkpoint_file}, starting training from scratch"
        )
        return 1, None


class CosineWithWarmupLR:
    """Adapted from https://github.com/karpathy/nanoGPT"""

    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        lr: float,
        lr_decay_iters: int,
        min_lr: float,
        epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr = lr
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.epoch = epoch
        self.step()

    def step(self):
        self.epoch += 1
        lr = self._get_lr(self.epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self, epoch: int):
        # 1) linear warmup for warmup_iters steps
        if epoch < self.warmup_iters:
            return self.lr * epoch / self.warmup_iters
        # 2) if epoch > lr_decay_iters, return min learning rate
        if epoch > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (epoch - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.lr - self.min_lr)


def compute_laplacian_eigen(
    edge_index,
    num_nodes,
    max_freq,
    normalized=False,
    normalize=False,
    large_graph=False,
):
    A = torch.zeros((num_nodes, num_nodes))
    A[edge_index[0], edge_index[1]] = 1

    if normalized:
        D12 = torch.diag(A.sum(1).clip(1) ** -0.5)
        I = torch.eye(A.size(0))
        L = I - D12 @ A @ D12
    else:
        D = torch.diag(A.sum(1))
        L = D - A
    eigvals, eigvecs = torch.linalg.eigh(L)

    if large_graph:
        idx1 = torch.argsort(eigvals)[: max_freq // 2]
        idx2 = torch.argsort(eigvals, descending=True)[: max_freq // 2]
        idx = torch.cat([idx1, idx2])
    else:
        idx = torch.argsort(eigvals)[:max_freq]

    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    eigvals = torch.real(eigvals).clamp_min(0)

    if normalize:
        eignorm: torch.Tensor = eigvecs.norm(p=2, dim=0, keepdim=True)
        eigvecs = eigvecs / eignorm.clamp_min(1e-12).expand_as(eigvecs)

    if num_nodes < max_freq:
        eigvals = F.pad(eigvals, (0, max_freq - num_nodes), value=float("nan"))
        eigvecs = F.pad(eigvecs, (0, max_freq - num_nodes), value=float("nan"))
    eigvals = eigvals.unsqueeze(0).repeat(num_nodes, 1)
    return eigvals, eigvecs


def compute_degrees(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    adj[edge_index[0], edge_index[1]] = 1
    return adj.sum(-1)


def compute_token_index(edge_index, num_nodes, order):
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    assert edge_index.size(1) > 0
    if order == 2:
        return edge_index
    elif order == 3:
        nodes = torch.arange(num_nodes)
        edges: torch.Tensor = edge_index.T
        repeated_edges = edges.repeat_interleave(num_nodes, 0)  # E * N x 2
        repeated_nodes = nodes.repeat(len(edges))  # E * N x 1

        candidate_tuples = torch.cat(
            [repeated_edges, repeated_nodes.unsqueeze(-1)], dim=-1
        )  # E * N x 3
        candidate_masks = []

        for begin, end in [(0, 2), (1, 2)]:
            mask = (
                (candidate_tuples[:, (begin, end)].unsqueeze(1) == edges)
                .all(-1)
                .nonzero()[:, 0]
            )
            candidate_masks.append(mask)

        idx = torch.cat(candidate_masks).unique()

        token_index = torch.cat(
            [
                candidate_tuples[idx][:, (0, 1, 2)],
                candidate_tuples[idx][:, (0, 2, 1)],
                candidate_tuples[idx][:, (2, 0, 1)],
            ]
        ).T

        token_index = torch.cat(
            [
                token_index,
                torch.cat([edge_index, edge_index[1].unsqueeze(0)]),
                torch.cat([edge_index[0].unsqueeze(0), edge_index]),
            ],
            -1,
        )

        return token_index
    else:
        raise ValueError(f"Order > 3 is not yet supported.")


def compute_real_edges(edge_index, num_nodes):
    _, attr = add_self_loops(
        edge_index, torch.ones(edge_index.size(1)), fill_value=0, num_nodes=num_nodes
    )
    return attr.to(bool)


class Transform:
    def __init__(
        self,
        num_vecs,
        order,
        normalized_laplacian,
        normalize_eigenvecs,
        large_graph=False,
    ):
        self.num_vecs = num_vecs
        self.order = order
        self.normalized = normalized_laplacian
        self.normalize = normalize_eigenvecs
        self.large_graph = large_graph

    def __call__(self, data):
        data.eigvals, data.eigvecs = compute_laplacian_eigen(
            data.edge_index,
            data.num_nodes,
            self.num_vecs,
            self.normalized,
            self.normalize,
            self.large_graph,
        )
        data.degrees = compute_degrees(data.edge_index, data.num_nodes)
        data.token_index = compute_token_index(
            data.edge_index, data.num_nodes, self.order
        )
        data.real_edges = compute_real_edges(data.edge_index, data.num_nodes)
        return data


class PaddedEigenEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, bias):
        super().__init__()
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(2, 2 * embed_dim, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 2, embed_dim, bias=bias),
            torch.nn.ReLU(),
        )

    def forward(self, eigvecs, eigvals):
        x = torch.stack((eigvecs, eigvals), 2)
        empty_mask = torch.isnan(x)
        x[empty_mask] = 0
        return self.ffn(x)


class EigenEmbedding(torch.nn.Module):
    def __init__(
        self, num_vecs, eigen_dim, inner_dim=None, bias=True, position_aware=False
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = num_vecs
        self.num_vecs = num_vecs
        self.position_aware = position_aware
        if position_aware:
            self.eps = torch.nn.Parameter(1e-12 * torch.arange(inner_dim).unsqueeze(0))

        self.phi = PaddedEigenEmbedding(eigen_dim, bias)
        self.rho = torch.nn.Sequential(
            torch.nn.Linear((eigen_dim), 2 * eigen_dim, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * eigen_dim, eigen_dim, bias=bias),
            torch.nn.ReLU(),
        )

    def forward(self, eigvals, eigvecs, edge_index, batch):
        eigvecs = eigvecs[:, : self.num_vecs]
        eigvals = eigvals[:, : self.num_vecs]

        if self.training:
            sign_flip = torch.rand(eigvecs.size(1), device=eigvecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            eigvecs = eigvecs * sign_flip.unsqueeze(0)

        if self.position_aware:
            eigvals = eigvals + self.eps[:, : self.num_vecs]

        eigen_embed = self.phi(eigvecs, eigvals)
        return self.rho(eigen_embed.sum(1))


def nan_to_zero(mat):
    empty_mask = torch.isnan(mat)
    mat[empty_mask] = 0
    return mat


class EigenMLPLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.in_proj = torch.nn.Linear(in_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.bn(x.reshape(-1, x.size(-1))).reshape(x.size())
        return self.dropout(F.relu(x))


class EigenMLP(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, out_dim, dropout):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                EigenMLPLayer(1 if i == 0 else embed_dim, embed_dim, dropout)
                for i in range(num_layers)
            ]
        )
        self.out_proj = torch.nn.Linear(embed_dim, out_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out_proj(x)
        return self.dropout(x)


class EigenGIN(torch.nn.Module):
    def __init__(self, num_layers, in_dim, embed_dim, pe_dim):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_dim if _ == 0 else embed_dim, embed_dim),
                        torch.nn.ReLU(),
                    ),
                    node_dim=0,
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.out_proj = torch.nn.Linear(embed_dim, pe_dim)

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            z = layer(x, edge_index)
            if i > 0:
                x = z + x
            else:
                x = z
        return self.out_proj(x)


class EquivariantEigenEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_phi_layers,
        num_rho_layers,
        num_vecs,
        eigen_dim,
        phi_dim,
        inner_dim,
        pe_dim,
        lower_rank=None,
    ):
        super().__init__()
        self.num_vecs = num_vecs
        self.lower_rank = lower_rank
        self.psi = EigenMLP(num_phi_layers, phi_dim, inner_dim, dropout=0.0)
        self.gin = EigenGIN(num_rho_layers, inner_dim, eigen_dim, pe_dim)

    def forward(self, eigvals, eigvecs, edge_index, batch):
        eigvals = nan_to_zero(eigvals[:, : self.num_vecs])  # N_sum, M
        eigvecs = nan_to_zero(eigvecs[:, : self.num_vecs])  # N_sum, M

        eigvals, _ = to_dense_batch(eigvals, batch)  # B, N, M
        eigvals = eigvals[:, 0, :].unsqueeze(-1)  # B, M, 1
        eigvals = self.psi(eigvals)  # B, M, N_psi

        eigvecs, mask = to_dense_batch(eigvecs, batch)  # B, N, M

        if self.lower_rank is not None:
            outer_eigvecs = eigvecs[:, : self.lower_rank]
        else:
            outer_eigvecs = eigvecs

        eigen_tensor = torch.einsum(
            "bijk,blj->bilk",
            eigvecs.unsqueeze(-1) * eigvals.unsqueeze(1),  # B, N, M, N_psi
            outer_eigvecs,
        )  # B, N, LN, N_psi
        pe = eigen_tensor[mask]  # N_sum, LN, N_psi
        pe = self.gin(pe, edge_index).sum(1)
        return pe


class PairEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.encoder = torch.nn.Linear(2 * in_dim, out_dim, bias=bias)

    def forward(self, x, token_index):
        return self.encoder(torch.cat([x[token_index[0]], x[token_index[1]]], -1))


class ConcatEncoder(torch.nn.Module):
    def __init__(self, num_in, in_dim, out_dim, bias=True):
        super().__init__()
        self.encoder = torch.nn.Linear(num_in * in_dim, out_dim, bias=bias)

    def forward(self, x, token_index):
        return self.encoder(
            torch.cat([x[token_index[i]] for i in range(token_index.size(0))], -1)
        )


class PairTokenizer(torch.nn.Module):
    def __init__(
        self,
        atom_encoder,
        bond_encoder,
        num_phi_layers,
        num_rho_layers,
        phi_dim,
        num_vecs,
        max_degree,
        eigen_dim,
        inner_dim,
        degree_dim,
        embed_dim,
        bias,
        pe_type,
        spe_lower_rank,
    ):
        super().__init__()
        if pe_type == "LPE":
            self.eigen_embed = EigenEmbedding(
                num_vecs, eigen_dim, inner_dim, bias, True
            )
        elif pe_type == "SPE":
            self.eigen_embed = EquivariantEigenEmbedding(
                num_phi_layers,
                num_rho_layers,
                num_vecs,
                eigen_dim,
                phi_dim,
                inner_dim,
                eigen_dim,
                spe_lower_rank,
            )
        else:
            raise ValueError(f"PE type {pe_type} is not supported")
        self.eigen_encoder = PairEncoder(eigen_dim, embed_dim, bias)

        if degree_dim > 0:
            self.degree_embed = torch.nn.Embedding(max_degree, degree_dim)
            self.degree_encoder = PairEncoder(degree_dim, embed_dim, bias)
        self.atom_encoder = atom_encoder
        self.node_encoder = PairEncoder(embed_dim, embed_dim, bias)
        self.bond_encoder = bond_encoder
        self.graph_token = torch.nn.Parameter(torch.zeros((1, 1, embed_dim)))

    def forward(self, data):
        token_index, real_edges = data.token_index, data.real_edges
        x = self.atom_encoder(data.x)

        if hasattr(self, "degree_embed"):
            d = self.degree_embed(data.degrees)
        l = self.eigen_embed(data.eigvals, data.eigvecs, data.edge_index, data.batch)

        e = torch.zeros(
            (token_index.size(1), x.size(1)), device=x.device, dtype=x.dtype
        )
        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            data.edge_attr = torch.ones(
                (data.edge_index.size(1), 1), device=data.x.device, dtype=data.x.dtype
            )
        e[real_edges] = self.bond_encoder(data.edge_attr)

        token_embed = (
            e + self.eigen_encoder(l, token_index) + self.node_encoder(x, token_index)
        )

        if hasattr(self, "degree_encoder"):
            token_embed = token_embed + self.degree_encoder(d, token_index)

        if not hasattr(data, "batch") or data.batch is None:
            batch = torch.zeros(data.x.size(0), device=data.x.device, dtype=torch.long)[
                token_index[0]
            ]
        else:
            batch = data.batch[token_index[0]]
        token_embed, mask = to_dense_batch(token_embed, batch)

        graph_token = self.graph_token.repeat(token_embed.size(0), 1, 1)
        token_embed = torch.cat([graph_token, token_embed], 1)

        padding_mask = torch.zeros(
            (mask.size(0), mask.size(1) + 1), dtype=mask.dtype, device=mask.device
        )
        padding_mask[:, 0] = 1
        padding_mask[:, 1:] = mask

        return token_embed, padding_mask


def distribute_edge_features(edge_index, edge_attr, token_index):
    order = token_index.size(0)
    distributed_edge_features = torch.zeros(
        (token_index.size(1), edge_attr.size(1)), device=token_index.device
    )

    for i in range(order * (order - 1) // 2):
        # NOTE: Example k = 3: (0, 1), (0, 2), (1, 2)
        begin = max(((i + 1) - (order - 1), 0))
        end = min(((i + 1), order - 1))

        index_match: torch.Tensor = (
            token_index[(begin, end), :].T.unsqueeze(-1) == edge_index
        )
        idx = index_match.all(1).nonzero().T
        distributed_edge_features[idx[0]] += edge_attr[idx[1]]

    return distributed_edge_features


class TripletTokenizer(torch.nn.Module):
    def __init__(
        self,
        atom_encoder,
        bond_encoder,
        num_phi_layers,
        num_rho_layers,
        phi_dim,
        num_vecs,
        max_degree,
        eigen_dim,
        inner_dim,
        degree_dim,
        embed_dim,
        bias,
        pe_type,
        spe_lower_rank,
    ):
        super().__init__()
        if pe_type == "LPE":
            self.eigen_embed = EigenEmbedding(
                num_vecs, eigen_dim, inner_dim, bias, True
            )
        elif pe_type == "SPE":
            self.eigen_embed = EquivariantEigenEmbedding(
                num_phi_layers,
                num_rho_layers,
                num_vecs,
                eigen_dim,
                phi_dim,
                inner_dim,
                eigen_dim,
                spe_lower_rank,
            )
        else:
            raise ValueError(f"PE type {pe_type} is not supported")
        self.eigen_encoder = ConcatEncoder(3, eigen_dim, embed_dim, bias)
        self.degree_embed = torch.nn.Embedding(max_degree, degree_dim)
        self.degree_encoder = ConcatEncoder(3, degree_dim, embed_dim, bias)
        self.atom_encoder = atom_encoder
        self.node_encoder = ConcatEncoder(3, embed_dim, embed_dim, bias)
        self.bond_encoder = bond_encoder
        self.graph_token = torch.nn.Parameter(torch.zeros((1, 1, embed_dim)))

    def forward(self, data):
        token_index, real_edges = data.token_index, data.real_edges
        x = self.atom_encoder(data.x)
        d = self.degree_embed(data.degrees)
        l = self.eigen_embed(data.eigvals, data.eigvecs, data.edge_index, data.batch)

        e = self.bond_encoder(data.edge_attr)
        e = distribute_edge_features(data.edge_index, e, token_index)

        token_embed = (
            e
            + self.eigen_encoder(l, token_index)
            + self.degree_encoder(d, token_index)
            + self.node_encoder(x, token_index)
        )

        batch = data.batch[token_index[0]]
        token_embed, mask = to_dense_batch(token_embed, batch)

        graph_token = self.graph_token.repeat(token_embed.size(0), 1, 1)
        token_embed = torch.cat([graph_token, token_embed], 1)

        padding_mask = torch.zeros(
            (mask.size(0), mask.size(1) + 1), dtype=mask.dtype, device=mask.device
        )
        padding_mask[:, 0] = 1
        padding_mask[:, 1:] = mask

        return token_embed, padding_mask


class LayerNorm(torch.nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(ndim))
        self.bias = torch.nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class FastSelfAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout,
        bias=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.c_attn = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.c_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.n_head = num_heads
        self.n_embd = embed_dim
        self.dropout = dropout

    def forward(self, x, key_padding_mask):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=~key_padding_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return [y]


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
        linear_attention: bool = False,
        bias: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.linear_attention = linear_attention

        self.ln_1 = LayerNorm(embed_dim, bias)
        self.ln_2 = LayerNorm(embed_dim, bias)

        if linear_attention:
            self.attention = SelfAttention(
                embed_dim, heads=num_heads, dropout=attention_dropout
            )
        else:
            self.attention = FastSelfAttention(
                embed_dim,
                num_heads,
                attention_dropout,
                bias=bias,
            )

        self.attention_dropout = torch.nn.Dropout(dropout)
        self.stochastic_depth = lambda x: (
            stochastic_depth(x, drop_path, "batch", self.training)
            if drop_path > 0
            else x
        )

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim, bias=bias),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim, embed_dim, bias=bias),
        )

    def forward(self, x, mask=None):
        _x = x
        x = self.ln_1(x)

        if self.linear_attention:
            x = self.attention(x)
        else:
            x = self.attention(x, key_padding_mask=~mask)[0]

        x = self.attention_dropout(x)
        x = self.stochastic_depth(x)

        x = _x + x
        return x + self.stochastic_depth(self.ffn(self.ln_2(x)))


def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def create_transformer(
    cfg, atom_encoder, bond_encoder, output_dim, device, head_type="graph"
):
    model = Transformer(
        order=cfg.order,
        atom_encoder=atom_encoder,
        bond_encoder=bond_encoder,
        num_phi_layers=cfg.num_phi_layers,
        num_rho_layers=cfg.num_rho_layers,
        phi_dim=cfg.phi_dim,
        num_vecs=cfg.num_vecs,
        max_degree=cfg.max_degree,
        eigen_dim=cfg.eigen_dim,
        inner_dim=cfg.inner_dim,
        degree_dim=cfg.degree_dim,
        num_layers=cfg.num_layers,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        output_dim=output_dim,
        dropout=cfg.dropout,
        attention_dropout=cfg.attention_dropout,
        linear_attention=cfg.linear_attention,
        bias=cfg.bias,
        stochastic_depth=cfg.stochastic_depth,
        token_ln=cfg.token_ln,
        pe_type=cfg.pe_type,
        spe_lower_rank=cfg.spe_lower_rank,
        head_type=head_type,
    ).to(device)
    model.reset_parameters()
    return model


class Transformer(torch.nn.Module):
    def __init__(
        self,
        order,
        atom_encoder,
        bond_encoder,
        num_phi_layers,
        num_rho_layers,
        phi_dim,
        num_vecs,
        max_degree,
        eigen_dim,
        inner_dim,
        degree_dim,
        num_layers,
        embed_dim,
        num_heads,
        output_dim,
        dropout=0.0,
        attention_dropout=0.0,
        linear_attention=False,
        bias=True,
        stochastic_depth=False,
        token_ln=False,
        pe_type=False,
        spe_lower_rank=None,
        head_type="graph",
    ):
        super().__init__()

        if order == 2:
            self.tokenizer = PairTokenizer(
                atom_encoder,
                bond_encoder,
                num_phi_layers,
                num_rho_layers,
                phi_dim,
                num_vecs,
                max_degree,
                eigen_dim,
                inner_dim,
                degree_dim,
                embed_dim,
                bias,
                pe_type,
                spe_lower_rank,
            )
        elif order == 3:
            self.tokenizer = TripletTokenizer(
                atom_encoder,
                bond_encoder,
                num_phi_layers,
                num_rho_layers,
                phi_dim,
                num_vecs,
                max_degree,
                eigen_dim,
                inner_dim,
                degree_dim,
                embed_dim,
                bias,
                pe_type,
                spe_lower_rank,
            )
        else:
            raise ValueError(f"Tokenizer for order {order} unavailable")

        if token_ln:
            self.token_ln = LayerNorm(embed_dim, bias)

        self.layers = torch.nn.Sequential(
            *[
                TransformerLayer(
                    embed_dim,
                    num_heads,
                    dropout,
                    attention_dropout,
                    linear_attention,
                    bias,
                    drop_path=(0.1 * (i + 1) / num_layers) if stochastic_depth else 0.0,
                )
                for i in range(num_layers)
            ]
        )
        self.head_type = head_type
        if head_type == "graph":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, embed_dim, bias=bias),
                torch.nn.GELU(),
                LayerNorm(embed_dim, bias),
                torch.nn.Linear(embed_dim, output_dim, bias=bias),
            )
        elif head_type == "node":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, embed_dim, bias=bias),
                torch.nn.ReLU(),
                torch.nn.Linear(embed_dim, embed_dim, bias=bias),
                torch.nn.ReLU(),
                torch.nn.Linear(embed_dim, output_dim, bias=bias),
            )
        elif head_type == "layerwise":
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, embed_dim, bias=bias),
                torch.nn.GELU(),
                LayerNorm(embed_dim, bias),
                torch.nn.Linear(embed_dim, output_dim, bias=bias),
            )
        self.num_heads = num_heads
        self.linear_attention = linear_attention

    def forward(self, data):
        x, mask = self.tokenizer(data)

        if self.linear_attention:
            mask = None
        else:
            mask = mask.unsqueeze(1) == mask.unsqueeze(2)
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        if hasattr(self, "token_ln"):
            x = self.token_ln(x)

        for layer in self.layers:
            x = layer(x, mask)

            if self.head_type == "node":
                # NOTE: Only works with single graphs, otherwise unbatch
                z = x[:, 1:].squeeze()
                z[~data.real_edges] = (
                    z[~data.real_edges]
                    + scatter(z[data.real_edges], data.edge_index[0])
                    + scatter(z[data.real_edges], data.edge_index[1])
                )  # NOTE: Add off-diagonals with a scatter

                if hasattr(self, "token_ln"):
                    x = self.token_ln(x)

        if self.head_type == "graph":
            return self.mlp(x[:, 0, :])
        elif self.head_type == "node":
            x = x[:, 1:].squeeze()[~data.real_edges]
            return self.mlp(x)
        elif self.head_type is None:
            return x[:, 0, :]

    def reset_parameters(self):
        self.apply(init_weights)


class GINLayer(torch.nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.conv = GINEConv(torch.nn.Dropout(dropout))
        self.ffn = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim, embed_dim),
        )
        self.norm = torch.nn.BatchNorm1d(embed_dim)

    def forward(self, data):
        x = self.conv(data.x, data.edge_index, data.edge_attr)
        x = data.x + x
        data.x = self.norm(x + self.ffn(x))
        return data


def create_gnn(
    cfg,
    atom_encoder,
    bond_encoder,
    output_dim,
    device,
):
    model = GNN(
        atom_encoder=atom_encoder,
        bond_encoder=bond_encoder,
        num_phi_layers=cfg.num_phi_layers,
        num_rho_layers=cfg.num_rho_layers,
        eigen_dim=cfg.eigen_dim,
        phi_dim=cfg.phi_dim,
        num_layers=cfg.num_layers,
        embed_dim=cfg.embed_dim,
        pooling=cfg.pooling,
        output_dim=output_dim,
        num_vecs=cfg.num_vecs,
        dropout=cfg.dropout,
        pe_type=cfg.pe_type,
        spe_lower_rank=cfg.spe_lower_rank,
    ).to(device)
    model.reset_parameters()
    return model


class GNN(torch.nn.Module):
    def __init__(
        self,
        atom_encoder,
        bond_encoder,
        num_phi_layers,
        num_rho_layers,
        eigen_dim,
        phi_dim,
        num_layers,
        embed_dim,
        pooling,
        output_dim,
        num_vecs,
        dropout,
        pe_type,
        spe_lower_rank,
    ):
        super().__init__()
        self.atom_encoder = atom_encoder
        self.bond_encoder = bond_encoder
        if pe_type == "LPE":
            self.eigen_embed = EigenEmbedding(num_vecs, embed_dim, position_aware=True)
        elif pe_type == "SPE":
            self.eigen_embed = EquivariantEigenEmbedding(
                num_phi_layers,
                num_rho_layers,
                num_vecs,
                eigen_dim,
                phi_dim,
                num_vecs,
                embed_dim,
                spe_lower_rank,
            )

        self.layers = torch.nn.ModuleList(
            [GINLayer(embed_dim, dropout) for _ in range(num_layers)]
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim, output_dim),
        )
        self.pooling = pooling

    def forward(self, data):
        data.x = self.atom_encoder(data.x)
        pe = self.eigen_embed(data.eigvals, data.eigvecs, data.edge_index, data.batch)
        data.x = data.x + pe
        data.edge_attr = self.bond_encoder(data.edge_attr)
        for layer in self.layers:
            data = layer(data)

        if self.pooling == "sum":
            return self.mlp(global_add_pool(data.x, data.batch))
        elif self.pooling == "mean":
            return self.mlp(global_mean_pool(data.x, data.batch))
        else:
            raise ValueError(f"Pooling {self.pooling} not supported")

    def reset_parameters(self):
        self.apply(init_weights)
