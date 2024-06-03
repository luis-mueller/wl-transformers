import hydra
import torch
import wandb
import torch.nn.functional as F
from torch_geometric.datasets import (
    Coauthor,
    Amazon,
)
from torch_geometric.seed import seed_everything
from loguru import logger
from wl_transformers import (
    ensure_root_folder,
    CosineWithWarmupLR,
    Transform,
    create_transformer,
)


@hydra.main(version_base=None, config_path="configs", config_name="node")
def main(cfg):
    data_dir, _ = ensure_root_folder(cfg.root)

    if cfg.wandb_project is not None:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            config=dict(cfg),
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Accelerator: {device}")

    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    logger.info(f"Data type: {dtype}")
    tdtype = torch.float16 if dtype == "float16" else torch.bfloat16

    seed_everything(cfg.seed)
    logger.info(f"Random seed: {cfg.seed} 🎲")

    transform = Transform(
        cfg.num_vecs,
        cfg.order,
        cfg.normalized_laplacian,
        cfg.normalize_eigenvecs,
        large_graph=cfg.large_graph,
    )

    if cfg.dataset_name == "CS":
        dataset = Coauthor(data_dir, "CS", pre_transform=transform)
    elif cfg.dataset_name in "Photo":
        dataset = Amazon(data_dir, "Photo", pre_transform=transform)

    num_nodes = dataset[0].x.size(0)
    idx = torch.randperm(num_nodes)
    labels = dataset[0].y[idx]

    train_nodes = torch.cat([idx[labels == i][60:] for i in range(dataset.num_classes)])
    val_nodes = torch.cat([idx[labels == i][:30] for i in range(dataset.num_classes)])
    test_nodes = torch.cat(
        [idx[labels == i][30:60] for i in range(dataset.num_classes)]
    )

    train_mask = torch.zeros(num_nodes, dtype=bool, device=device)
    train_mask[train_nodes] = True

    val_mask = torch.zeros(num_nodes, dtype=bool, device=device)
    val_mask[val_nodes] = True

    test_mask = torch.zeros(num_nodes, dtype=bool, device=device)
    test_mask[test_nodes] = True

    atom_encoder = torch.nn.Linear(dataset.num_features, cfg.embed_dim)

    if dataset.num_edge_features == 0:
        bond_encoder = torch.nn.Linear(1, cfg.embed_dim)
    else:
        bond_encoder = torch.nn.Linear(dataset.num_edge_features, cfg.embed_dim)

    logger.info("Creating transformer")
    model = create_transformer(
        cfg,
        atom_encoder,
        bond_encoder,
        output_dim=dataset.num_classes,
        device=device,
        head_type="node",
    )
    logger.info(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), cfg.lr, weight_decay=cfg.weight_decay
    )
    logger.info(f"Optimizer with lr {cfg.lr}, weight decay {cfg.weight_decay} ready")

    if cfg.cosine_lr_schedule:
        warmup_epochs = int(0.025 * cfg.num_epochs)
        scheduler = CosineWithWarmupLR(
            optimizer,
            int(0.025 * cfg.num_epochs),
            lr=cfg.lr,
            lr_decay_iters=cfg.num_epochs,
            min_lr=0,
        )
        logger.info(f"Cosine lr scheduler with {warmup_epochs} warm-up epochs ready")

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    def train():
        model.train()
        data = dataset[0].to(device)

        with torch.autocast(device_type=device, dtype=tdtype, enabled=True):
            logits = model(data)
            loss = F.cross_entropy(logits[train_mask], data.y[train_mask])

        scaler.scale(loss).backward()
        if cfg.gradient_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.gradient_norm
            )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        return loss.item()

    @torch.no_grad()
    def test(split):
        model.eval()
        data = dataset[0].to(device)
        idx = val_mask if split == "val" else test_mask

        with torch.autocast(device_type=device, dtype=tdtype, enabled=True):
            logits = model(data)
            acc = (
                (logits[idx].softmax(-1).argmax(-1) == data.y[idx])
                .to(torch.float)
                .mean()
            )

        return acc.item()

    best_val_acc = None
    logger.info(f"Starting fine-tuning for {cfg.num_epochs} epochs 🚀")
    for epoch in range(cfg.num_epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss = train()
        if cfg.cosine_lr_schedule:
            scheduler.step()
        val_acc = test("val")

        if best_val_acc is None or val_acc >= best_val_acc:
            test_acc = test("test")
            best_val_acc = val_acc

        if cfg.wandb_project is not None:
            wandb.log(
                {
                    "lr": lr,
                    "loss": loss,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "best_val_acc": best_val_acc,
                }
            )

        logger.info(
            f"Epoch: {epoch} LR: {lr:.5f} Loss: {loss:.5f} Val. acc {val_acc:.5f} Test acc {test_acc:.5f}"
        )

    logger.info(f"Training complete 🥳")

    if cfg.wandb_project is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
