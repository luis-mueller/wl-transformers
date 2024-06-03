import hydra
import torch
import wandb
from torch_geometric.datasets import TUDataset
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader
from loguru import logger
from wl_transformers import (
    ensure_root_folder,
    configure_optimizers,
    transform_dataset,
    load_checkpoint,
    CosineWithWarmupLR,
    Transform,
    create_transformer,
)


def load_dataset(root: str):
    infile = open("fine-tuning/train_al_10.index", "r")
    for line in infile:
        indices_train = line.split(",")
        indices_train = [int(i) for i in indices_train]

    infile = open("fine-tuning/val_al_10.index", "r")
    for line in infile:
        indices_val = line.split(",")
        indices_val = [int(i) for i in indices_val]

    infile = open("fine-tuning/test_al_10.index", "r")
    for line in infile:
        indices_test = line.split(",")
        indices_test = [int(i) for i in indices_test]

    indices = indices_train
    indices.extend(indices_val)
    indices.extend(indices_test)

    return TUDataset(f"{root}/alchemy", name="alchemy_full")[indices]


@hydra.main(version_base=None, config_path="configs", config_name="alchemy")
def main(cfg):
    data_dir, ckpt_dir = ensure_root_folder(cfg.root)

    if cfg.wandb_project is not None:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            dir=cfg.root,
            config=dict(cfg),
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Accelerator: {device}")

    seed_everything(cfg.seed)
    logger.info(f"Random seed: {cfg.seed} 🎲")

    dataset = load_dataset(data_dir)

    transform = Transform(
        cfg.num_vecs, cfg.order, cfg.normalized_laplacian, cfg.normalize_eigenvecs
    )

    logger.info(f"Pre-transforming dataset")
    dataset = transform_dataset(dataset, transform)

    mean = dataset.y.mean(dim=0, keepdim=True)
    std = dataset.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.y - mean) / std
    mean, std = mean.to(device), std.to(device)

    train_dataset = dataset[0:10000]
    val_dataset = dataset[10000:11000]
    test_dataset = dataset[11000:]

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)

    atom_encoder = torch.nn.Linear(6, cfg.embed_dim)
    bond_encoder = torch.nn.Linear(4, cfg.embed_dim)

    logger.info("Creating transformer")
    model = create_transformer(
        cfg, atom_encoder, bond_encoder, output_dim=12, device=device
    )

    if cfg.checkpoint is not None:
        checkpoint_file = f"{ckpt_dir}/{cfg.checkpoint}.pt"
        load_checkpoint(
            checkpoint_file, model, keep_encoders=False, keep_tokenizer=cfg.order == 2
        )

    logger.info(model)

    optimizer = configure_optimizers(
        model,
        cfg.weight_decay,
        cfg.lr,
        (0.9, 0.95),
        device,
    )
    logger.info(f"Optimizer with lr {cfg.lr}, weight decay {cfg.weight_decay} ready")

    warmup_epochs = int(0.025 * cfg.num_epochs)
    scheduler = CosineWithWarmupLR(
        optimizer,
        int(0.025 * cfg.num_epochs),
        lr=cfg.lr,
        lr_decay_iters=cfg.num_epochs,
        min_lr=0,
    )
    logger.info(f"Cosine lr scheduler with {warmup_epochs} warm-up epochs ready")

    scaler = torch.cuda.amp.GradScaler()

    def train():
        model.train()
        loss_all = 0

        lf = torch.nn.L1Loss()
        for data in train_loader:
            data = data.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                loss = lf(model(data), data.y)
            scaler.scale(loss).backward()
            if cfg.gradient_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.gradient_norm
                )
            loss_all += loss.item() * data.num_graphs
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        return loss_all / len(train_loader.dataset)

    @torch.no_grad()
    def test(loader):
        model.eval()
        error = torch.zeros([1, 12]).to(device)

        for data in loader:
            data = data.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                error += ((data.y * std - model(data) * std).abs() / std).sum(dim=0)

        error = error / len(loader.dataset)
        return error.mean().item()

    best_val_error = None
    logger.info(f"Starting fine-tuning for {cfg.num_epochs} epochs 🚀")
    for epoch in range(cfg.num_epochs):
        scheduler.step()
        lr = scheduler.optimizer.param_groups[0]["lr"]
        loss = train()
        val_error = test(val_loader)

        if best_val_error is None or val_error <= best_val_error:
            test_error = test(test_loader)
            best_val_error = val_error

        if cfg.wandb_project is not None:
            wandb.log(
                {
                    "lr": lr,
                    "loss": loss,
                    "val_error": val_error,
                    "test_error": test_error,
                }
            )

        logger.info(
            f"Epoch: {epoch} LR: {lr:.5f} Loss: {loss:.5f} Val. error {val_error:.5f} Test error {test_error:.5f}"
        )

    logger.info(f"Training complete 🥳")

    if cfg.wandb_project is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
