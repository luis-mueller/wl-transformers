import hydra
import torch
import wandb
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader
from loguru import logger
from wl_transformers import (
    CosineWithWarmupLR,
    ensure_root_folder,
    configure_optimizers,
    load_checkpoint,
    Transform,
    create_transformer,
    create_gnn,
)


@torch.no_grad()
def evaluation(evaluator, model, loader, device, tdtype, metric):
    y_pred = []
    y_true = []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type=device, dtype=tdtype, enabled=True):
            logits = model(batch)
        y_true.append(batch.y.view(logits.shape).detach().cpu())
        y_pred.append(logits.detach().cpu())
    return evaluator.eval(dict(y_true=torch.cat(y_true), y_pred=torch.cat(y_pred)))[
        metric
    ]


@hydra.main(version_base=None, config_path="./configs", config_name="ogbg")
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

    dtype = "float16"
    logger.info(f"Data type: {dtype}")
    tdtype = torch.float16

    seed_everything(cfg.seed)
    logger.info(f"Random seed: {cfg.seed} 🎲")

    transform = Transform(
        cfg.num_vecs, cfg.order, cfg.normalized_laplacian, cfg.normalize_eigenvecs
    )

    dataset = PygGraphPropPredDataset(
        name=cfg.dataset_name,
        root=data_dir,
        transform=transform,
    )
    logger.info(f"Dataset {cfg.dataset_name} has {len(dataset)} molecules")

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(
        dataset[split_idx["train"]], batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset[split_idx["valid"]], batch_size=cfg.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]], batch_size=cfg.batch_size, shuffle=True
    )

    evaluator = Evaluator(cfg.dataset_name)
    metric = evaluator.eval_metric
    assert metric == "rocauc"

    val_after = len(dataset[split_idx["train"]]) // cfg.batch_size
    log_after = val_after // 10
    num_steps = val_after * cfg.num_epochs

    atom_encoder = AtomEncoder(emb_dim=cfg.embed_dim)
    bond_encoder = BondEncoder(emb_dim=cfg.embed_dim)

    if cfg.backbone == "transformer":
        logger.info("Creating transformer")
        model = create_transformer(
            cfg, atom_encoder, bond_encoder, output_dim=dataset.num_tasks, device=device
        )
    elif cfg.backbone == "gnn":
        logger.info("Creating GNN")
        model = create_gnn(
            cfg,
            atom_encoder,
            bond_encoder,
            dataset.num_tasks,
            device,
        )
    else:
        raise ValueError(f"Backbone {cfg.backbone} not supported")

    if cfg.checkpoint is not None:
        checkpoint_file = f"{ckpt_dir}/{cfg.checkpoint}.pt"
        load_checkpoint(checkpoint_file, model, keep_encoders=True)

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
        warmup_epochs * val_after,
        lr=cfg.lr,
        lr_decay_iters=num_steps,
        min_lr=0,
    )
    logger.info(f"Cosine lr scheduler with {warmup_epochs} warm-up epochs ready")

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    logger.info(
        f"Logging after {log_after}, validating after {val_after} steps. Training for {num_steps} steps 🚀"
    )

    step = 1
    best_val_score = None
    test_score = None
    while step <= num_steps:
        model.train()
        loss_window = []
        for batch in train_loader:
            batch = batch.to(device)
            with torch.autocast(device_type=device, dtype=tdtype, enabled=True):
                logits = model(batch)
                is_labeled = batch.y == batch.y
                loss = F.binary_cross_entropy_with_logits(
                    logits[is_labeled], batch.y.to(tdtype)[is_labeled]
                )
            scaler.scale(loss).backward()
            if cfg.gradient_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.gradient_norm
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if step % int(log_after) == 0:
                if len(loss_window) == 0:
                    logger.warning("No loss data available")
                elif cfg.wandb_project is not None:
                    wandb.log(
                        dict(
                            train_loss=sum(loss_window) / len(loss_window),
                            lr=optimizer.param_groups[0]["lr"],
                        ),
                        step=step,
                    )
                loss_window = []
            else:
                loss_window.append(float(loss.detach().cpu()))

            if step % int(val_after) == 0:
                logger.info("Evaluating model")
                model.eval()
                val_score = evaluation(
                    evaluator, model, val_loader, device, tdtype, metric
                )
                if best_val_score is None or val_score > best_val_score:
                    best_val_score = val_score
                    test_score = evaluation(
                        evaluator, model, test_loader, device, tdtype, metric
                    )

                logger.info(
                    dict(
                        step=step,
                        val_score=val_score,
                        best_val_score=best_val_score,
                        test_score=test_score,
                    )
                )
                if cfg.wandb_project is not None:
                    wandb.log(
                        dict(
                            val_score=val_score,
                            best_val_score=best_val_score,
                            test_score=test_score,
                        ),
                        step=step,
                    )
                model.train()

            step += 1

    logger.info(f"Training complete 🥳")

    if cfg.wandb_project is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
