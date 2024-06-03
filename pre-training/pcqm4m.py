import os
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4MEvaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything
import hydra
import torch
import torch.nn.functional as F
from loguru import logger
import wandb
from wl_transformers import (
    ensure_root_folder,
    count_parameters,
    configure_optimizers,
    CosineWithWarmupLR,
    continue_from_checkpoint,
    save_checkpoint,
    Transform,
    create_transformer,
)
import time
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


@torch.no_grad()
def evaluate(evaluator, model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(
                -1,
            )

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


def ddp_setup():
    if torch.cuda.device_count() > 1:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def accelerator_setup():
    if torch.cuda.is_available():
        device = "cuda"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            device_id = int(os.environ["LOCAL_RANK"])
            master_process = device_id == 0
        else:
            device_id = 0
            master_process = True
    else:
        device = "cpu"
        device_id = "cpu"
        device_count = 1
        master_process = True

    return device, device_id, device_count, master_process


@hydra.main(version_base=None, config_path="./configs", config_name="pcqm4m")
def main(cfg):
    ddp_setup()

    device, device_id, device_count, master_process = accelerator_setup()
    logger.info(f"Accelerator: {device}, num. devices {device_count}")

    data_dir, ckpt_dir = ensure_root_folder(cfg.root, master_process)

    if cfg.wandb_project is not None:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            config=dict(cfg),
        )

    torch.set_float32_matmul_precision("medium")
    logger.info(f"Setting float32 matmul precision to medium")

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
        cfg.num_vecs, cfg.order, cfg.normalized_laplacian, cfg.normalize_eigenvecs
    )

    logger.info(f"Loading dataset from {data_dir}")
    dataset = PygPCQM4Mv2Dataset(root=data_dir, transform=transform)
    logger.info("Dataset loaded")

    split_idx = dataset.get_idx_split()

    if device_count > 1:
        train_loader = DataLoader(
            dataset[split_idx["train"]],
            batch_size=cfg.batch_size // device_count,
            num_workers=cfg.num_workers,
            shuffle=False,
            sampler=DistributedSampler(dataset[split_idx["train"]]),
        )
    else:
        train_loader = DataLoader(
            dataset[split_idx["train"]],
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
        )

    if master_process:
        val_loader = DataLoader(
            dataset[split_idx["valid"]],
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        evaluator = PCQM4MEvaluator()

    atom_encoder = AtomEncoder(emb_dim=cfg.embed_dim)
    bond_encoder = BondEncoder(emb_dim=cfg.embed_dim)

    logger.info("Creating transformer")
    model = create_transformer(
        cfg, atom_encoder, bond_encoder, output_dim=1, device=device_id
    )

    if master_process:
        num_params = count_parameters(model)

        if cfg.wandb_project is not None:
            wandb.log(dict(num_params=num_params))

        logger.info(model)
        logger.info(f"Number of parameters: {num_params}")

    optimizer = configure_optimizers(
        model,
        cfg.weight_decay,
        cfg.lr,
        (0.9, 0.95),
        device,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    if cfg.checkpoint is not None:
        checkpoint_file = f"{ckpt_dir}/{cfg.checkpoint}.pt"
        logger.info(f"Trying to continue from checkpoint {checkpoint_file}")
        step, best_val_score = continue_from_checkpoint(
            checkpoint_file, model, optimizer, scaler, device_id
        )
        if step > 1 and master_process:
            logger.info("Sanity-checking model performance")
            val_score = evaluate(evaluator, model, val_loader, device)
            logger.info(f"Best val score: {best_val_score}, reproduced: {val_score}")
    else:
        step, best_val_score = 1, None

    scheduler = CosineWithWarmupLR(
        optimizer, cfg.num_warmup_steps, cfg.lr, cfg.num_steps, 0.0, step - 1
    )

    if master_process:
        logger.info(f"Optimizer + scheduler with lr {cfg.lr} ready")

    if device_count > 1:
        logger.info("Creating DDP module")
        model = DDP(model, device_ids=[device_id])

    logger.info(
        f"Starting/resuming training for {int(cfg.num_steps) - (step - 1)} steps 🚀"
    )

    if master_process:
        start_time = time.time()
    if device_count > 1:
        train_loader.sampler.set_epoch(epoch := 0)
    while step <= cfg.num_steps:
        model.train()
        loss_window = []
        for batch in train_loader:
            batch = batch.to(device_id)
            with torch.autocast(device_type=device, dtype=tdtype, enabled=True):
                logits = model(batch)
                loss = F.l1_loss(logits.squeeze(), batch.y)
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

            if master_process:
                loss_window.append(float(loss.detach().cpu()))

            if step % int(cfg.log_after) == 0 and master_process:
                if cfg.wandb_project is not None:
                    wandb.log(
                        dict(
                            train_loss=sum(loss_window) / len(loss_window),
                            lr=optimizer.param_groups[0]["lr"],
                            time=time.time() - start_time,
                        ),
                        step=step,
                    )
                loss_window = []
                start_time = time.time()

            if step % int(cfg.val_after) == 0:
                logger.info(f"Completed epoch [{device_id}]")
                if device_count > 1:
                    epoch += 1
                    train_loader.sampler.set_epoch(epoch)
                if master_process:
                    logger.info("Evaluating model")
                    model.eval()
                    val_score = evaluate(evaluator, model, val_loader, device_id)
                    if best_val_score is None or val_score < best_val_score:
                        best_val_score = val_score

                        if cfg.checkpoint is not None:
                            module = model.module if device_count > 1 else model
                            save_checkpoint(
                                checkpoint_file,
                                step,
                                module,
                                optimizer,
                                scaler,
                                best_val_score,
                            )

                    logger.info(
                        dict(
                            step=step,
                            val_score=val_score,
                            best_val_score=best_val_score,
                        )
                    )
                    if cfg.wandb_project is not None:
                        wandb.log(
                            dict(val_score=val_score, best_val_score=best_val_score),
                            step=step,
                        )
                    model.train()

            step += 1

    logger.info(f"Training complete 🥳")
    if master_process:
        wandb.finish()

    if device_count > 1:
        destroy_process_group()


if __name__ == "__main__":
    main()
