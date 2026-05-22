import os
import gc
import torch
import wandb
import torch.distributed as dist
from pathlib import Path

from llm_training_project.training.TraningStateManager import TrainingStateManager
from llm_training_project.checkpoints_dir.checkpoint import CheckpointManager
from llm_training_project.config.train_config import LLM_training_config, update_config_paths
from llm_training_project.config.model_config import LLM_model_config
from llm_training_project.utils.HF_utils import HFUtils
from llm_training_project.training.trainer import train_on_shard
from llm_training_project.shards.ShardManager import ShardManager
from llm_training_project.dataloader.DataLoader import get_dataloader
from llm_training_project.utils.distributed import setup_distributed, cleanup_distributed
from llm_training_project.utils.wandb_interactive import should_stop_from_hf
from llm_training_project.model.model import LLM

# 🔷 Logging system
from llm_training_project.log.diagnostic_metrics import DiagnosticMetrics
from llm_training_project.log.wandb_logger import WandBLogger
from llm_training_project.log.async_wandb_logger import AsyncWandBLogger

from kaggle_secrets import UserSecretsClient


def main():

    # =========================================================
    # 🔹 CONFIG + PATH SETUP
    # =========================================================
    BASE_DIR = Path(__file__).resolve().parent

    train_config = LLM_training_config.load_from_yaml(
        str(BASE_DIR / "llm_training_project/config/configs/train_config.yaml")
    )
    train_config = update_config_paths(train_config, BASE_DIR)

    model_config = LLM_model_config.load_from_yaml(
        str(BASE_DIR / "llm_training_project/config/configs/model_config.yaml")
    )

    hf_api = HFUtils.load_config_from_yaml(
        str(BASE_DIR / "llm_training_project/config/configs/hf_config.yaml")
    )


    user_secrets = UserSecretsClient()
    hf_api.hf_token = user_secrets.get_secret("hf_token")

    # =========================================================
    # 🔹 DDP SETUP
    # =========================================================
    local_rank = setup_distributed()
    rank = dist.get_rank() if train_config.num_devices > 1 else 0
    world_size = dist.get_world_size() if train_config.num_devices > 1 else 1

    device = torch.device(f"cuda:{local_rank}")

    if world_size > 1:
        dist.barrier()

    # =========================================================
    # 🔹 CHECKPOINT DOWNLOAD (ONLY RANK 0)
    # =========================================================
    if rank == 0:
        hf_api.load_checkpoint_from_hf(train_config.checkpoint_dir)

    if world_size > 1:
        dist.barrier()

    # =========================================================
    # 🔹 CORE MANAGERS
    # =========================================================
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=train_config.checkpoint_dir,
        device=device
    )

    shard_manager = ShardManager(
        json_path=train_config.shard_manager_json_path
    )

    state_manager = TrainingStateManager(
        train_config,
        model_config,
        checkpoint=checkpoint_manager,
        device=device
    )

    model_states = state_manager.load_training_state(
        model_cls=LLM,
        local_rank=local_rank
    )

    # after the checkpoint download barrier
    val_local_path = train_config.val_local_path

    if rank == 0:
        val_file_url = train_config.val_file_url
        os.makedirs(os.path.dirname(val_local_path), exist_ok=True)
        hf_api.download_hf_file_from_url(
            file_url=val_file_url,
            local_datasetdir=os.path.dirname(val_local_path)
        )

    if world_size > 1:
        dist.barrier()

    # =========================================================
    # 🔹 METRICS + LOGGER (ONLY RANK 0)
    # =========================================================
    metrics_engine = None
    logger = None

    if rank == 0:
        metrics_engine = DiagnosticMetrics()

        base_logger = WandBLogger(
            project="LLM_195M_Training",
            run_name="exp-run",
            config={**train_config.dict(), **model_config.dict()}
        )

        # 🔥 Async wrapper
        logger = AsyncWandBLogger(base_logger)

    # =========================================================
    # 🔁 TRAINING LOOP (SHARDS)
    # =========================================================
    for _ in range(len(shard_manager.shard_files)):

        shard_url = shard_manager.get_next_shard()

        parts = shard_url.split("/")
        relative_path = os.path.join(parts[-2], parts[-1])

        file_local_path = os.path.join(
            train_config.dataset_dir,
            relative_path
        )

        # -------------------------
        # DOWNLOAD SHARD (RANK 0)
        # -------------------------
        if rank == 0:
            os.makedirs(os.path.dirname(file_local_path), exist_ok=True)
            hf_api.download_hf_file_from_url(
                file_url=shard_url,
                local_datasetdir=train_config.dataset_dir
            )

        if world_size > 1:
            dist.barrier()

        # -------------------------
        # DATALOADER
        # -------------------------
        dataloader = get_dataloader(
            shard_file=file_local_path,
            batch_size=train_config.batch_size,
            world_size=world_size,
            rank=rank if world_size > 1 else 0,
            num_workers=train_config.num_workers,
            validate=True
        )

        # =====================================================
        # 🔹 TRAIN SHARD
        # =====================================================
        shard_stats = train_on_shard(
            model=model_states["model"],
            criterion=torch.nn.CrossEntropyLoss(),
            dataloader=dataloader,
            optimizer=model_states["optimizer"],
            scheduler=model_states["scheduler"],
            device=device,
            scaler=model_states["scaler"],
            max_grad_norm=train_config.max_grad_norm,
            gradient_accumulation_steps=train_config.accumulation_steps,
            start_global_step=model_states["global_step"],
            logger=logger if rank == 0 else None,
            metrics_engine=metrics_engine if rank == 0 else None,
            rank=rank,
            dtype=train_config.dtype,
            val_local_path = val_local_path,
        )

        model_states["global_step"] = shard_stats["global_step"]

        # -------------------------
        # CLEANUP
        # -------------------------
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()

        # -------------------------
        # CHECKPOINT + SHARD MGMT
        # -------------------------
        if rank == 0:

            local_ckpt_path = checkpoint_manager.save_checkpoint(
                model=model_states["model"],
                optimizer=model_states["optimizer"],
                scheduler=model_states["scheduler"],
                scaler=model_states["scaler"],
                global_step=model_states["global_step"],
                wandb_run_id=wandb.run.id if wandb.run else None,
                name=f"checkpoint_step_{model_states['global_step']}"
            )

            hf_api.save_checkpoint_to_hf(
                checkpoint_path=local_ckpt_path,
                commit_message="ckpt_saved"
            )

            shard_manager.remove_shard()

        if world_size > 1:
            dist.barrier()

        shard_manager.reload()

        # -------------------------
        # REMOTE STOP SIGNAL
        # -------------------------
        if should_stop_from_hf(rank, world_size, device):
            break

    # =========================================================
    # 🔹 CLEANUP
    # =========================================================
    if world_size > 1:
        cleanup_distributed()

    # 🔥 Important: flush async logger
    if rank == 0 and logger is not None:
        logger.finish()


if __name__ == "__main__":
    main()