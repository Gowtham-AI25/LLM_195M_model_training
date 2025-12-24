import torch 
import os
import wandb
from pathlib import Path
import torch.distributed as dist
from llm_training_project.training.TraningStateManager import TrainingStateManager
from llm_training_project.checkpoints_dir.checkpoint import CheckpointManager
from llm_training_project.config.train_config import LLM_training_config, update_config_paths
from llm_training_project.config.model_config import LLM_model_config
from llm_training_project.utils.HF_utils import HFUtils
from llm_training_project.training.trainer import train_on_shard
from llm_training_project.log.tensorboard_logger import TensorBoardLogger
from llm_training_project.shards.ShardManager import ShardManager
from llm_training_project.dataloader.DataLoader import get_dataloader
from llm_training_project.utils.distributed import setup_distributed, cleanup_distributed
from llm_training_project.utils.wandb_interactive import should_stop_from_wandb
from kaggle_secrets import UserSecretsClient



def main():
    # Detect the directory where train.py is located
    BASE_DIR = Path(__file__).resolve().parent 

    # Load the training configuration
    train_config = LLM_training_config.load_from_yaml( str(BASE_DIR / "llm_training_project/config/configs/train_config.yaml"))
    train_config = update_config_paths(train_config, BASE_DIR)  # update all paths according to base path
    # Load the model configuration
    model_config = LLM_model_config.load_from_yaml(str(BASE_DIR / "llm_training_project/config/configs/model_config.yaml"))
    # Load the Hugging Face configuration
    hf_api = HFUtils.load_config_from_yaml( str(BASE_DIR / "llm_training_project/config/configs/hf_config.yaml"))
    
    user_secrets = UserSecretsClient()
    hf_api.hf_token = user_secrets.get_secret("hf_access_token")

    local_rank = setup_distributed()
    rank = dist.get_rank() if train_config.num_devices > 1 else 0
    world_size = dist.get_world_size() if train_config.num_devices > 1 else 1
    device = torch.device(f"cuda:{local_rank}")

    # checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir = train_config.checkpoint_dir,
        device = device
    )

    # Create ShardManager to handle data shards
    shard_manager = ShardManager(
        json_path = train_config.shard_manager_json_path
    )

    # initializ TrainingStateManager to get model, optimizer, scaler, scheduler, etc.
    state_manager = TrainingStateManager(
        train_config,
        model_config,
        checkpoint = checkpoint_manager,
        device = device
    )

    # Load or create training state
    model_states = state_manager.load_training_state(model_cls=LLM, local_rank=local_rank)

    tb_logger = None
    if rank == 0:
        # W&B logger
        resume_id = model_states.get("wandb_run_id")
        # Initialize W&B
        wandb.init(
            project="LLM_195M_Training",
            id=resume_id, 
            resume="allow" if resume_id else None,
            config={**train_config.dict(), **model_config.dict()}
        )
        
        # tensorboard logger
        tb_logger = TensorBoardLogger(
            log_dir = train_config.tensorboard_log_dir
        )

    for _ in range(len(shard_manager.shard_files)):
        shard_url = shard_manager.get_next_shard()

        parts = shard_url.split("/")
        relative_path = os.path.join(parts[-2], parts[-1])  # tokens/batch_0002.pt
        
        file_local_path = os.path.join(
            train_config.dataset_dir,
            relative_path
        )
        # Rank 0 only downloads file
        # -------------------------
        if dist.get_rank() == 0:
            os.makedirs(os.path.dirname(file_local_path), exist_ok=True)
        
            hf_api.download_hf_file_from_url(
                file_url=shard_url,
                local_datasetdir=train_config.dataset_dir
            )
        
        # Synchronize all GPUs
        dist.barrier()
        
        dataloader = get_dataloader(
            shard_file = file_local_path,
            batch_size = train_config.batch_size,
            world_size = world_size,
            rank = rank if world_size > 1 else 0,
            num_workers = train_config.num_workers,
            validate = True
        )

        shard_stats = train_on_shard(
            model = model_states["model"],
            criterion = torch.nn.CrossEntropyLoss(),
            dataloader = dataloader,
            optimizer = model_states["optimizer"],
            scheduler = model_states["scheduler"],
            device = state_manager.device,
            scaler = model_states["scaler"],
            max_grad_norm = train_config.max_grad_norm,
            gradient_accumulation_steps = train_config.accumulation_steps,
            start_global_step = model_states["global_step"],
            writer = tb_logger,
            rank = rank,
            dtype = train_config.dtype
        )

        model_states["global_step"] = shard_stats["global_step"]

        if rank == 0:
            print(f"Finished training a single shard : {shard_stats}")

            # Save checkpoint after each shard
            local_ckpt_path = checkpoint_manager.save_checkpoint(
                model = model_states["model"],
                optimizer = model_states["optimizer"],
                scheduler = model_states["scheduler"],
                scaler = model_states["scaler"],
                global_step = model_states["global_step"],
                wandb_run_id = wandb.run.id if wandb.run else None,
                name = f"checkpoints_at_step_{model_states['global_step']}"       
            )
            
            hf_api.save_checkpoint_to_hf( checkpoint_path = local_ckpt_path, commit_message = "ckpt_saved")
            
            shard_file = shard_manager.remove_shard()
            print(f"Removed shard file {shard_file} from shard manager.")

        if world_size> 1:
            dist.barrier()
        
        shard_manager.reload()  # reload shard list after removal in all ranks
        
        # All ranks obey the same decision
        if should_stop_from_wandb(rank, world_size, device):
            if rank == 0:
                print("Stopping training as requested via W&B 'stop_training' signal.")
            break
            
    if world_size > 1: 
        cleanup_distributed()

if __name__ == "__main__":
    main()


















