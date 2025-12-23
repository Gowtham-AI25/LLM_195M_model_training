import torch 
from llm_training_project.training.TraningStateManager import TrainingStateManager
from llm_training_project.checkpoints_dir.checkpoint import CheckpointManager
from llm_training_project.config.train_config import LLM_training_config
from llm_training_project.config.model_config import LLM_model_config
from llm_training_project.utils.HF_utils import HFUtils
from llm_training_project.training.trainer import train_on_shard
from llm_training_project.log.tensorboard_logger import TensorBoardLogger
from llm_training_project.shards.ShardManager import ShardManager
from llm_training_project.dataloader.DataLoader import get_dataloader
from llm_training_project.utils.distributed import setup_distributed, cleanup_distributed
from llm_training_project.model.model import LLM
import torch.distributed as dist


def main():

    train_config = LLM_training_config.load_from_yaml("configs/train_config.yaml")
    model_config = LLM_model_config.load_from_yaml("configs/model_config.yaml")
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

    tb_logger = None
    if rank == 0:
        # tensorboard logger
        tb_logger = TensorBoardLogger(
            log_dir = train_config.tensorboard_log_dir
        )

    # initializ TrainingStateManager to get model, optimizer, scaler, scheduler, etc.
    state_manager = TrainingStateManager(
        train_config,
        model_config,
        checkpoint = checkpoint_manager,
        device = device
    )

    hf_api = HFUtils.load_config_from_yaml("llm_training_project/config/configs/hf_config.yaml")

    # Load or create training state
    model_states = state_manager.load_training_state(LLM)

    for _ in range(len(shard_manager.shard_files)):
        
        shard_file = shard_manager.get_next_shard()
    
        file_local_path = hf_api.download_shard(
            hf_shard_path = shard_file,
            local_dir = train_config.dataset_dir
        )

        dataloader = get_dataloader(
            shard_file = file_local_path,
            batch_size = train_config.batch_size_per_device,
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
            gradient_accumulation_steps = train_config.gradient_accumulation_steps,
            start_global_step = model_states["global_step"],
            writer = tb_logger,
            rank = rank,
        )

        model_states["global_step"] = shard_stats["global_step"]

        if rank == 0:
            print(f"Finished training a single shard : {shard_stats}")

            # Save checkpoint after each shard
            checkpoint_manager.save_checkpoint(
                model = model_states["model"],
                optimizer = model_states["optimizer"],
                scheduler = model_states["scheduler"],
                scaler = model_states["scaler"],
                global_step = model_states["global_step"],
                name = f"checkpoints_at_step_{model_states['global_step']}"       
            )

        if world_size> 1:
            dist.barrier()
        
        if rank == 0:
            shard_file = shard_manager.remove_shard()
            print(f"Removed shard file {shard_file} from shard manager.")
        
        if world_size > 1:
            dist.barrier()

    if world_size > 1: 
        cleanup_distributed()

if __name__ == "__main__":
    main()

