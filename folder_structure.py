# from pathlib import Path

# BASE = Path("llm_training_project")

# paths = [
#     # root
#     BASE / "train.py",
#     BASE / "requirements.txt",

#     # config
#     BASE / "config" / "__init__.py",
#     BASE / "config" / "model_config.py",
#     BASE / "config" / "train_config.py",
#     BASE / "config" / "paths.py",

#     # model
#     BASE / "model" / "__init__.py",
#     BASE / "model" / "llm.py",

#     # data
#     BASE / "data" / "__init__.py",
#     BASE / "data" / "shard_dataset.py",
#     BASE / "data" / "shard_manager.py",

#     # engine
#     BASE / "engine" / "__init__.py",
#     BASE / "engine" / "trainer.py",
#     BASE / "engine" / "checkpoint.py",
#     BASE / "engine" / "ddp.py",

#     # utils
#     BASE / "utils" / "__init__.py",
#     BASE / "utils" / "logging.py",
#     BASE / "utils" / "distributed.py",

#     # runtime dirs
#     BASE / "shards",
#     BASE / "checkpoints",
# ]

# for p in paths:
#     if p.suffix:   # file
#         p.parent.mkdir(parents=True, exist_ok=True)
#         p.touch(exist_ok=True)
#     else:          # directory
#         p.mkdir(parents=True, exist_ok=True)

# print("✅ Project structure created successfully.")

