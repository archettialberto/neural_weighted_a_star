import ray

from trainer.grid_model_wrapper import GridModelWrapper
from trainer.grid_trainer import GridTrainer

NUM_CPUS = 4
NUM_GPUS = 1
MEMORY = 32
OBJECT_STORE_MEMORY = 8


def main():
    ray.init(
        num_cpus=NUM_CPUS,
        num_gpus=NUM_GPUS,
        _memory=1024 * 1024 * 1024 * MEMORY,
        object_store_memory=1024 * 1024 * 1024 * OBJECT_STORE_MEMORY
    )
    trainer = GridTrainer(
        model_class=GridModelWrapper
    )
    trainer.fit()


if __name__ == "__main__":
    main()
