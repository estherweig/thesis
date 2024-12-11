import sys
sys.path.append('.')

from src.models import training
from experiments.config import train_config

#seeds=[42, 43, 44]
seeds=[42]
for seed in seeds:
    #training.run_training(config=train_config.vgg16_surface_params, seed=seed)
    print(f"running next model")
    training.run_training(config=train_config.vgg16_quality_params, seed=seed)


#training.run_training(config=train_config.vgg16_surface_params)

#training.run_training(config=train_config.vgg16_quality_params)s