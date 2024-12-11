import sys
sys.path.append('.')

from src.models import prediction
from experiments.config import predict_config
from src.utils import helper

# model_names = [
#     "hierarchical-B_CNN-classification-use_model_structure-20241027_161111-seed42.pt",
#     "hierarchical-B_CNN-classification-use_model_structure-20241027_161111-seed43.pt",
#     "hierarchical-B_CNN-classification-use_model_structure-20241027_161111-seed44.pt",
#     "hierarchical-B_CNN-classification-use_model_structure-20241027_161111-seed45.pt",
#     "hierarchical-B_CNN-classification-use_model_structure-20241027_161111-seed46.pt"
# # ]


seeds = [42]
# Iterate over each seed and run the prediction
for seed in seeds:
    helper.set_seed(seed)
    print(f"Running prediction with seed {seed}")
    
    # Optionally, update the config to include the seed info
    config = {
        **predict_config.C_CNN_CORN_GT,
        "seed": seed  # Add the seed to config if you want to keep track of it
    }
    
    # Run the prediction
    prediction.run_dataset_predict_csv(config)

#prediction.run_dataset_predict_csv(predict_config.C_CNN_CORN_GT)
#prediction.run_dataset_predict_csv(predict_config.C_CNN)

#prediction.cam_prediction(predict_config.C_CNN_CORN_GT)