from src import constants as const
from experiments.config import global_config



# Models:
# B_CNN_CORN_GT: hierarchical-B_CNN-corn-use_ground_truth-20241031_005138-32tlqlfa_epoch11.pt


vgg16_surface = {
    **global_config.global_config,
    "name": "surface_prediction",
    "model_dict": {"trained_model": "surface-vgg16-20240202_125044-1uv8oow5.pt"},
    "dataset": "V5_c3",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
}

CC_smoothness_CAM = {
    **global_config.global_config,
    "name": "Smoothness_asphalt_CLASSIFICATION_prediction",
    "model_dict": {"trained_model": "smoothness-asphalt-vgg16-classification-CC-20241103_102236_epoch0.pt"},
    "dataset": "V1_0",
    "ds_type": "train",
    "metadata": "streetSurfaceVis_v1_0.csv",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
    "save_features": True,
}

FLATTEN = {
    **global_config.global_config,
    "name": "Flatten_CLASSIFICATION_prediction",
    "model_dict": {"trained_model": "flatten-vgg16-classification-flatten-20241105_210302_epoch0.pt"},
    #"model_dict": {"trained_model": "flatten-vgg16-classification-flatten-20241106_220720-8zz0sjns42_epoch10.pt"}, 
    "dataset": r"V1_0",
    "ds_type": "train",
    "metadata": r"streetSurfaceVis_v1_0.csv",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
    "save_features": True,
}

B_CNN = {
    **global_config.global_config,
    "name": "B_CNN_CLASSIFICATION_prediction",
    "model_dict": {"trained_model": "hierarchical-B_CNN-classification-use_model_structure20241027_124835_epoch0.pt"},
    #"model_dict": {"trained_model": "hierarchical-B_CNN-classification-use_model_structure-20241106_222713-z8l98z7u42_epoch9.pt"},
    #"model_dict": {"trained_model": "multilabel-BCNN-20240504_141652-jzr601kb_epoch39.pt"}, 
    "dataset": r"V1_0",
    "ds_type": "test",
    "metadata": r"streetSurfaceVis_v1_0.csv",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
    "save_features": True,
}

C_CNN = {
    **global_config.global_config,
    "name": "C_CNN_CLASSIFICATION_prediction",
    "model_dict": {"trained_model": "hierarchical-Condition_CNN-classification-use_model_structure-20241110_13285442_epoch0.pt"},
    #"model_dict": {"trained_model": "multilabel-BCNN-20240504_141652-jzr601kb_epoch39.pt"}, 
    "dataset": r"V1_0",
    "ds_type": "test",
    "metadata": r"streetSurfaceVis_v1_0.csv",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
    "save_features": False,
}

H_NET = {
    **global_config.global_config,
    "name": "H_NET_CLASSIFICATION_prediction",
    "model_dict": {"trained_model": "hierarchical-HiearchyNet-classification-use_model_structure-20241106_222805-k45vh7jt42_epoch8.pt"},
    #"model_dict": {"trained_model": "multilabel-BCNN-20240504_141652-jzr601kb_epoch39.pt"}, 
    "dataset": r"V1_0",
    "ds_type": "train",
    "metadata": r"streetSurfaceVis_v1_0.csv",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
    "save_features": True,
}

GH_CNN = {
    **global_config.global_config,
    "name": "GH_CNN_CLASSIFICATION_prediction",
    "model_dict": {"trained_model": "hierarchical-GH_CNN-classification-use_model_structure-20241106_222749-y1aa7o3d42_epoch11.pt"},
    #"model_dict": {"trained_model": "multilabel-BCNN-20240504_141652-jzr601kb_epoch39.pt"}, 
    "dataset": r"V1_0",
    "ds_type": "train",
    "metadata": r"streetSurfaceVis_v1_0.csv",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
    "save_features": True,
}


C_CNN_CORN_GT = {
    **global_config.global_config,
    "name": "C_CNN_CORN_GT_prediction",
    "model_dict": {"trained_model": "hierarchical-Condition_CNN-corn-use_ground_truth-20241130_11592342_epoch1.pt"},
    #"model_dict": {"trained_model": "hierarchical-Condition_CNN-corn-use_ground_truth-20241109_193510-1efl9o3u42_epoch4.pt"},
    #"model_dict": {"trained_model": "multilabel-BCNN-20240504_141652-jzr601kb_epoch39.pt"}, 
    "dataset": r"V1_0",
    "ds_type": "test",
    "metadata": r"streetSurfaceVis_v1_0.csv",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
    "save_features": True,
}

CC = {
    **global_config.global_config,
    "name": "CC_prediction",
    "head": const.CLASSIFICATION,
    "level": const.CC,
    "model_dict": {
        "trained_model": "surface-vgg16-classification-CC-20241103_102134_epoch0.pt",
        "level": const.TYPE,
        "submodels": {
            const.ASPHALT: {
                "trained_model": "smoothness-asphalt-vgg16-classification-CC-20241113_21174642_epoch0.pt",
                "level": const.QUALITY,
            },
            const.CONCRETE: {
                "trained_model": "smoothness-concrete-vgg16-classification-CC-20241113_221528-5bu9pd2d42_epoch1.pt",
                "level": const.QUALITY,
            },
            const.PAVING_STONES: {
                "trained_model": "smoothness-paving_stones-vgg16-classification-CC-20241113_221528-5bu9pd2d42_epoch1.pt",
                "level": const.QUALITY,
            },
            const.SETT: {
                "trained_model": "smoothness-sett-vgg16-classification-CC-20241113_221528-5bu9pd2d42_epoch1.pt",
                "level": const.QUALITY,
            },
            const.UNPAVED: {
                "trained_model": "smoothness-unpaved-vgg16-classification-CC-20241113_221528-5bu9pd2d42_epoch1.pt",
                "level": const.QUALITY,
            },
        },
    },
    "dataset": "V1_0",
    "ds_type": "test",
    "metadata": "streetSurfaceVis_v1_0.csv",
    "transform": {
        "resize": const.H256_W256,
        "crop": const.CROP_LOWER_MIDDLE_THIRD,
        "normalize": (const.V1_0_ANNOTATED_MEAN, const.V1_0_ANNOTATED_SD),
    },
    "batch_size": 96,
    "save_features": True,
}




# B_CNN = {
#     **global_config.global_config,
#     "name": "B_CNN_prediction",
#     "model_dict": {"trained_model": "multilabel-BCNN_pretrained-20240505_133427-c549if0b_epoch39.pt"}, 
#     "dataset": "V11/annotated", #V5_c5/unsorted_images",
#     "transform": {
#         "resize": const.H256_W256,
#         "crop": const.CROP_LOWER_MIDDLE_THIRD,
#         "normalize": (const.V6_ANNOTATED_MEAN, const.V6_ANNOTATED_SD),
#     },
#     "batch_size": 96,
#     "save_features": True
# }
# B_CNN = {
#     **global_config.global_config,
#     "name": "B_CNN_prediction",
#     "model_dict": {"trained_model": "multilabel-BCNN_pretrained-20240505_133427-c549if0b_epoch39.pt"}, 
#     "dataset": "V11/annotated", #V5_c5/unsorted_images",
#     "transform": {
#         "resize": const.H256_W256,
#         "crop": const.CROP_LOWER_MIDDLE_THIRD,
#         "normalize": (const.V6_ANNOTATED_MEAN, const.V6_ANNOTATED_SD),
#     },
#     "batch_size": 96,
#     "save_features": True
# }


# C_CNN_PRE = {
#     **global_config.global_config,
#     "name": "C_CNN_prediction",
#     "model_dict": {"trained_model": "hierarchical-Condition_CNN_CLM_PRE-20240820_163509_epoch0.pt"}, 
#     "dataset": "V12/annotated", #V5_c5/unsorted_images",
#     "transform": {
#         "resize": const.H256_W256,
#         "crop": const.CROP_LOWER_MIDDLE_THIRD,
#         "normalize": (const.V6_ANNOTATED_MEAN, const.V6_ANNOTATED_SD),
#     },
#     "level": const.HIERARCHICAL,
#     "head": 'regression', #'regression', 'classification', 'obd', 'clm'
#     "hierarchy_method": const.GROUNDTRUTH, #'use_ground_truth', 'use_condition_layer', 'top_coarse_prob'
#     "batch_size": 96,
#     "save_features": False
# }
