"""A tiny, randomly initialised nnU-Net written to disk in the study's layouts.

It is not a trained model. It exists so the real loading and prediction code
path (plans parsing, custom trainer name, sliding window, resampling, axis
handling, softmax export) can be exercised without the study's weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from app.labels import EXPECTED_CHANNEL_NAMES, EXPECTED_DATASET_LABELS

from . import synthetic

DATASET_JSON = {
    "channel_names": EXPECTED_CHANNEL_NAMES,
    "labels": EXPECTED_DATASET_LABELS,
    "numTraining": 475,
    "file_ending": ".nii.gz",
}


def plans(features=(4, 8, 16)) -> dict:
    spacing_zyx = list(synthetic.SPACING[::-1])
    shape_zyx = list(synthetic.SHAPE[::-1])
    resampling = "resample_data_or_seg_to_shape"
    return {
        "dataset_name": "Dataset502_MUGliomaPost",
        "plans_name": "nnUNetPlans",
        "original_median_spacing_after_transp": spacing_zyx,
        "original_median_shape_after_transp": shape_zyx,
        "image_reader_writer": "SimpleITKIO",
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],
        "experiment_planner_used": "ExperimentPlanner",
        "label_manager": "LabelManager",
        "foreground_intensity_properties_per_channel": {
            str(c): {"max": 1000.0, "mean": 400.0, "median": 400.0, "min": 0.0,
                     "percentile_00_5": 50.0, "percentile_99_5": 900.0, "std": 100.0}
            for c in range(4)
        },
        "configurations": {
            "3d_fullres": {
                "data_identifier": "nnUNetPlans_3d_fullres",
                "preprocessor_name": "DefaultPreprocessor",
                "batch_size": 2,
                "patch_size": [32, 32, 32],
                "median_image_size_in_voxels": shape_zyx,
                "spacing": spacing_zyx,
                "normalization_schemes": ["ZScoreNormalization"] * 4,
                "use_mask_for_norm": [True] * 4,
                "resampling_fn_data": resampling,
                "resampling_fn_seg": resampling,
                "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None},
                "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1, "order_z": 0, "force_separate_z": None},
                "resampling_fn_probabilities": resampling,
                "resampling_fn_probabilities_kwargs": {"is_seg": False, "order": 1, "order_z": 0, "force_separate_z": None},
                "architecture": {
                    "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
                    "arch_kwargs": {
                        "n_stages": len(features),
                        "features_per_stage": list(features),
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "kernel_sizes": [[3, 3, 3]] * len(features),
                        "strides": [[1, 1, 1]] + [[2, 2, 2]] * (len(features) - 1),
                        "n_conv_per_stage": [1] * len(features),
                        "n_conv_per_stage_decoder": [1] * (len(features) - 1),
                        "conv_bias": True,
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
                        "dropout_op": None,
                        "dropout_op_kwargs": None,
                        "nonlin": "torch.nn.LeakyReLU",
                        "nonlin_kwargs": {"inplace": True},
                    },
                    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
                },
                "batch_dice": False,
            }
        },
    }


def write_member(
    directory: Path,
    *,
    seed: int,
    layout: str = "archive",
    trainer_name: str = "nnUNetTrainer_MUGlioma_no29",
    dataset_json: dict | None = None,
    features=(4, 8, 16),
    checkpoint_name: str = "checkpoint_best.pth",
) -> Path:
    """layout "archive": files directly in ``directory`` (the study zip's model/ folder);
    layout "nnunet": checkpoint under ``directory/fold_0`` (nnU-Net results layout)."""
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    directory.mkdir(parents=True, exist_ok=True)
    plan = plans(features)
    dataset = dataset_json or DATASET_JSON
    (directory / "plans.json").write_text(json.dumps(plan))
    (directory / "dataset.json").write_text(json.dumps(dataset))

    torch.manual_seed(seed)
    manager = PlansManager(plan)
    configuration = manager.get_configuration("3d_fullres")
    network = nnUNetTrainer.build_network_architecture(manager, configuration, 4, 5, enable_deep_supervision=False)
    with torch.no_grad():  # spread the logits so the random net predicts several classes
        for parameter in network.parameters():
            parameter.mul_(3.0)
    checkpoint = {
        "network_weights": network.state_dict(),
        "trainer_name": trainer_name,
        "init_args": {"configuration": "3d_fullres", "fold": 0},
        "inference_allowed_mirroring_axes": (0, 1, 2),
        "current_epoch": 82,
    }
    target = directory / "fold_0" if layout == "nnunet" else directory
    target.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target / checkpoint_name)
    return directory
