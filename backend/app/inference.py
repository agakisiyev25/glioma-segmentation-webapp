"""nnU-Net v2 model discovery, contract checks and prediction.

Follows docs/MODEL_CONTRACT.md §3–§5. In short:

* weights come only from ``NNUNET_MODEL_DIR``; no other model is ever used;
* the weights' ``dataset.json`` must match the study's channel order and label
  map, otherwise inference is refused;
* the study's layout (one independent model per outer fold, each with its own
  plans and a custom trainer class) is supported directly.

``torch`` and ``nnunetv2`` are imported lazily so demo mode runs without them.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .config import Settings
from .labels import EXPECTED_CHANNEL_NAMES, EXPECTED_DATASET_LABELS
from .preprocess import hd_bet_available

log = logging.getLogger(__name__)

STUDY_TRAINER_PREFIX = "nnUNetTrainer_MUGlioma_"
PREFERRED_CHECKPOINTS = ("checkpoint_best.pth", "checkpoint_final.pth")
PAPER_REFERENCE = {
    "title": (
        "Nested Patient-Grouped Benchmarking of 3D Deep-Learning Pipelines for Multiclass "
        "Postoperative Glioma Segmentation on Longitudinal MRI"
    ),
    "pipeline": "nnU-Net v2 3d_fullres",
    "primary_metric": "mean patient-level Dice, tumor-related region (labels 1–3)",
    "reported_value": 0.879,
    "reported_ci95": [0.856, 0.900],
    "evaluation": "pooled out-of-fold predictions of five single models, 203 patients, internal validation only",
}


class ModelContractError(RuntimeError):
    """The weights folder does not hold the model described in MODEL_CONTRACT.md."""


@dataclass
class ModelMember:
    """One independently trained nnU-Net (its own plans), with one or more folds."""

    name: str
    folder: Path
    checkpoints: list[Path]
    trainer_name: str | None = None
    configuration: str | None = None
    target_spacing: list[float] | None = None
    patch_size: list[int] | None = None
    sha256: dict[str, str] = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "checkpoints": [
                {"file": f"{p.parent.name}/{p.name}" if p.parent != self.folder else p.name, "sha256": self.sha256.get(str(p))}
                for p in self.checkpoints
            ],
            "trainer": self.trainer_name,
            "plans": "nnUNetPlans",
            "configuration": self.configuration,
            "target_spacing_mm": self.target_spacing,
            "patch_size": self.patch_size,
        }


@dataclass
class PredictionResult:
    labels: np.ndarray  # uint8, indexed [i, j, k] on the grid of the input files
    confidence: np.ndarray  # float32 max softmax probability, same grid
    info: dict
    warnings: list[str]


# ------------------------------------------------------------------ discovery


def _pick_checkpoint(directory: Path, override: str | None) -> Path | None:
    names = (override,) if override else PREFERRED_CHECKPOINTS
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _member_from_dir(directory: Path, root: Path, override: str | None) -> ModelMember | None:
    if not (directory / "plans.json").is_file() or not (directory / "dataset.json").is_file():
        return None
    checkpoints = []
    for fold_dir in sorted(p for p in directory.glob("fold_*") if p.is_dir()):
        checkpoint = _pick_checkpoint(fold_dir, override)
        if checkpoint is not None:
            checkpoints.append(checkpoint)
    if not checkpoints:
        direct = _pick_checkpoint(directory, override)  # study archive: model/checkpoint_best.pth
        if direct is not None:
            checkpoints.append(direct)
    if not checkpoints:
        return None
    name = "." if directory == root else directory.relative_to(root).as_posix()
    return ModelMember(name=name, folder=directory, checkpoints=checkpoints)


def discover_members(model_dir: Path, checkpoint_override: str | None = None) -> list[ModelMember]:
    """Find every model under ``model_dir`` (layouts A, B and C of the contract)."""
    root_member = _member_from_dir(model_dir, model_dir, checkpoint_override)
    if root_member is not None:
        return [root_member]
    members = []
    for plans in sorted(model_dir.rglob("plans.json")):
        if len(plans.relative_to(model_dir).parts) > 6:
            continue
        member = _member_from_dir(plans.parent, model_dir, checkpoint_override)
        if member is not None:
            members.append(member)
    return members


def validate_dataset_json(dataset: dict, where: str) -> None:
    channels = {str(k): str(v).strip().upper() for k, v in dataset.get("channel_names", {}).items()}
    expected_channels = {k: v.upper() for k, v in EXPECTED_CHANNEL_NAMES.items()}
    if channels != expected_channels:
        raise ModelContractError(
            f"{where}: dataset.json channel_names are {dataset.get('channel_names')}, "
            f"expected {EXPECTED_CHANNEL_NAMES} (T1, T1c, T2, FLAIR as _0000…_0003)."
        )
    labels = dataset.get("labels", {})
    try:
        normalised = {str(k): int(v) for k, v in labels.items()}
    except (TypeError, ValueError):
        raise ModelContractError(f"{where}: dataset.json uses region-based labels, which this app does not support.")
    if normalised != EXPECTED_DATASET_LABELS:
        raise ModelContractError(
            f"{where}: dataset.json labels are {labels}, expected {EXPECTED_DATASET_LABELS}."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# -------------------------------------------------------------------- status


@dataclass
class ModelStatus:
    mode: str  # "inference" | "demo" | "model_error"
    message: str
    model: "SegmentationModel | None" = None
    hd_bet_available: bool = False

    def describe(self) -> dict | None:
        return self.model.describe() if self.model is not None else None


def resolve_model(settings: Settings) -> ModelStatus:
    """Decide the operating mode at start-up.

    * no ``NNUNET_MODEL_DIR`` / no weights in it → demo mode;
    * weights found and valid → inference;
    * weights found but wrong, or the inference stack missing → ``model_error``
      (never a silent fallback to demo or to another model).
    """
    bet = hd_bet_available()
    if settings.model_dir is None:
        return ModelStatus("demo", "NNUNET_MODEL_DIR is not set. Running in demo mode.", hd_bet_available=bet)
    if not settings.model_dir.is_dir():
        return ModelStatus(
            "demo", "NNUNET_MODEL_DIR does not exist. Running in demo mode.", hd_bet_available=bet
        )
    members = discover_members(settings.model_dir, settings.checkpoint_name)
    if not members:
        return ModelStatus(
            "demo",
            "No nnU-Net weights (plans.json + dataset.json + checkpoint) found in NNUNET_MODEL_DIR. "
            "Running in demo mode.",
            hd_bet_available=bet,
        )
    try:
        for member in members:
            validate_dataset_json(json.loads((member.folder / "dataset.json").read_text()), member.name)
        try:
            import nnunetv2  # noqa: F401
            import torch  # noqa: F401
        except ImportError as error:
            raise ModelContractError(
                "Model weights were found, but torch/nnunetv2 are not installed "
                "(pip install -r requirements-inference.txt)."
            ) from error
        model = SegmentationModel(members, settings)
    except ModelContractError as error:
        return ModelStatus("model_error", str(error), hd_bet_available=bet)
    return ModelStatus("inference", model.summary_line(), model=model, hd_bet_available=bet)


# --------------------------------------------------------------------- model


class SegmentationModel:
    def __init__(self, members: list[ModelMember], settings: Settings):
        import torch

        self.members = members
        self.settings = settings
        self._predictors: dict[str, object] = {}
        self._lock = threading.Lock()
        if settings.device == "cpu":
            self.device = "cpu"
        elif settings.device == "cuda":
            if not torch.cuda.is_available():
                raise ModelContractError("NNUNET_DEVICE=cuda was requested but no CUDA GPU is available.")
            self.device = "cuda"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else None
        self.gpu_memory_gb = (
            round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if self.device == "cuda" else None
        )

    # ---------------------------------------------------------------- info

    @property
    def is_cross_model_ensemble(self) -> bool:
        return len(self.members) > 1

    def summary_line(self) -> str:
        folds = sum(len(m.checkpoints) for m in self.members)
        kind = (
            f"cross-model ensemble of {len(self.members)} nnU-Net models"
            if self.is_cross_model_ensemble
            else f"1 nnU-Net model ({folds} fold{'s' if folds != 1 else ''})"
        )
        return f"{kind} on {self.device.upper()}"

    def describe(self) -> dict:
        import importlib.metadata as metadata

        import torch

        return {
            "family": "nnU-Net v2",
            "configuration": "3d_fullres",
            "nnunetv2_version": metadata.version("nnunetv2"),
            "torch_version": torch.__version__,
            "device": self.device,
            "gpu": self.gpu_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "members": [m.describe() for m in self.members],
            "ensemble": (
                "softmax probabilities averaged across independently trained models"
                if self.is_cross_model_ensemble
                else ("nnU-Net fold ensemble" if len(self.members[0].checkpoints) > 1 else "single model")
            ),
            "test_time_mirroring": not self.settings.disable_tta,
            "sliding_window_step": 0.5,
            "paper": PAPER_REFERENCE,
            "notes": self._notes(),
        }

    def _notes(self) -> list[str]:
        notes = []
        if self.is_cross_model_ensemble:
            notes.append(
                "The paper's Dice of 0.879 is a pool of single-model, out-of-fold predictions. "
                "An ensemble of several of those models was not evaluated in the paper."
            )
        if self.settings.disable_tta:
            notes.append("Test-time mirroring is disabled (NNUNET_DISABLE_TTA); the study used it.")
        return notes

    # ------------------------------------------------------------- loading

    def _load_member(self, member: ModelMember):
        """Build an nnUNetPredictor for one member without needing its custom trainer class installed."""
        import nnunetv2
        import torch
        from batchgenerators.utilities.file_and_folder_operations import join
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

        dataset_json = json.loads((member.folder / "dataset.json").read_text())
        validate_dataset_json(dataset_json, member.name)
        plans_manager = PlansManager(json.loads((member.folder / "plans.json").read_text()))

        parameters, trainer_name, configuration_name, mirroring_axes = [], None, None, None
        for index, checkpoint_path in enumerate(member.checkpoints):
            member.sha256[str(checkpoint_path)] = _sha256(checkpoint_path)
            # nnU-Net checkpoints are pickles; these are the operator's own weights.
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
            if index == 0:
                trainer_name = checkpoint["trainer_name"]
                configuration_name = checkpoint["init_args"]["configuration"]
                mirroring_axes = checkpoint.get("inference_allowed_mirroring_axes")
            parameters.append(checkpoint["network_weights"])

        if configuration_name != "3d_fullres":
            raise ModelContractError(
                f"{member.name}: checkpoint was trained with configuration {configuration_name!r}; "
                "the study model is 3d_fullres."
            )

        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), trainer_name, "nnunetv2.training.nnUNetTrainer"
        )
        if trainer_class is None:
            if not trainer_name.startswith(STUDY_TRAINER_PREFIX):
                raise ModelContractError(
                    f"{member.name}: unknown trainer class {trainer_name!r}. Only stock nnU-Net trainers and the "
                    f"study's {STUDY_TRAINER_PREFIX}* trainers are accepted."
                )
            # The study trainer changes only training length, seeding and early
            # stopping; it inherits build_network_architecture unchanged.
            trainer_class = nnUNetTrainer

        configuration_manager = plans_manager.get_configuration(configuration_name)
        num_inputs = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        num_outputs = plans_manager.get_label_manager(dataset_json).num_segmentation_heads
        # Same dispatch as nnUNetPredictor.initialize_from_trained_model_folder (2.8.1).
        if "plans_manager" in inspect.signature(trainer_class.build_network_architecture).parameters:
            network = trainer_class.build_network_architecture(
                plans_manager, configuration_manager, num_inputs, num_outputs, enable_deep_supervision=False
            )
        else:
            network = trainer_class.build_network_architecture(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_inputs,
                num_outputs,
                enable_deep_supervision=False,
            )

        on_gpu = self.device == "cuda"
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=not self.settings.disable_tta,
            # Keeping the full-volume logits on a small GPU risks out-of-memory.
            perform_everything_on_device=bool(on_gpu and (self.gpu_memory_gb or 0) >= 10),
            device=torch.device(self.device),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.manual_initialization(
            network, plans_manager, configuration_manager, parameters, dataset_json, trainer_name, mirroring_axes
        )
        member.trainer_name = trainer_name
        member.configuration = configuration_name
        member.target_spacing = [float(v) for v in configuration_manager.spacing]
        member.patch_size = [int(v) for v in configuration_manager.patch_size]
        return predictor

    def _predictor(self, member: ModelMember):
        if member.name not in self._predictors:
            self._predictors[member.name] = self._load_member(member)
        return self._predictors[member.name]

    # ---------------------------------------------------------- prediction

    def predict(self, image_paths: list[Path], progress: Callable[[float, str], None]) -> PredictionResult:
        """Segment one case. ``image_paths`` are T1, T1c, T2, FLAIR (channels 0–3)."""
        import torch
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

        warnings: list[str] = []
        with self._lock:
            # Same reader the study used for .nii.gz (nnU-Net's default).
            # Array layout is [channel, z, y, x].
            images, properties = SimpleITKIO().read_images([str(p) for p in image_paths])
            mean_probabilities = None
            first_labels = None
            for index, member in enumerate(self.members):
                progress(index / len(self.members), f"nnU-Net model {index + 1} of {len(self.members)} ({member.name})")
                predictor = self._predictor(member)
                try:
                    labels, probabilities = predictor.predict_single_npy_array(images, properties, None, None, True)
                except RuntimeError as error:
                    if self.device != "cuda" or "out of memory" not in str(error).lower():
                        raise
                    # Do not fail the job on a small GPU: redo this model on CPU.
                    log.warning("CUDA out of memory; repeating this model on CPU")
                    warnings.append("The GPU ran out of memory; inference fell back to CPU for at least one model.")
                    torch.cuda.empty_cache()
                    predictor.network.to("cpu")
                    predictor.device = torch.device("cpu")
                    predictor.perform_everything_on_device = False
                    labels, probabilities = predictor.predict_single_npy_array(images, properties, None, None, True)
                probabilities = np.asarray(probabilities, dtype=np.float32)
                if first_labels is None:
                    first_labels = np.asarray(labels)
                mean_probabilities = (
                    probabilities if mean_probabilities is None else mean_probabilities + probabilities
                )
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            mean_probabilities /= float(len(self.members))

        if self.is_cross_model_ensemble:
            labels_zyx = np.argmax(mean_probabilities, axis=0)
        else:
            labels_zyx = first_labels  # nnU-Net's own conversion for a single model
            agreement = float((np.argmax(mean_probabilities, axis=0) == labels_zyx).mean())
            if agreement < 0.999:
                raise RuntimeError(
                    f"Label map and probability map disagree ({agreement:.4f}); refusing to show a misaligned confidence map."
                )
        confidence_zyx = mean_probabilities.max(axis=0)

        # SimpleITK arrays are [z, y, x]; NIfTI voxel indices are [x, y, z] = [i, j, k].
        labels = np.ascontiguousarray(np.transpose(labels_zyx, (2, 1, 0))).astype(np.uint8)
        confidence = np.ascontiguousarray(np.transpose(confidence_zyx, (2, 1, 0))).astype(np.float32)
        progress(1.0, "Inference finished")
        return PredictionResult(labels=labels, confidence=confidence, info=self.describe(), warnings=warnings)
