import torch
import numpy as np
from torchvision.models.video import s3d, S3D_Weights, swin3d_b, swin3d_t, Swin3D_B_Weights, Swin3D_T_Weights, mvit_v2_s, MViT_V2_S_Weights, r2plus1d_18, R2Plus1D_18_Weights
from pathlib import Path
from path_configurator import * 
from sklearn.model_selection import StratifiedKFold
from typing import Optional, Dict, Any
from bespoke_tools import *
from bespoke_ai_training_tools import * 
from bespoke_video_tools import convert_seconds_to_timestamp
from transformers import VideoMAEForVideoClassification
from cfa_video_classifier import *
from cf_transform import PeripheralEnvisionate

def create_model(model_type: str, num_classes: int, device: torch.device):
    #Video classification model is implicit

    num_classes = num_classes
    if model_type == 'S3D':
        model = s3d(weights=S3D_Weights.KINETICS400_V1)
        in_channels = model.classifier[-1].in_channels
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Conv3d(in_channels, num_classes, kernel_size=1)
        )
    elif model_type == "Swin3D_B_Kinetics_and_Imagenet22K":
        model = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)
        in_features = model.head.in_features
        model.head = torch.nn.Linear(in_features, num_classes)

    elif model_type == "Swin3D_B_Kinetics":
        model = swin3d_b(weights=Swin3D_B_Weights.KINETICS400_V1)
        in_features = model.head.in_features
        model.head = torch.nn.Linear(in_features, num_classes)

    elif model_type == "Swin3D_Tiny":
        model = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
        in_features = model.head.in_features
        model.head = torch.nn.Linear(in_features, num_classes)

    elif model_type == "MViTv2-S":
        model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
        in_features = model.head[1].in_features
        model.head[1] = torch.nn.Linear(in_features, num_classes)
    
    elif model_type == 'ResnetR2+1D_R18':
        model = r2plus1d_18(weights = R2Plus1D_18_Weights.KINETICS400_V1)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)

    elif model_type == "VideoMAEv2-L":
        model = VideoMAEv2Classifier("OpenGVLab/VideoMAEv2-Large", num_classes)

    elif model_type == "VideoMAEv2-B":
        model = VideoMAEv2Classifier("OpenGVLab/VideoMAEv2-Base", num_classes)

    elif model_type == "VideoMAEv2-H":
        model = VideoMAEv2Classifier("OpenGVLab/VideoMAEv2-Huge", num_classes)

    elif model_type == "VideoMAE-B-Kinetics":
        # Pretrained classifier head from HF (Kinetics-400 labels)
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", num_labels=num_classes, ignore_mismatched_sizes=True)

    else:
        raise ValueError("Mate, this model you're trying to create doesn't bloody exist!")

    model = model.to(device)

    return model

def save_model(model, model_type: str, filename: str, models_path: str | Path = get_models_path()):

    models_path = Path(models_path)
    save_folder_path = models_path / model_type / "models"
    save_folder_path.mkdir(exist_ok=True,parents=True)
    save_path = save_folder_path / filename
    state_dict = model.state_dict()

    torch.save(state_dict, save_path)

    return save_path

def load_model(model: torch.nn.Module, model_type: str, filename: str, device: torch.device,
               models_path: str | Path = get_models_path()) -> torch.nn.Module:
    models_path = Path(models_path)
    load_folder_path = models_path / model_type / "models"
    load_path = load_folder_path / filename

    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    return model

def get_model_transforms(model_type: str, use_original_transforms:bool,clip_length_in_frames:int=None):
    
    if model_type == 'S3D':
        # Mean and std from KINETICS400_V1 weights for S3D
        KINETICS_S3D_400_MEAN = [0.43216, 0.394666, 0.37645]
        KINETICS_S3D_400_STD = [0.22803, 0.22145, 0.216989]

        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=KINETICS_S3D_400_MEAN, std=KINETICS_S3D_400_STD),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=KINETICS_S3D_400_MEAN, std=KINETICS_S3D_400_STD),
        ])

        if use_original_transforms:
            'TRANSFORMS OVERRIDE - USE ORIGINAL TRANSFORMS FROM MODEL PAPER (196x196 centre crop, no horizontal flip)'
    
            base_transforms = S3D_Weights.KINETICS400_V1.transforms()
            train_transforms = transforms.Compose([base_transforms, PermuteCTHWtoTCHW()])

            test_transforms = train_transforms
        if clip_length_in_frames is None:
            clip_length_in_frames = 16

    elif model_type == 'ResnetR2+1D_R18':
        RESNET_2PLUS1D_R18_400_MEAN = [0.43216, 0.394666, 0.37645]
        RESNET_2PLUS1D_R18_400_STD  = [0.22803, 0.22145, 0.216989]

        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=RESNET_2PLUS1D_R18_400_MEAN, std=RESNET_2PLUS1D_R18_400_STD),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=RESNET_2PLUS1D_R18_400_MEAN, std=RESNET_2PLUS1D_R18_400_STD),
        ])
        if use_original_transforms:
            'TRANSFORMS OVERRIDE - USE ORIGINAL TRANSFORMS FROM MODEL PAPER'
    
            base_transforms = R2Plus1D_18_Weights.KINETICS400_V1.transforms()
            train_transforms = transforms.Compose([base_transforms, PermuteCTHWtoTCHW()])

            test_transforms = train_transforms
        if clip_length_in_frames is None:
            clip_length_in_frames = 16

    elif model_type == "MViTv2-S":
        MVITV2_S_400_MEAN = [0.45, 0.45, 0.45]
        MVITV2_S_400_STD  = [0.225, 0.225, 0.225]

        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=MVITV2_S_400_MEAN, std=MVITV2_S_400_STD),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=MVITV2_S_400_MEAN, std=MVITV2_S_400_STD),
        ])
        if use_original_transforms:
            base_transforms = MViT_V2_S_Weights.KINETICS400_V1.transforms()
            train_transforms = transforms.Compose([base_transforms, PermuteCTHWtoTCHW()])

            test_transforms = train_transforms
        if clip_length_in_frames is None:
            clip_length_in_frames = 16

    elif model_type == "Swin3D_Tiny":
        SWIN3D_TINY_400_MEAN = [0.485, 0.456, 0.406] 
        SWIN3D_TINY_400_STD  = [0.229, 0.224, 0.225]

        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=SWIN3D_TINY_400_MEAN, std=SWIN3D_TINY_400_STD),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=SWIN3D_TINY_400_MEAN, std=SWIN3D_TINY_400_STD),
        ])

        if use_original_transforms:
            # TRANSFORMS OVERRIDE - USE ORIGINAL TRANSFORMS FROM MODEL PAPER
            base_transforms = Swin3D_T_Weights.KINETICS400_V1.transforms()
            train_transforms = transforms.Compose([base_transforms, PermuteCTHWtoTCHW()])
            test_transforms = train_transforms

        if clip_length_in_frames is None:
            clip_length_in_frames = 32

    elif model_type == "Swin3D_B_Kinetics_and_Imagenet22K":
        SWIN3D_BIG_400_IMAGENET_MEAN = [0.485, 0.456, 0.406] 
        SWIN3D_BIG_400_IMAGENET_STD  = [0.229, 0.224, 0.225]

        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=SWIN3D_BIG_400_IMAGENET_MEAN, std=SWIN3D_BIG_400_IMAGENET_STD),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=SWIN3D_BIG_400_IMAGENET_MEAN, std=SWIN3D_BIG_400_IMAGENET_STD),
        ])

        if use_original_transforms:
            # TRANSFORMS OVERRIDE - USE ORIGINAL TRANSFORMS FROM MODEL PAPER
            base_transforms = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1.transforms()
            train_transforms = transforms.Compose([base_transforms, PermuteCTHWtoTCHW()])
            test_transforms = train_transforms

        if clip_length_in_frames is None:
            clip_length_in_frames = 32

    elif model_type == "Swin3D_B_Kinetics":
        SWIN3D_BIG_400_MEAN = [0.485, 0.456, 0.406] 
        SWIN3D_BIG_400_STD  = [0.229, 0.224, 0.225]

        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=SWIN3D_BIG_400_MEAN, std=SWIN3D_BIG_400_STD),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=SWIN3D_BIG_400_MEAN, std=SWIN3D_BIG_400_STD),
        ])

        if use_original_transforms:
            # TRANSFORMS OVERRIDE - USE ORIGINAL TRANSFORMS FROM MODEL PAPER
            base_transforms = Swin3D_B_Weights.KINETICS400_V1.transforms()
            train_transforms = transforms.Compose([base_transforms, PermuteCTHWtoTCHW()])
            test_transforms = train_transforms

        if clip_length_in_frames is None:
            clip_length_in_frames = 32

    elif model_type == "VideoMAEv2-L" or model_type == 'VideoMAEv2-B' or model_type == 'VideoMAEv2-H' or 'InternVideo2-1B':
        VIDEOMAEV2_MEAN = [0.485, 0.456, 0.406]
        VIDEOMAEV2_STD  = [0.229, 0.224, 0.225]

        train_transforms = transforms.Compose([
            PeripheralEnvisionate(output_size=(224,224), band_fractions=(0.2,0.6,0.2)),               
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.Normalize(mean=VIDEOMAEV2_MEAN, std=VIDEOMAEV2_STD),
        ])
        test_transforms = transforms.Compose([
            PeripheralEnvisionate(output_size=(224,224), band_fractions=(0.2,0.6,0.2)),
            transforms.Normalize(mean=VIDEOMAEV2_MEAN, std=VIDEOMAEV2_STD),
        ])
        
        if use_original_transforms:
            train_transforms = transforms.Compose([
                transforms.Resize(224),                 # matches HF processor
                transforms.CenterCrop(224),             # matches HF processor
                transforms.RandomHorizontalFlip(p=0.5), # optional augmentation
                transforms.Normalize(mean=VIDEOMAEV2_MEAN, std=VIDEOMAEV2_STD),
            ])
            test_transforms = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=VIDEOMAEV2_MEAN, std=VIDEOMAEV2_STD),
            ])

        if clip_length_in_frames is None:
            clip_length_in_frames = 16
            
    elif model_type == "VideoMAE-B-Kinetics":
        VIDEOMAE_MEAN = [0.485, 0.456, 0.406]
        VIDEOMAE_STD  = [0.229, 0.224, 0.225]

        train_transforms = transforms.Compose([
            transforms.Resize(224,224),                
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.Normalize(mean=VIDEOMAE_MEAN, std=VIDEOMAE_STD),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(224,224),
            transforms.Normalize(mean=VIDEOMAE_MEAN, std=VIDEOMAE_STD),
        ])

        if use_original_transforms:
            train_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),  # optional augmentation
                transforms.Normalize(mean=VIDEOMAE_MEAN, std=VIDEOMAE_STD),
            ])
            test_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=VIDEOMAE_MEAN, std=VIDEOMAE_STD),
            ])

        if clip_length_in_frames is None:
            clip_length_in_frames = 16

    else:
        raise ValueError(f"No such Model of type <{model_type}>, Buster!")
    
    return train_transforms, test_transforms, clip_length_in_frames

def generate_confusion_matrix_dict(confusion_matrix, labels_index):
    """
    Save a confusion matrix in a human-readable JSON format.

    Args:
        confusion_matrix_array: NumPy array or list-of-lists, shape [num_classes, num_classes].
                                Row = true class, Col = predicted class.
        labels_index: Dictionary mapping class_name -> class_index.
        output_file_path: Path to save the JSON file.

    Returns:
        dict: JSON-serializable dictionary containing the labeled confusion matrix
              and per-class TP/FP/FN/TN statistics.
    """
    # Normalize to NumPy array of ints
    confusion_matrix_array = np.asarray(confusion_matrix, dtype=int)

    # Build index → name map (handles non-contiguous indices, e.g. "Splashscreens": 17)
    index_to_class_name = {idx: name for name, idx in labels_index.items()}
    max_index = max(index_to_class_name.keys())
    ordered_indices = list(range(max_index + 1))
    ordered_class_names = [index_to_class_name.get(i, f"unknown_{i}") for i in ordered_indices]

    # Validate shape
    if confusion_matrix_array.shape[0] != confusion_matrix_array.shape[1]:
        raise ValueError(f"confusion_matrix_array must be square, got shape {confusion_matrix_array.shape}")
    if confusion_matrix_array.shape[0] != len(ordered_indices):
        raise ValueError(
            f"confusion_matrix_array size ({confusion_matrix_array.shape[0]}) "
            f"does not match number of classes ({len(ordered_indices)})."
        )

    # Build nested dictionary version of the confusion matrix
    confusion_matrix_dict = {}
    for true_idx, true_class_name in zip(ordered_indices, ordered_class_names):
        confusion_matrix_dict[true_class_name] = {}
        for pred_idx, pred_class_name in zip(ordered_indices, ordered_class_names):
            confusion_matrix_dict[true_class_name][pred_class_name] = int(confusion_matrix_array[true_idx, pred_idx])

    return confusion_matrix_dict

def save_model_metadata(model_type: str, model_filename: str, fold_no: int, labels_index: Dict[str, int],
    dataset_info: Dict[str, int],evaluation_results, stratified_kfold_object: StratifiedKFold,
    seconds_taken_to_create_model: int, video_backend: str,
    train_transforms = None, training_hyperparameters: Optional[Dict[str, Any]] = None,
    per_epoch_metadata: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None, models_path: Optional[str | Path] = get_models_path()) -> Path:
    
    time_taken_to_create_model = convert_seconds_to_timestamp(seconds_taken_to_create_model)
    model_type_path = Path(models_path) / model_type

    avg_loss, accuracy, classification_report_dict, confusion_matrix, evaluation_metadata = evaluation_results
    # Extract StratifiedKFold config
    stratified_kfold_config = {"n_splits": stratified_kfold_object.n_splits, "shuffle": stratified_kfold_object.shuffle,
    "random_state": stratified_kfold_object.random_state}

    #Generate confusion matrix dictionary
    confusion_matrix_dict = generate_confusion_matrix_dict(confusion_matrix, labels_index)
    
    current_timestamp = get_current_time()
    metadata = {"model_type": model_type, "model_filename": model_filename, "fold_number": fold_no,"num_classes": len(labels_index),
    "dataset_sizes":dataset_info,"weighted_average_f1" : classification_report_dict["weighted avg"]["f1-score"],
    "average_loss": avg_loss, "accuracy": accuracy,"labels_index": labels_index,
    "classification_report": classification_report_dict, "stratified_kfold": stratified_kfold_config, 
    "time_taken_to_create_model":time_taken_to_create_model,"saved_at_roughly":current_timestamp,"video_backend":video_backend}

    if notes is not None:
        metadata["notes"] = notes

    if train_transforms is not None:
        metadata["train_transforms"] = extract_transforms_info(train_transforms)

    if training_hyperparameters is not None:
        metadata["training_hyperparameters"] = training_hyperparameters 

    metadata["confusion_matrix"] = confusion_matrix_dict

    metadata_filename_stem = Path(model_filename).stem.replace("_model","")
    metadata_filename = metadata_filename_stem + "_metadata.json"
    metadata_folder_path = model_type_path / "metadata"
    metadata_folder_path.mkdir(exist_ok=True,parents=True)
    metadata_path = metadata_folder_path / metadata_filename

    evaluation_metadata_filename = metadata_filename_stem + "_evaluation_metadata.json"
    evaluation_metadata_path = metadata_folder_path / evaluation_metadata_filename

    per_epoch_metadata_filename = metadata_filename_stem + "_epoch_level_metadata.json"
    per_epoch_metadata_path = metadata_folder_path / per_epoch_metadata_filename

    try:
        write_dict_to_json(metadata, metadata_path)
        write_dict_to_json(evaluation_metadata, evaluation_metadata_path)
        if per_epoch_metadata is not None:                           
            write_dict_to_json(per_epoch_metadata, per_epoch_metadata_path)

    except TypeError as e:
        print(e)
        print("Warning! Saving Model Metadata Failed.")
        return

    return metadata_path