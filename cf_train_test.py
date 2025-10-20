import os
# CPU FALLBACK  
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import time
import platform
import random
from bespoke_ffmpeg_silencer import * 
from tqdm import tqdm
from torch.utils.data import DataLoader
transforms = silent_import("torchvision.transforms.v2")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from bespoke_tools import *
from cf_video_labeller import *
from bespoke_ai_training_tools import * 
from cf_dataset import CustomVideoDataset, generate_class_weights
from cf_glance import * 
from cf_model import * 
from cf_early_stop import CustomEarlyStopper
from cf_assert import * 
from cf_scheduler import create_scheduler

@time_a_function()
def train_single_epoch(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR,
                       criterion, device: torch.device) -> None:
    model.train()
    running_loss = 0.0
    num_samples = 0

    dataloader_avec_progress_bar = tqdm(dataloader, desc="Training", unit="batch",colour='cyan',dynamic_ncols=True)
    for batch_frames, batch_labels, batch_metadata in dataloader_avec_progress_bar:
        if batch_frames is None and batch_labels is None and batch_metadata is None:
            continue

        batch_frames = batch_frames.to(device)  # [B, T, C, H, W]
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        model_signature = inspect.signature(model.forward)
        if "pixel_values" in model_signature.parameters:
            outputs = model(pixel_values=batch_frames).logits  # Direct Hugging Face implementation
        else:
            batch_frames = batch_frames.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]
            outputs = model(batch_frames)               # TorchVision / OpenGVLab

        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        current_learning_rate = scheduler.get_last_lr()[0]

        batch_size = batch_frames.size(0)
        num_samples += batch_size
        running_loss += loss.item() * batch_size
        avg_loss = running_loss / num_samples

        dataloader_avec_progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}","Average Loss": f"{avg_loss:.4f}", "Learning Rate":f"{current_learning_rate:.2e}"})
        #print(f"Batch loss: {loss.item():.4f} in interation {iteration} of {len(dataloader)}")

    print(f"Finished training epoch with train Loss: {avg_loss:.4f}")

@time_a_function()
def evaluate(model, dataloader, criterion, device, labels_index):
    model.eval()
    running_loss = 0.0
    num_samples = 0
    all_predictions = []
    all_labels = []
    all_clip_metadata = []

    with torch.no_grad():
        dataloader_avec_progress_bar = tqdm(dataloader, desc="Testing", unit="batch",colour='cyan',dynamic_ncols=True)
        for batch_frames, batch_labels, batch_metadata in dataloader_avec_progress_bar:
            batch_frames = batch_frames.to(device)
            batch_labels = batch_labels.to(device)  

            model_signature = inspect.signature(model.forward)
            if "pixel_values" in model_signature.parameters:
                outputs = model(pixel_values=batch_frames).logits
            else:
                batch_frames = batch_frames.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]
                outputs = model(batch_frames) # [B, num_classes]

            loss = criterion(outputs, batch_labels)
            batch_size = batch_frames.size(0)
            num_samples += batch_size
            running_loss += loss.item() * batch_size
            avg_loss = running_loss / num_samples

            #CLIP METADATA
            all_clip_metadata.extend(extract_batch_evaluation_metadata(outputs, batch_size, batch_labels, batch_metadata, labels_index))

            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch_labels.cpu().tolist())

            dataloader_avec_progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}","Average Loss": f"{avg_loss:.4f}"})

    accuracy = accuracy_score(all_labels, all_predictions)

    # Get label names ordered by index
    sorted_labels = [None] * len(labels_index)
    for label_name, idx in labels_index.items():
        sorted_labels[idx] = label_name

    labels = list(range(len(sorted_labels)))
    report = classification_report(all_labels,all_predictions,labels=labels,target_names=sorted_labels,zero_division=0,digits=4,output_dict=False)
    dict_report = classification_report(all_labels,all_predictions,labels=labels,target_names=sorted_labels,zero_division=0,digits=4,output_dict=True)

    confusion = confusion_matrix(all_labels, all_predictions, labels=labels)

    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    print("Per-class metrics:\n", report)
    return avg_loss, accuracy, dict_report, confusion, all_clip_metadata

if __name__ == '__main__':

    #supported_model_types = ["S3D","Swin3D_B_Kinetics_and_Imagenet22K","Swin3D_Tiny","MViTv2-S","ResnetR2+1D_R18","Swin3D_B_Kinetics"
    # ,"VideoMAEv2-L", "VideoMAEv2-B", "VideoMAEv2-H","VideoMAE-B-Kinetics"]
    model_type = "VideoMAEv2-H" 
    use_original_transforms = True
    loss_type = 'weighted_cross_entropy'

    '''actual evaluation recipes (as documented on each weight card) for the current torchvision video models on Kinetics-400:
	•	R(2+1)D-18 → frame_rate = 15,   clips_per_video = 5,  clip_len = 16.  
	•	S3D        → frame_rate = 15,   clips_per_video = 1,  clip_len = 128.  
	•	MViT-V1-B  → frame_rate = 7.5,  clips_per_video = 5,  clip_len = 16.  
	•	MViT-V2-S  → frame_rate = 7.5,  clips_per_video = 5,  clip_len = 16.
	•	Swin3D-T   → frame_rate = 15,   clips_per_video = 12, clip_len = 32.  
	•	Swin3D-S   → frame_rate = 15,   clips_per_video = 12, clip_len = 32.  
	•	Swin3D-B   → frame_rate = 15,   clips_per_video = 12, clip_len = 32.  
'''

    train_transforms, test_transforms,clip_length_in_frames = get_model_transforms(model_type,use_original_transforms,16)

    # Load your labels dict & index
    labels_dict = load_json(get_label_dictionary_path())  # video_path -> label
    labels_index = load_json(get_label_index_path())      # label -> int index
    assert_contiguous_labels_index(labels_index)
    
    # Prepare video paths and their labels for stratified splitting
    video_paths = list(labels_dict.keys())
    video_labels = [labels_dict[vp] for vp in video_paths]

    #YO THIS IS IMPORTANT!! 🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉
    #===============================
    k_fold_folds_no = 5
    patience = 12
    #===============================

    # I love a bit of deterministic evaluation 🎲🎲🎲🎲🎲🎲🎲🎲🎲
    random_seed = 69
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    skf = StratifiedKFold(n_splits=k_fold_folds_no, shuffle=True, random_state=random_seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(random_seed) #🎲
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    #🍉🍉 Device Capability Stuff 🍉🍉
    #===============================
    big_boppa_models = ["VideoMAEv2-H"]
    medium_boppa_models = ['VideoMAEv2-L']
    prefetch_factor = None

    if platform.system() == "Linux":
        num_cpu_workers = 16
        if model_type in big_boppa_models:
            num_cpu_workers = 12
    else:
        num_cpu_workers = 4
    batch_size = 8
    if model_type in big_boppa_models:
        batch_size = 2
    if model_type in medium_boppa_models:
        batch_size = 3
    #===============================

    classes_path = get_library_path() / "Classes"

    fold_no = 0
    for train_idx, val_idx in skf.split(video_paths, video_labels):
        fold_no += 1
        
        print_divider()
        print(f"                        ❋ Fold {fold_no} ❋")
        print_divider()

        train_videos = {video_paths[i]: video_labels[i] for i in train_idx}
        val_videos = {video_paths[i]: video_labels[i] for i in val_idx}

        video_backend = 'pyav'
        print("Creating Training Dataset...")
        train_dataset = CustomVideoDataset(classes_path, train_videos, clip_length_in_frames=clip_length_in_frames, transforms=train_transforms,video_backend=video_backend)
        print("Creating Testing Dataset...")
        val_dataset = CustomVideoDataset(classes_path, val_videos, clip_length_in_frames=clip_length_in_frames, transforms=test_transforms,video_backend=video_backend)
        #🍉 SEE Training dataset TEST 🍉
        glance_dataset(train_dataset, get_glance_info_path(), "train_dataset_samples.json")
        glance_dataset(val_dataset, get_glance_info_path(), "test_dataset_samples.json")

        print("Preparing Dataloaders...")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_cpu_workers, collate_fn=custom_collate,prefetch_factor=prefetch_factor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_cpu_workers, collate_fn=custom_collate)

        #🍉 SEE VIDEOS TEST 🍉
        glance_gifs_from_dataloader(train_loader,1)
        
        print("Creating model...")

        ## Model creation
        model = create_model(model_type, len(labels_index),device)

        #Set epoch start to 0 if you want to see blind testing (funny)
        epoch_start = 1
        load_existing_model = False
        test_model_first = False
        if load_existing_model:
            existing_model_filename = "fold2_VideoMAEv2-H_model.pth"
            model = load_model(model, model_type, existing_model_filename, device)
            epoch_start = 6
            test_model_first = True
            print(f"[CONTINUE] Model training resumed @epoch-{epoch_start} using model at location {existing_model_filename}")

        ##==========🧮 YOOOOO important variables  🧮==============##
        number_of_epochs=100 #Inclusive because numbers
        if model_type in big_boppa_models:
            peak_learning_rate = 7.375e-6
        elif model_type in medium_boppa_models:
            peak_learning_rate = 4.375e-5
        else:
            peak_learning_rate = 1e-4
        weight_decay = 0.05

        optimizer = torch.optim.AdamW(model.parameters(), lr=peak_learning_rate,weight_decay=weight_decay)
        #LEARNING RATE COSINE DECAY STUFF
        warm_up_epochs = 2
        steps_per_epoch = len(train_loader)
        number_of_total_steps = steps_per_epoch * number_of_epochs  # planned max, even if you stop early
        number_of_warm_up_steps = int(len(train_loader) * warm_up_epochs)  # fixed 2-epoch warmup
        decay_type = 'cosine'

        scheduler = create_scheduler(optimizer, number_of_warm_up_steps, number_of_total_steps,decay_type)

        criterion = torch.nn.CrossEntropyLoss()
        if loss_type == 'weighted_cross_entropy':
            class_weights = generate_class_weights(train_dataset,device,normalize=True)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        #Early Stop stuff
        # --- 
        early_stop_metric = 'weighted_f1'

        average_loss_early_stopper = CustomEarlyStopper(patience=patience, min_delta=0.001,mode='min')
        weighted_f1_early_stopper = CustomEarlyStopper(patience=patience, min_delta=0.001,mode='max')
        if early_stop_metric == 'average_loss':
            early_stopper = average_loss_early_stopper
            other_early_stopper = weighted_f1_early_stopper
            other_early_stop_metric = 'weighted_f1'
        elif early_stop_metric == 'weighted_f1':
            early_stopper = weighted_f1_early_stopper
            other_early_stopper = average_loss_early_stopper
            other_early_stop_metric = 'average_loss'
        else:
            raise ValueError(f"Mate what is that early stop metric?! {early_stop_metric} - incompatible")
        best_model = None
        best_epoch = None 
        best_results = None
        best_epochs_other_metric = None
        best_other_metric_epoch = None
        # ---
        model_filename_to_save = f"fold{fold_no}_{model_type}_model.pth"

        train_test_start_time = time.time()
        try:
            #Epoch 0 no fine-tune
            for epoch in range(epoch_start,number_of_epochs+1):
                if epoch == 0 or test_model_first:
                    print("Model Initialised — evaluating initial performance without fine-tuning")
                    test_model_first = False
                else:
                    print(f"Currently training Epoch No. {epoch} of {number_of_epochs} in Fold {fold_no} of {k_fold_folds_no}")
                    train_single_epoch(model, train_loader, optimizer, scheduler, criterion, device)
                    print(f"Evaluating performance of Epoch No. {epoch}")
                per_epoch_evaluation_results = evaluate(model, val_loader, criterion, device,labels_index)

                #Early stop stuff
                #============================================
                epoch_average_loss = per_epoch_evaluation_results[0]
                epoch_weighted_f1 = per_epoch_evaluation_results[2]["weighted avg"]["f1-score"]
                if early_stop_metric == 'weighted_f1':
                    epoch_metric_result = epoch_weighted_f1
                    epoch_other_metric_result = epoch_average_loss
                elif early_stop_metric == 'average_loss':
                    epoch_metric_result = epoch_average_loss
                    epoch_other_metric_result = epoch_weighted_f1
                
                should_stop_training = early_stopper.step(epoch_metric_result)
                other_early_stop_step = other_early_stopper.step(epoch_other_metric_result)

                #Save best model during training in case of manual stop
                if best_model is None or epoch_metric_result == early_stopper.best:
                    best_model = {k: v.cpu() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    best_results = per_epoch_evaluation_results
                    best_epochs_other_metric = epoch_other_metric_result
                if epoch_other_metric_result == other_early_stopper.best:
                    best_other_metric_epoch = epoch

                if should_stop_training:
                    print(f"[EARLY STOP] Training stopped during epoch No. {epoch}. Model saved from iteration {best_epoch}.")
                    model.load_state_dict(best_model)
                    break

                print(f"Current Best Model was achieved @epoch-{best_epoch} with {early_stop_metric.replace('_'," ")} of {early_stopper.best:.4f} & {other_early_stop_metric.replace('_'," ")} of {best_epochs_other_metric:.4f}")
                print(f"Current Best {other_early_stop_metric.replace('_'," ")} was achieved @epoch-{best_other_metric_epoch} with value {other_early_stopper.best:.4f}")
                print(f"Patience Meter: {early_stopper.bad_epochs}/{early_stopper.patience}")
                #SAVE AFTER EACH EPOCH
                model_saved_path = save_model(model, model_type, model_filename_to_save)
                #============================================
        except KeyboardInterrupt:
            print(f"\n[MANUAL STOP] You stopped the training before full completion.")
            if best_model is not None:
                model.load_state_dict(best_model)
                print(f"Model saved from iteration {best_epoch}.")
            else:
                print("No model produced, training was immediately cancelled.")
                should_quit_all_folds_if_no_model = False
                if should_quit_all_folds_if_no_model:
                    print("[QUIT] Quitting program because no model was produced.")
                    quit()
                else:
                    continue

        train_test_end_time = time.time()
        total_train_test_time = train_test_end_time - train_test_start_time

        fold_evaluation_result = best_results

        # Save model after training each fold
        model_saved_path = save_model(model, model_type, model_filename_to_save)

        training_hyperparameters = {"max_learning_rate":peak_learning_rate,"final_learning_rate": optimizer.param_groups[0]['lr'], "batch_size": train_loader.batch_size,
            "epochs": number_of_epochs,"optimizer": optimizer.__class__.__name__,"loss_function": criterion.__class__.__name__, "loss_class_weights":  criterion.weight.tolist() if criterion.weight is not None else None,
            "num_workers": train_loader.num_workers}
        if decay_type != None and decay_type != 'no_decay':
            training_hyperparameters['decay_type'] = decay_type
            training_hyperparameters['weight_decay'] = weight_decay

        dataset_info = {"training_videos_count":len(train_videos), "training_clips_count":len(train_dataset),
                           "testing_videos_count": len(val_videos), "testing_clips_count": len(val_dataset)}
        # Save metadata
        metadata_path = save_model_metadata(model_type=model_type,model_filename=model_filename_to_save,fold_no=fold_no,labels_index=labels_index, dataset_info = dataset_info,
                                            evaluation_results = fold_evaluation_result, stratified_kfold_object=skf, seconds_taken_to_create_model=total_train_test_time,
                                            video_backend=video_backend,train_transforms=train_transforms,training_hyperparameters=training_hyperparameters,
                                            notes=f"Torchvision {model_type} Video Classifier Pre-Trained on Kinetics-400 Dataset & Fine-tuned using custom AFL Dataset")

        print(f"Saved model for fold {fold_no} at {model_saved_path}")
        print(f"Saved metadata for fold {fold_no} at {metadata_path}")

        #🍉🍉🍉Cease after one fold !!!!🍉🍉🍉
        #================================
        # print("Quitting after one fold, considering the model isn't done")
        # quit()
        #================================
        print_divider()
        print(f"[FOLD {fold_no}] Training and Testing Complete")
        if fold_no < k_fold_folds_no:
            print(f"[INITIALISING] Training Initialising for Fold No. {fold_no + 1}")
        print_divider()

    print("TRAINING and TESTING DONE WOOOOOOO")