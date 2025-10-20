from path_configurator import * 
from bespoke_ai_training_tools import * 

def glance_dataset(dataset, glance_info_path: str | Path, filename: str | Path):
    
    glance_info_path = Path(glance_info_path)
    Path.mkdir(glance_info_path,exist_ok=True)
    filename = str(filename)
    
    sample_keys = [
        "relative_video_path",
        "start_index",
        "stride",
        "label",
        "major_category",
        "minor_category",
        "filename",
        "total_frames"
    ]
    # Convert tuples to dicts
    samples = [dict(zip(sample_keys, sample)) for sample in dataset.samples]
    print(f"Samples saved to file {glance_info_path}/{filename}!")
    write_dict_to_json(samples, glance_info_path / filename,print_message=False)

@time_a_function()
def glance_gifs_from_dataloader(dataloader,number_of_gif_batches):
    
    labels_index = load_json(get_label_index_path())
    mean, std = get_normalize_mean_std(dataloader.dataset) 
    iterator_dataloader = iter(dataloader)
    output_directory = get_example_gifs_path()
    Path.mkdir(output_directory,exist_ok=True)
    clear_directory_shallow(output_directory)
    for i in range(number_of_gif_batches):
        batch_frames, batch_labels, batch_metadata = next(iterator_dataloader)
        save_batch_gif(batch_frames, batch_metadata, batch_labels, labels_index, mean, std, output_dir=output_directory, assumed_fps=25)
    