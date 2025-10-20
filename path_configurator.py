import json 
import platform
from pathlib import Path
from bespoke_tools import * 

# NOTE Windows support in this not currently included but not that hard to include (if platform.system()==)
# NOTE These variables are only for writing when running the program directly, reading path comes from paths.json
linux_library_path = "<ENTER LIBRARY PATH HERE>"
mac_library_path = "<ENTER LIBRARY PATH HERE>" #(where you chuck all the dataset clips)

mac_project_storage_path = "<ENTER PROJECT STORAGE PATH HERE>" #where you put all the code
linux_project_storage_path = "<ENTER PROJECT STORAGE PATH HERE>"

if platform.system() == "Darwin":  # MacOS
    library_path = mac_library_path
    project_storage_path = mac_project_storage_path
    paths_head_path = "paths_mac.json"
elif platform.system() == "Linux":
    library_path = linux_library_path
    project_storage_path = linux_project_storage_path
    paths_head_path = "paths_linux.json"
else:
    raise RuntimeError(f"Yeah nah, you can't use this OS {platform.system()} – you haven't built for this one lol")

classes_head_path = "Classes"
classes_path = library_path + "/" + classes_head_path

clip_folder_base_path = library_path
clip_folder_head_path = "<FOLDER OF CUT-UP VIDEO TO CLASSIFY>"
clip_folder_path = clip_folder_base_path + "/" + clip_folder_head_path

label_dictionary_head_path = "label_dict.json"
label_dictionary_path = project_storage_path + "/" + label_dictionary_head_path

label_index_head_path = "label_index.json"
label_index_path = project_storage_path + "/" + label_index_head_path

paths_path = project_storage_path + "/" + paths_head_path

example_gifs_folder_head_path = "batch_gifs_example"
example_gifs_folder_path = project_storage_path + "/" + example_gifs_folder_head_path

models_folder_head_path = "model_library"
models_folder_path = project_storage_path + "/" + models_folder_head_path

glance_info_folder_head_path = "glance"
glance_info_folder_path = project_storage_path + "/" + glance_info_folder_head_path

# Downloaded models/weights folders -- unused in GitHub version (for testing new models, feel free to utilise)
downloaded_models_folder_head_path = "downloaded_models"
downloaded_models_folder_path = project_storage_path + "/" + downloaded_models_folder_head_path

downloaded_weights_folder_head_path = "downloaded_weights"
downloaded_weights_folder_path = project_storage_path + "/" + downloaded_weights_folder_head_path

# Font file paths
regular_font_path = ""
bold_font_path = ""
italic_font_path = ""
bold_italic_font_path = ""

def write_paths(paths_path=paths_path, project_storage_path = project_storage_path, library_path=library_path, classes_path=classes_path, clip_folder_path=clip_folder_path,
                 label_dictionary_path=label_dictionary_path, 
                label_index_path=label_index_path, example_gifs_folder_path=example_gifs_folder_path,
                models_folder_path=models_folder_path, glance_info_folder_path=glance_info_folder_path,
                downloaded_models_folder_path=downloaded_models_folder_path, downloaded_weights_folder_path=downloaded_weights_folder_path,
                regular_font_path=regular_font_path, bold_font_path=bold_font_path, italic_font_path=italic_font_path, bold_italic_font_path=bold_italic_font_path) -> None:
    paths_dict = {
        "ProjectFolder": str(project_storage_path),
        "LibraryFolder": str(library_path),
        "ClassesFolder": str(classes_path),
        "ClipSourceFolder": str(clip_folder_path),
        "LabelDictionaryFile": str(label_dictionary_path),
        "LabelIndexFile": str(label_index_path),
        "ExampleGifsFolder": str(example_gifs_folder_path),
        "ModelsFolder": str(models_folder_path),
        "GlanceInfoFolder": str(glance_info_folder_path),
        "DownloadedModelsFolder": str(downloaded_models_folder_path),
        "DownloadedWeightsFolder": str(downloaded_weights_folder_path),
        "RegularFontFile": str(regular_font_path),
        "BoldFontFile": str(bold_font_path),
        "ItalicFontFile": str(italic_font_path),
        "BoldItalicFontFile": str(bold_italic_font_path),
    }

    write_dict_to_json(paths_dict, paths_path)

def get_project_path(paths_path=paths_path) -> Path:
    with open(paths_path, 'r') as file:
        paths_dict = json.load(file)
    return Path(paths_dict['ProjectFolder'])

def get_library_path(paths_path=paths_path) -> Path:
    with open(paths_path, 'r') as file:
        paths_dict = json.load(file)
    return Path(paths_dict['LibraryFolder'])

def get_classes_path(paths_path=paths_path) -> Path:
    with open(paths_path, 'r') as file:
        paths_dict = json.load(file)
    return Path(paths_dict['ClassesFolder'])

def get_clip_source_path(paths_path=paths_path) -> Path:
    with open(paths_path, 'r') as file:
        paths_dict = json.load(file)
    return Path(paths_dict['ClipSourceFolder'])

def get_label_dictionary_path(paths_path=paths_path) -> Path:
    with open(paths_path, 'r') as file:
        paths_dict = json.load(file)
    return Path(paths_dict['LabelDictionaryFile'])

def get_label_index_path(paths_path=paths_path) -> Path:
    with open(paths_path, 'r') as file:
        paths_dict = json.load(file)
    return Path(paths_dict['LabelIndexFile'])

def get_example_gifs_path(paths_path=paths_path) -> Path:
    with open(paths_path, 'r') as file:
        paths_dict = json.load(file)
    return Path(paths_dict['ExampleGifsFolder'])

def get_models_path(paths_path=paths_path) -> Path:
    with open(paths_path, 'r') as file:
        paths_dict = json.load(file)
    return Path(paths_dict['ModelsFolder'])

def get_glance_info_path(paths_path=paths_path) -> Path:
    with open(paths_path, 'r') as file:
        paths_dict = json.load(file)
    return Path(paths_dict['GlanceInfoFolder'])

def get_downloaded_models_folder_path(paths_path=paths_path) -> Path:
    with open(paths_path, 'r') as file:
        paths_dict = json.load(file)
    return Path(paths_dict['DownloadedModelsFolder'])

def get_downloaded_weights_path(paths_path=paths_path) -> Path:
    with open(paths_path, 'r') as file:
        paths_dict = json.load(file)
    return Path(paths_dict['DownloadedWeightsFolder'])

def get_regular_font_path(paths_path=paths_path) -> Path:
    with open(paths_path, 'r') as file:
        paths_dict = json.load(file)
    return Path(paths_dict['RegularFontFile'])

def get_bold_font_path(paths_path=paths_path) -> Path:
    with open(paths_path, 'r') as file:
        paths_dict = json.load(file)
    return Path(paths_dict['BoldFontFile'])

def get_italic_font_path(paths_path=paths_path) -> Path:
    with open(paths_path, 'r') as file:
        paths_dict = json.load(file)
    return Path(paths_dict['ItalicFontFile'])

def get_bold_italic_font_path(paths_path=paths_path) -> Path:
    with open(paths_path, 'r') as file:
        paths_dict = json.load(file)
    return Path(paths_dict['BoldItalicFontFile'])

if __name__ == '__main__':
    write_paths()
    print(f"Video Library Location: {get_library_path()}")
    print(f"Video Clips to Classify Location: {get_clip_source_path()}")
    print(f"Example GIFs folder Location: {get_example_gifs_path()}")
    print(f"Models folder Location: {get_models_path()}")
    print(f"Glance Information folder Location: {get_glance_info_path()}")
    print(f"Downloaded Models folder Location: {get_downloaded_models_folder_path()}")
    print(f"Downloaded Weights folder Location: {get_downloaded_weights_path()}")
    print(f"Regular Font File Location: {get_regular_font_path()}")
    print(f"Bold Font File Location: {get_bold_font_path()}")
    print(f"Italic Font File Location: {get_italic_font_path()}")
    print(f"Bold Italic Font File Location: {get_bold_italic_font_path()}")
    print_divider()
    print("Oi, be sure to run cf_video_labeller if you want this to be fed in to the model btw")
