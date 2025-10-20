## cf = classification
## Creates video labels for use with torchvision from the already classified video in folder 
from pathlib import Path
from path_configurator import * 
from bespoke_video_tools import *
from collections import Counter

def filter_for_minimum_video_count(minimum_number_of_videos: int, label_dictionary, class_map):
    '''ChatGPT Written but fixed up by me'''
    if minimum_number_of_videos > 0:
        # Count how many videos per class index
        class_counts = Counter(label_dictionary.values())

        # Keep only classes with >= minimum_number_of_videos
        valid_indices = {index for index, count in class_counts.items() if count >= minimum_number_of_videos}

        # Filter down to only valid classes
        filtered_label_dictionary = {key: value for key, value in label_dictionary.items() if value in valid_indices}
        filtered_class_map = {name: index for name, index in class_map.items() if index in valid_indices}

        #Re-index to make indices dense and continuous
        reindex_map = {old_index: new_index for new_index, old_index in enumerate(sorted(valid_indices))}
        filtered_label_dictionary = {key: reindex_map[value] for key, value in filtered_label_dictionary.items()}
        filtered_class_map = {name: reindex_map[index] for name, index in filtered_class_map.items()}

        return filtered_label_dictionary, filtered_class_map

    return label_dictionary, class_map
    
    
def create_label_dictionary(classes_folder_path: Path, option="broad",allow_duplicates=False,verbose=False,minimum_number_of_videos=0,classes_to_remove=[],classes_to_remove_fully=[]):
    label_dictionary = {}
    class_map = {}
    if option not in ['broad','specific']:
        raise ValueError("Mate you bloody didn't put in the right args")
    
    broad_classes = sorted([file for file in classes_folder_path.iterdir() if file.is_dir()])
    specific_classes = sorted([file for file in classes_folder_path.rglob('*') if file.is_dir() and file not in broad_classes])

    if option=='broad':
        classes = broad_classes
        #Remove unwanted classes
        classes = [class_path for class_path in classes if class_path.name not in classes_to_remove]
    elif option == 'specific':
        classes = specific_classes
        #Remove unwanted classes
        classes = [class_path for class_path in classes if class_path.parent.name not in classes_to_remove]
    
    observed_video_set = set()
    filenamefinder_dictionary = {}
    for class_path in classes:
        class_name = class_path.name
        if option=='specific':
            class_name = str(class_path.parent.name) + ' - ' + class_name
        index = max(class_map.values(),default=-1) + 1
        if class_name not in classes_to_remove_fully:
            class_map[class_name] = index

        for potential_video_file in class_path.rglob('*'):
            if check_file_is_video(potential_video_file):
                if allow_duplicates == False:
                    if potential_video_file.name in observed_video_set:
                        if verbose:
                            print("DUPLICATE: ",potential_video_file)
                        try:
                            rel_path_to_pop = filenamefinder_dictionary[potential_video_file.name]
                            label_dictionary.pop(rel_path_to_pop)
                        except KeyError:
                            if verbose:
                                print("KeyError: Probs already popped")
                        continue
                rel_path = potential_video_file.relative_to(classes_folder_path)

                #Custom relabelling stuff, otherwise blank condition (if False)
                # if rel_path.parent.name == 'Handball Receive':
                #     label_dictionary[str(rel_path)] = class_map['Handballs']

                # if rel_path.parent.parent.name == 'Shots':
                #     label_dictionary[str(rel_path)] = class_map['Kicks']

                #else:

                observed_video_set.add(potential_video_file.name)
                filenamefinder_dictionary[potential_video_file.name] = str(rel_path)

                #Remove all instances of videos found in class folder (putting in classes to remove will preserve duplicates in other classes as e.g. 'Replays' folders goes unseen)
                if class_name in classes_to_remove_fully:
                    continue

                label_dictionary[str(rel_path)] = index

    label_dictionary, class_map = filter_for_minimum_video_count(minimum_number_of_videos, label_dictionary, class_map)
    return label_dictionary, class_map 

def create_focused_label_dictionary(classes_folder_path: Path, option="broad",allow_duplicates=False,verbose=False,minimum_number_of_videos=0,classes_to_remove=[],classes_to_remove_fully=[]):
    label_dictionary = {}
    class_map = {"Scores":0}
    if option not in ['broad','specific']:
        raise ValueError("Mate you bloody didn't put in the right args")
    
    broad_classes = sorted([file for file in classes_folder_path.iterdir() if file.is_dir()])
    specific_classes = sorted([file for file in classes_folder_path.rglob('*') if file.is_dir() and file not in broad_classes])

    if option=='broad':
        classes = broad_classes
        #Remove unwanted classes
        classes = [class_path for class_path in classes if class_path.name not in classes_to_remove]
    elif option == 'specific':
        classes = specific_classes
        #Remove unwanted classes
        classes = [class_path for class_path in classes if class_path.parent.name not in classes_to_remove]
    
    observed_video_set = set()
    filenamefinder_dictionary = {}
    for class_path in classes:
        class_name = class_path.name
        if option=='specific':
            class_name = str(class_path.parent.name) + ' - ' + class_name
        index = max(class_map.values(),default=-1) + 1
        if class_name not in classes_to_remove_fully and class_name != 'Shots' and class_name != 'Behinds' and class_name != "Goals":
            class_map[class_name] = index

        for potential_video_file in class_path.rglob('*'):
            if check_file_is_video(potential_video_file):
                if allow_duplicates == False:
                    if potential_video_file.name in observed_video_set:
                        if verbose:
                            print("DUPLICATE: ",potential_video_file)
                        try:
                            rel_path_to_pop = filenamefinder_dictionary[potential_video_file.name]
                            label_dictionary.pop(rel_path_to_pop)
                        except KeyError:
                            if verbose:
                                print("KeyError: Probs already popped")
                        continue
                rel_path = potential_video_file.relative_to(classes_folder_path)

                observed_video_set.add(potential_video_file.name)
                filenamefinder_dictionary[potential_video_file.name] = str(rel_path)

                #Remove all instances of videos found in class folder (putting in classes to remove will preserve duplicates in other classes as e.g. 'Replays' folders goes unseen)
                if class_name in classes_to_remove_fully:
                    continue
                
                if class_name == 'Shots':
                    label_dictionary[str(rel_path)] = class_map['Kicks']
                elif class_name == 'Behinds' or class_name == 'Goals':
                    label_dictionary[str(rel_path)] = class_map['Scores']
                # elif rel_path.parent.name == 'Handball Receive':
                #     label_dictionary[str(rel_path)] = class_map['Handball Receive']

                else:
                    label_dictionary[str(rel_path)] = index

    label_dictionary, class_map = filter_for_minimum_video_count(minimum_number_of_videos, label_dictionary, class_map)
    return label_dictionary, class_map 

def create_other_paper_label_dictionary(classes_folder_path: Path, option="broad",allow_duplicates=False,verbose=False,minimum_number_of_videos=0,classes_to_remove=[]):
    label_dictionary = {}
    class_map = {'Contested Marks':0}
    if option not in ['broad','specific']:
        raise ValueError("Mate you bloody didn't put in the right args")
    
    broad_classes = sorted([file for file in classes_folder_path.iterdir() if file.is_dir()])
    specific_classes = sorted([file for file in classes_folder_path.rglob('*') if file.is_dir() and file not in broad_classes])

    if option=='broad':
        classes = broad_classes
        #Remove replays 
        classes = [class_path for class_path in classes if class_path.name not in classes_to_remove]
    elif option == 'specific':
        classes = specific_classes
        #Remove replays 
        classes = [class_path for class_path in classes if class_path.parent.name not in classes_to_remove]
    
    observed_video_set = set()
    filenamefinder_dictionary = {}
    for index, class_path in enumerate(classes,start=1):
        class_name = class_path.name
        if option=='specific':
            class_name = str(class_path.parent.name) + ' - ' + class_name
        class_map[class_name] = index

        for potential_video_file in class_path.rglob('*'):
            if check_file_is_video(potential_video_file):
                if allow_duplicates == False:
                    if potential_video_file.name in observed_video_set:
                        if verbose:
                            print("DUPLICATE: ",potential_video_file)
                        try:
                            rel_path_to_pop = filenamefinder_dictionary[potential_video_file.name]
                            label_dictionary.pop(rel_path_to_pop)
                        except KeyError:
                            if verbose:
                                print("KeyError: Probs already popped")
                        continue
                rel_path = potential_video_file.relative_to(classes_folder_path)
                #Custom relabelling stuff, otherwise blank condition (if False)

                if rel_path.parent.parent.name == 'Marks' and rel_path.parent.name.startswith('Contested'):
                    label_dictionary[str(rel_path)] = class_map['Contested Marks']
                elif rel_path.parent.parent.name == 'Marks' and rel_path.parent.name.startswith('Semi-Contested'):
                    continue
                elif rel_path.parent.parent.name == 'Establishing Shots' and rel_path.parent.name.startswith('Players'):
                    continue
                else:
                    label_dictionary[str(rel_path)] = index

                observed_video_set.add(potential_video_file.name)
                filenamefinder_dictionary[potential_video_file.name] = str(rel_path)

    label_dictionary, class_map = filter_for_minimum_video_count(minimum_number_of_videos, label_dictionary, class_map)
    return label_dictionary, class_map 


if __name__ == '__main__':
    label_dict, label_index = create_label_dictionary(get_classes_path(),classes_to_remove=['Fights','Free Kicks'],classes_to_remove_fully=['Replays'])
    label_dict, label_index = create_focused_label_dictionary(get_classes_path(),classes_to_remove=['Fights','Free Kicks','Out of Bounds'],classes_to_remove_fully=['Replays'])

    #KICKS ONLY SPECIFIC 
    #label_dict, label_index = create_label_dictionary(get_classes_path(),option='specific',minimum_number_of_videos=5,classes_to_remove=["Goal Umpire Signal","Goals","Ground Contests","Hit Outs","Line Ups","Out of Bounds","Pick Ups","Running","Shots","Splashscreens","Stoppages","Tackles",'Fights','Replays','Free Kicks','Aerial Contests','Behinds','Replays','Marks','Handballs','Establishing Shots'])
    #For comparing to other AFL paper, reduces data quality / breadth
    #label_dict, label_index = create_other_paper_label_dictionary(get_library_path()/"Classes",classes_to_remove=["Goal Umpire Signal","Goals","Ground Contests","Hit Outs","Line Ups","Out of Bounds","Pick Ups","Running","Shots","Splashscreens","Stoppages","Tackles",'Fights','Replays','Free Kicks','Aerial Contests','Behinds'])
    write_dict_to_json(label_dict, get_label_dictionary_path())
    write_dict_to_json(label_index, get_label_index_path())