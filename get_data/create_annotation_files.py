import ast
import json
from collections import namedtuple

import pandas as pd


def get_frame_path(participant_id, video_id, frame_num):
    path = f"/home/sxa180157/ptg/epic_kitchens_100/object_detection_labels/EPIC-KITCHENS/{participant_id}/object_detection_images/{video_id}/{frame_num}.txt"
    return path


def get_frame_size(video_info_file):
    video_info = pd.read_csv(video_info_file)
    ResolutionData = namedtuple('ResolutionData', 'width height')
    # label = LabelData(object_class=object_class, bounding_box=bb)

    resolution_dict = {
        row["video"]: ResolutionData(int(row['resolution'].split("x")[0]), int(row['resolution'].split("x")[1])) for
        index, row in video_info.iterrows()}
    return resolution_dict


def get_string_from_list(list):
    this_str = "[ "
    for each_word in list:
        this_str += f'\"{str(each_word)}\", '
    this_str += "]"
    return this_str


def get_dataset_dict():
    import yaml
    all_nouns = pd.read_csv('annotations/EPIC_noun_classes.csv')
    nouns_list = [row['class_key'] for index, row in all_nouns.iterrows()]
    dict_file = {'path': "/home/sxa180157/ptg/epic_kitchens_100/object_detection_labels/",
                 'train': "EPIC-KITCHENS",
                 'val': "EPIC-KITCHENS",
                 'test': "",
                 "nc": len(nouns_list),
                 'names': json.dumps(nouns_list)}

    with open(r'./epic_kitchens.yaml', 'w') as file:
        documents = yaml.dump(dict_file, file, sort_keys=False)


def process_dataset(labels):
    # Label dict - key - location, value - bounding boxes and class value
    labels_dict = {}
    LabelData = namedtuple('LabelData', 'video_id object_class bounding_box')
    for index, label in labels.iterrows():
        # location = '/datasets/EPIC-KITCHENS/' + label['participant_id'] + '/object_detection_images/' + label[
        #     'video_id'] + '/' + str(label['frame']).rjust(10, '0') + '.jpg'
        location = get_frame_path(label['participant_id'], label['video_id'], str(label['frame']).rjust(10, '0'))
        bounding_boxes = ast.literal_eval(label['bounding_boxes'])
        object_class = label['noun_class']  # Use id, a same class may have different names
        video_id = label['video_id']
        partial_labels = []
        for bb in bounding_boxes:
            label = LabelData(video_id=video_id, object_class=object_class, bounding_box=bb)
            partial_labels.append(label)
        if len(bounding_boxes) == 0:
            continue
        if location not in labels_dict:
            # Add
            labels_dict[location] = partial_labels
        else:
            # Modify
            old_bounding_boxes = labels_dict[location]
            new_bounding_boxes = old_bounding_boxes + partial_labels
            labels_dict[location] = new_bounding_boxes
    return labels_dict


def write_file(labels):
    import pathlib
    print('-------- SAVING LABELS IN YOLO FORMAT --------')
    resolution_dict = get_frame_size("./annotations/EPIC_video_info.csv")
    labels_dict = process_dataset(labels)
    for key in labels_dict:
        with open(key, 'w+') as train_file:
            directory = key.rpartition('/')[0]
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            for label in labels_dict[key]:
                frame_width = resolution_dict[label.video_id].width
                frame_height = resolution_dict[label.video_id].height
                x_center = str((label.bounding_box[1] + (label.bounding_box[3] // 2)) / frame_width)
                y_center = str((label.bounding_box[0] + (label.bounding_box[2] // 2)) / frame_height)
                width = str(label.bounding_box[3] / frame_width)
                height = str((label.bounding_box[2]) / frame_height)
                train_file.write(f"{str(label.object_class)} {x_center} {y_center} {width} {height}")
                train_file.write('\n')

    # Each row is class x_center y_center width height format.


if __name__ == "__main__":
    # labels = pd.read_csv('annotations/EPIC_train_object_labels.csv')
    # labels = labels.head(10)
    # write_file(labels)
    get_dataset_dict()
