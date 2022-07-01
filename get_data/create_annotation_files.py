import ast
from collections import namedtuple

import pandas as pd


def get_frame_path(participant_id, video_id, frame_num):
    path = f"/home/sxa180157/ptg/epic_kitchens_100/object_detection_labels/EPIC-KITCHENS/{participant_id}/object_detection_images/{video_id}/{frame_num}.txt"
    return path


def get_frame_size(video_info_file):
    video_info = pd.read_csv(video_info_file)
    ResolutionData = namedtuple('ResolutionData', 'width height')
    resolution_dict = {row["video"]: row['resolution'].split("x") for index, row in video_info.iterrows()}
    return resolution_dict


def process_dataset(labels):
    all_nouns = pd.read_csv('downloaded-labels/EPIC_noun_classes.csv')
    nouns_list = [row['class_key'] for index, row in all_nouns.iterrows()]
    ClassData = namedtuple('ClassData', 'new_id name')
    class_dict = {}
    # Label dict - key - location, value - bounding boxes and class value
    labels_dict = {}
    LabelData = namedtuple('LabelData', 'object_class bounding_box')
    for index, label in labels.iterrows():
        # location = '/datasets/EPIC-KITCHENS/' + label['participant_id'] + '/object_detection_images/' + label[
        #     'video_id'] + '/' + str(label['frame']).rjust(10, '0') + '.jpg'
        location = get_frame_path(label['participant_id'], label['video_id'], str(label['frame']).rjust(10, '0'))
        bounding_boxes = ast.literal_eval(label['bounding_boxes'])
        object_class = label['noun_class']  # Use id, a same class may have different names

        # Save classes in dictionary
        if object_class not in class_dict:
            class_data = ClassData(new_id=len(class_dict), name=nouns_list[object_class])  # Save the first noun name
            class_dict[object_class] = class_data
            object_class = class_data.new_id  # Save new_id to print into file
        else:
            object_class = class_dict[object_class].new_id

        # Save labels in dictionary

        partial_labels = []
        for bb in bounding_boxes:
            label = LabelData(object_class=object_class, bounding_box=bb)
            partial_labels.append(label)
        if len(bounding_boxes) == 0:
            continue
        if location not in labels_dict:
            # Add
            labels_dict[location] = partial_labels
        else:
            # Modify
            old_boundingboxes = labels_dict[location]
            new_boundingboxes = old_boundingboxes + partial_labels
            labels_dict[location] = new_boundingboxes
    return labels_dict


def write_file(labels):
    import pathlib
    print('-------- SAVING LABELS IN YOLO FORMAT --------')
    labels_dict = process_dataset(labels)
    for key in labels_dict:
        with open(key, 'w') as train_file:
            directory = key.rpartition('/')[0]
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            for label in labels_dict[key]:
                x_center = str(label.bounding_box[1] + (label.bounding_box[3] // 2))
                y_center = str(label.bounding_box[0] + (label.bounding_box[2] // 2))
                width = str(label.bounding_box[3])
                height = str(label.bounding_box[2])
                train_file.write(f"{str(label.object_class)} {x_center} {y_center} {width} {height}")
                train_file.write('\n')

    # Each row is class x_center y_center width height format.


# labels = pd.read_csv('annotations/EPIC_train_object_labels.csv')
# write_file(labels)
print(get_frame_size("./annotations/EPIC_video_info.csv"))
