import ast
from collections import namedtuple

import pandas as pd


def get_frame_path(participant_id, video_id, frame_num):
    path = f"/home/sxa180157/ptg/epic_kitchens_100/object_detection_labels/EPIC-KITCHENS/{participant_id}/object_detection_images/{video_id}/{frame_num}.txt"
    return path


def process_dataset(labels):
    all_nouns = pd.read_csv('downloaded-labels/EPIC_noun_classes.csv')
    nouns_list = [row['class_key'] for index, row in all_nouns.iterrows()]
    ClassData = namedtuple('ClassData', 'new_id name')
    class_dict = {}

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


print('-------- SAVING LABELS IN YOLO FORMAT --------')
with open('processed-labels/train.txt', 'w') as train_file:
    labels_dict = process_dataset(labels)
    for key in labels_dict:
        train_file.write(key + ' ')
        for label in labels_dict[key]:
            train_file.write(str(label.bounding_box[1]) + ',' + str(label.bounding_box[0]) + ',' + str(
                label.bounding_box[1] + label.bounding_box[3]) + ',' + str(
                label.bounding_box[0] + label.bounding_box[2]) + ',' + str(label.object_class) + ' ')
        train_file.write('\n')
