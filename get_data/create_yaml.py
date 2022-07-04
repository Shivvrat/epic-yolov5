import json

import pandas as pd

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


if __name__ == "__main__":
    # labels = pd.read_csv('annotations/EPIC_train_object_labels.csv')
    # labels = labels.head(10)
    # write_file(labels)
    get_dataset_dict()
