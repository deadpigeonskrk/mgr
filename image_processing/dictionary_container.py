import json, random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 

with open('PhotoBook/v2/gold-extracted.json', 'r') as file:
    data = json.load(file)
    dict_dial_img_description = {}
    lst_paths_ph_bk = []
    # print(data["person_truck/COCO_train2014_000000258505.jpg"])
    for key in data.keys():
        list_utterances = []
        for subkey in data[key]:
            for dictionary_object in data[key][subkey]:
                list_utterances.append(dictionary_object["Message_Text"])
        dict_dial_img_description[key] = list_utterances
        lst_paths_ph_bk.append(key)

def show_img_dial_descriptions(title, main_dict):
    list_of_keys = list(main_dict.keys())
    if isinstance(title, int):
        title = list_of_keys[title]
    for i, descr in enumerate(main_dict[title]):
        print(i, "-", descr)

    image = mpimg.imread("PhotoBook/images/" + title)
    plt.imshow(image)


# show_img_dial_descriptions(50, dict_dial_img_description)