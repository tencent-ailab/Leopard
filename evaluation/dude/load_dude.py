import sys
import os
from tqdm import tqdm
from datasets import load_dataset

sys.path.append('../models/')

from utils import read_jsonl, write_json, write_jsonl

split = 'val'
ds = load_dataset("jordyvl/DUDE_loader", 'Amazon_original', split=split,
                  data_dir="./DUDE/")  # with custom extracted data directory

all_imgs_dir = f"./DUDE/images/{split}/"

all_image_lists = os.listdir(all_imgs_dir)

dict_list = []
for ids, x in enumerate(tqdm(ds)):
    dic_to_save = {}
    dic_to_save["id"] = "dude_" + x['questionId']
    dic_to_save['question'] = x['question']
    dic_to_save['answers'] = x['answers']

    matching_files = [file for file in all_image_lists if file.startswith(x['docId'])]

    matching_files = sorted(matching_files, key=lambda x: int(x.split('_')[-1].replace('.jpg', '')))
    abs_imgs = [os.path.join(all_imgs_dir, img) for img in matching_files]
    # if len(matching_files) > 20:
    #     continue
    dic_to_save["images"] = abs_imgs

    dic_to_save['metadata'] = x
    dict_list.append(dic_to_save)

write_jsonl(f'dude_{split}.jsonl', dict_list)

formated_all_data = []
all_data = read_jsonl('./dude_val.jsonl')

for exp in tqdm(all_data):
    question = exp['question']
    answers = exp['answers']
    images = [img.replace(
        './DUDE/images/',
        '../dude/images/') for img in exp['images']]

    formated_one_sample = {'id': exp['id'],
                           "images_path": images,
                           'question': question,
                           'options': None,
                           'concated_options': None,
                           "answers": answers,
                           'ques_type': 'open-ended'}

    formated_all_data.append(formated_one_sample)

print(f"prepared {len(formated_all_data)} samples for DUDE dataset.")

write_jsonl('../eval_dude.jsonl', formated_all_data)
