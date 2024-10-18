import sys
from tqdm import tqdm
import os
sys.path.append('../models/')

from utils import read_json, read_jsonl, write_json, write_jsonl, parquet2list, remove_duplicates

val_data = []
vanilla_data = read_json('val.json')
for one_sample in vanilla_data['data']:
    img_list = [
        os.path.join('./images', f"{i}.jpg") for i
        in one_sample['page_ids']]
    img_tokens = "<image>" * len(img_list)
    question = one_sample['question']
    answers = one_sample['answers']
    formed_one_dic = {'images': img_list, 'question': question, 'answers': answers, }
    val_data.append(formed_one_dic)
write_jsonl('./formed_val.jsonl', val_data)

formated_all_data = []
all_data = read_json('formed_val.json')

for exp in tqdm(all_data):
    question = exp['question']
    answers = exp['answers']
    image_paths = exp['images']

    img_token = "<image>" * len(image_paths)
    question = f"{img_token} {question}"

    formated_one_sample = {"images_path": image_paths,
                           'question': question,
                           'options': None,
                           'concated_options': None,
                           "answers": answers,
                           'ques_type': 'open-ended'}

print(f"prepared {len(formated_all_data)} samples for MP-DocVQA dataset.")

write_jsonl('../eval_mpdocvqa.jsonl', formated_all_data)
