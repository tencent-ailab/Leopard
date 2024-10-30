import os
import sys
import re
from tqdm import tqdm
import random

sys.path.append('../models/')

from utils import arg_parser, read_json, read_jsonl, sort_key, write_jsonl, instruction_pool_concise_ans

img_path = 'SlideVQA/images/test/'

all_samples = read_jsonl(
    'SlideVQA/annotations/qa/test.jsonl')

questions_withing_one_deck = {}
formated_all_data = []
for sample in tqdm(all_samples):

    pid = sample['deck_name']
    img_dir = os.path.join('../slidevqa/images', pid)
    img_paths = [os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith('.jpg')]
    if len(img_paths) != 20:
        continue
    img_paths = sorted(img_paths, key=lambda x: int(re.search(r'-(\d+)-\d+\.jpg', x).group(1)))

    formated_one_sample = {"images_path": img_paths,
                           'question': sample['question'],
                           "answers": sample['answer'],
                           'ques_type': 'open-ended',
                           'options': ''
                           }
    formated_all_data.append(formated_one_sample)

print(f"prepared {len(formated_all_data)} samples for SlideVQA dataset.")
write_jsonl('../eval_slidevqa.jsonl', formated_all_data)
