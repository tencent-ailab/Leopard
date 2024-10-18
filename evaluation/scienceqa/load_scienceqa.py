import os

from tqdm import tqdm

import sys

sys.path.append('../models/')

from utils import read_jsonl, read_json, write_jsonl, write_json, parquet2list, save_binary_image, symbols
from eval_utils import option_refine

vanilla_data = parquet2list('test-00000-of-00001-f0e719df791966ff.parquet')
formated_all_data = []
gid = 0
for sample in tqdm(vanilla_data):
    if not sample['image']:
        continue
    image_binary = sample['image']['bytes']
    image_name = f"images/scienceqa_{gid}.png"
    gid += 1
    images_path = [os.path.join('../scienceqa', image_name)]
    save_binary_image(image_binary, image_name)

    question = sample['question']
    options = sample['choices']
    answer = symbols[sample['answer']]
    hint = sample['hint']

    concat_choices = option_refine(options)
    query = (f"<image> {question}\n"
             f"Hint: {hint}\nChoices:\n{concat_choices}")

    formated_one_sample = {"images_path": images_path,
                           'question': query,
                           'options': [str(o) for o in options],
                           "answers": answer,
                           'ques_type': 'multiple-choice'
                           }

    formated_all_data.append(formated_one_sample)
print(f"prepared {len(formated_all_data)} samples for ScienceQA dataset.")

write_jsonl('../eval_scienceqa.jsonl', formated_all_data)
