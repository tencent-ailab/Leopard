import json
import random
import re
import os
import sys
import os
import random
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.append('../models/')

from utils import write_jsonl, read_json
from tqdm import tqdm

data_path = './val_v1.0_withQT.json'
vanilla_data = read_json(data_path)

train_data = []

formated_all_data = []
for one_sample in tqdm(vanilla_data['data']):
    img_name = one_sample['image'].replace('documents/', '')
    # ori_path = f"/apdcephfs_us/share_300814644/user/mengzhaojia/multi_img_data/sp_docvqa/images/{img_name}"
    # cmd = f"cp {ori_path} ./images/ "
    # os.system(cmd)
    new_img_path = [os.path.join('../docvqa/images/', img_name)]

    question = one_sample['question']

    formated_one_sample = {"images_path": new_img_path,
                           'question': f"<image> {question}",
                           'options': None,
                           "answers": one_sample['answers'],
                           'ques_type': 'open-ended'}

    formated_all_data.append(formated_one_sample)

print(f"prepared {len(formated_all_data)} samples for DocVQA dataset.")

write_jsonl('../eval_docvqa.jsonl', formated_all_data)
