import os
from tqdm import tqdm
import json
import sys
sys.path.append('../models/')

from utils import read_jsonl, write_json, write_jsonl,read_json



formated_all_data = []
all_data_path = os.listdir('data/')
for data_name in tqdm(all_data_path):

    data_path = f"data/{data_name}"

    json_path = os.path.join(data_path, 'chart-path_and_question-answer_pair.json')
    qa_data = read_json(json_path)
    all_samples = []
    img_path_list = []
    for i, one_sample in enumerate(qa_data):
        question = one_sample['question']
        qid = one_sample['id']
        img_paths = one_sample['image']
        for img_path in img_paths:
            img_path = img_path.replace('\\', '/')
            last_name = img_path.split('.')[-1]
            img_path_list.append(f"../multichart/{img_path}")
        if one_sample['type'] == 'multiple-choice':
            options = ["A", "B", "C", "D"]
        else:
            options = ''
        formated_one_sample = {"images_path": img_path_list,
                               'question': question,
                               'options': options,
                               "answers": [one_sample['answer']],
                               'ques_type': one_sample['type'] }
        formated_all_data.append(formated_one_sample)

print(f"prepared {len(formated_all_data)} samples for MultiChartQA dataset.")
write_jsonl('../eval_multichart.jsonl', formated_all_data)

