import os
from tqdm import tqdm
import sys

sys.path.append('../models/')
from eval_utils import  symbols
from utils import read_jsonl, write_json, write_jsonl, parquet2list


pqdata = './testmini-00000-of-00001-725687bf7a18d64b.parquet'
all_data = parquet2list(pqdata)
formated_all_data = []
for exp in tqdm(all_data):
    query = exp['query']
    image_path = os.path.join('../mathvista/images', f"{exp['pid']}.jpg")

    question_type = exp['question_type']
    if question_type == 'free_form':
        question_type = 'open-ended'
        answer = exp['answer']
        options = ''
    else:
        options = [str(opt) for opt in exp['choices']]
        question_type = 'multiple-choice'
        options_symbols_dic = {opt: symbols[oid] for oid, opt in enumerate(exp['choices'])}
        answer = options_symbols_dic[exp['answer']]

    formated_one_sample = {"images_path": [image_path],
                           'question': query,
                           "answers": answer,
                           'ques_type': question_type,
                           'options': options
                           }
    formated_all_data.append(formated_one_sample)

print(f"prepared {len(formated_all_data)} samples for MathVista dataset.")

write_jsonl('../eval_mathvista.jsonl', formated_all_data)
