
from tqdm import tqdm
import sys
sys.path.append('../models/')
from utils import read_jsonl, read_json, write_jsonl, write_json

vanilla_data = read_json('TextVQA_0.5.1_val.json')['data']
formated_all_data = []
gid = 0
for sample in tqdm(vanilla_data):
    img_path = f"../textvqa/images/train_images/{sample['image_id']}.jpg"
    question = sample['question']
    answers = sample['answers']

    query = f"<image> {question}"
    formated_one_sample = {"images_path": [img_path],
                           'question': question,
                           'options': None,
                           "answers": answers,
                           'ques_type': 'open-ended'
                           }

    formated_all_data.append(formated_one_sample)
print(f"prepared {len(formated_all_data)} samples for TextVQA dataset.")

write_jsonl('../eval_textvqa.jsonl', formated_all_data)
