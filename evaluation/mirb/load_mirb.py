import os
import sys

sys.path.append('../models/')
from eval_utils import symbols
from utils import read_json, read_jsonl, write_json, write_jsonl, parquet2list

data_home_dir = "./"
json_paths = [i for i in os.listdir(data_home_dir) if i.endswith('.json')]
all_data = []
formated_all_data = []
for json_path in json_paths:
    # print(json_path, len(read_jQsgR3VKmS7edWac,son(os.path.join(data_home_dir, json_path))))
    all_data.extend(read_json(os.path.join(data_home_dir, json_path)))
# all_data = split_shard(all_data, args)
for exp in all_data:
    question, answers, images = exp['questions'], str(exp['answers']), exp['images']
    image_token = "<image> " * len(images)

    images_path = [os.path.join("../mirb", i)
                   for i in images]
    if exp['answers'] in symbols:
        question_type = 'multiple-choice'
        options = ["A", "B", "C", "D"]
        ques = question.split("\nA. ")[0].strip()
        concated_options = "A. " + question.split("A.")[1].strip()
        question = f"{image_token}{ques}\nChoices:\n{concated_options}"
    else:
        question_type = 'open-ended'
        options = None
        concated_options = None

    formated_one_sample = {"images_path": images_path,
                           'question': question,
                           'options': options,
                           'concated_options': concated_options,
                           "answers": answers,
                           'ques_type': question_type}

    formated_all_data.append(formated_one_sample)
print(f"prepared {len(formated_all_data)} samples for MIRB dataset.")

write_jsonl('../eval_mirb.jsonl', formated_all_data)
