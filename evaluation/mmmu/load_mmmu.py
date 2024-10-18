import os
import re
import pyarrow.parquet as pq
import glob
from tqdm import tqdm
import sys

sys.path.append('../models/')

from utils import read_jsonl, read_json, write_jsonl, write_json
from eval_utils import option_refine


def retain_first_k_images(s: str, k: int) -> str:
    image_positions = [m.start() for m in re.finditer(r'<image>', s)]

    if k >= len(image_positions):
        return s
    k -= 1
    retained_part = s[:image_positions[k] + len('<image>')]
    remaining_part = re.sub(r'<image>', '', s[image_positions[k] + len('<image>'):])

    return retained_part + remaining_part


sall_data = []
all_files = glob.glob("MMMU/*/val*.parquet")
for i, f in enumerate(all_files):
    table = pq.read_table(f)
    df = table.to_pandas()
    list_of_dicts = df.to_dict('records')
    all_data.extend(list_of_dicts)
formated_all_data = []
for exp in tqdm(all_data):
    qid = exp['id']
    question, options, answers = exp['question'], eval(exp['options']), exp['answer']
    concated_options = option_refine(options)
    images_path = []
    find_image = f"{question} {concated_options}"
    for i in range(7):
        if exp[f"image_{i + 1}"] is not None and f"<image {i + 1}>" in find_image:
            # image = Image.open(io.BytesIO(exp[f"image_{i + 1}"]['bytes'])).convert('RGB')
            img_name = f"images/{qid}_{i}.png"
            binary_data = exp[f"image_{i + 1}"]['bytes']
            with open(img_name, 'wb') as file:
                file.write(binary_data)
            images_path.append(os.path.join('../mmmu', img_name))

    for i in range(7):
        question = question.replace(f"<image {i + 1}>", "<image>")
        concated_options = concated_options.replace(f"<image {i + 1}>", "<image>")
    if exp['question_type'] == 'multiple-choice':
        query = f"{question}\nChoices:\n{concated_options}"
    else:
        query = f"{question}"
    if query.count("<image>") > len(images_path):
        query = retain_first_k_images(query, len(images_path))
    formated_one_sample = {"images_path": images_path,
                           'question': query,
                           'options': options,
                           "answers": answers,
                           'ques_type': exp['question_type']}

    formated_all_data.append(formated_one_sample)

print(f"prepared {len(formated_all_data)} samples for MMMU dataset.")

write_jsonl('../eval_mmmu.jsonl', formated_all_data)
