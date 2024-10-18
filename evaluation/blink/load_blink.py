from datasets import load_dataset
import sys
from tqdm import tqdm
sys.path.append('../models/')
from utils import write_jsonl

subtasks = ['Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning',
            'Relative_Reflectance', 'Visual_Correspondence', 'Counting', 'IQ_Test',
            'Object_Localization', 'Semantic_Correspondence', 'Visual_Similarity',
            'Forensic_Detection', 'Jigsaw', 'Relative_Depth', 'Spatial_Relation']
dataset_name = 'BLINK-Benchmark/BLINK'

formated_all_data = []
for subtask in subtasks:
    data = load_dataset(dataset_name, subtask)
    # print(data['val'])
    for sample in tqdm(data['val']):
        idx = sample['idx']

        img_path_1 = f"./images/{idx}_img_1.jpeg"
        img_path_2 = f"./images/{idx}_img_2.jpeg"
        img_path_3 = f"./images/{idx}_img_3.jpeg"
        img_path_4 = f"./images/{idx}_img_4.jpeg"
        img_path_list = []
        if sample['image_1']:
            sample['image_1'].save(img_path_1)
            img_path_list.append(f"../blink/images/{idx}_img_1.jpeg")
        if sample['image_2']:
            sample['image_2'].save(img_path_2)
            img_path_list.append(f"../blink/images/{idx}_img_2.jpeg")
        if sample['image_3']:
            sample['image_3'].save(img_path_3)
            img_path_list.append(f"../blink/images/{idx}_img_3.jpeg")
        if sample['image_4']:
            sample['image_4'].save(img_path_4)
            img_path_list.append(f"../blink/images/{idx}_img_4.jpeg")
        img_tokens = "<image>" * len(img_path_list)
        question = f"{img_tokens} {sample['prompt']}"

        formated_one_sample = {"images_path": img_path_list,
                               'question': question,
                               'options': sample['choices'],
                               "answers": sample['answer'],
                               'ques_type': 'multiple-choice'}
        formated_all_data.append(formated_one_sample)
print(f"prepared {len(formated_all_data)} samples for Blink dataset.")

write_jsonl('../eval_blink.jsonl', formated_all_data)
