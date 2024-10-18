import json
import sys
from tqdm import tqdm

sys.path.append('../models/')
from utils import read_json, read_jsonl, write_json, write_jsonl, parquet2list, save_binary_image
from eval_utils import option_refine, symbols

all_data = []
subset_dirs = ['action_ground', 'action_prediction',
               'element_ground', 'element_ocr',
               'heading_ocr', 'web_caption', 'webqa']
all_files = [
    f"./{f}/test-00000-of-00001.parquet"
    for f in subset_dirs]
for subset in all_files:
    all_data += parquet2list(subset)
print(len(all_data), 'In total')

formated_all_data = []
for qid, exp in enumerate(tqdm(all_data)):
    image_type = 'text_centric'

    images_bytes = exp['image']['bytes']
    save_binary_image(images_bytes, f"images/visualwebbench_{exp['id']}.jpg")
    image_paths = [f'../visualwebbench/images/visualwebbench_{exp["id"]}.jpg']

    task_type = exp['task_type']
    ques_category = task_type

    if task_type == "action_ground":
        ques_type = 'multiple-choice'
        cur_instruction = exp['instruction']
        options = symbols[:8]
        question = (f"<image> In this website screenshot, I have labeled IDs for some HTML elements as candidates. "
                    f"Tell me which one I should click to complete the following task: {cur_instruction}\n"
                    f"You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.")

        answer = symbols[exp['answer']]
    elif task_type == 'action_prediction':
        ques_type = 'multiple-choice'
        options = exp['options'].tolist()
        concated_options = option_refine(options)
        question = (f"<image> You are given a screenshot of a webpage with a red rectangle bounding box. "
                    f"Please select the best webpage description that matches the new webpage after clicking the selected element in the bounding box.\n"
                    f"Choices:\n{concated_options}\n"
                    f"You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.")

        answer = symbols[exp['answer']]

    elif task_type == 'element_ground':
        ques_type = 'multiple-choice'

        question = (f"<image> In this website screenshot, I have labeled IDs for some HTML elements as candidates. "
                    f"Tell me which one best matches the description: {exp['elem_desc']}\n"
                    f"You should directly tell me your choice in a single uppercase letter, and do not output any explanation or any other contents.")
        options = symbols[:8]
        answer = symbols[exp['answer']]

    elif task_type == 'element_ocr':
        ques_type = 'captioning'
        question = (f"<image> You are given a screenshot of a webpage with a red rectangle bounding box.\n"
                    f"Please perform OCR in the bounding box and recognize the text content within the red bounding box."
                    # f"You should use the following format: The text content within the red bounding box is: <YOUR ANSWER>"
                    )

        options = None
        answer = exp['answer']

    elif task_type == 'heading_ocr':
        ques_type = 'captioning'
        question = (
            f"<image> You are given a screenshot of a webpage. Please generate the main text within the screenshot, which can be regarded as the heading of the webpage.\n\n"
            f"You should directly tell me the main content, and do not output any explanation or any other contents.")
        options = None
        answer = exp['answer']

    elif task_type == 'web_caption':
        ques_type = 'captioning'
        question = (
            f"<image> You are given a screenshot of a webpage. Please generate the meta web description information of this webpage, i.e., content attribute in <meta name='description' content=""> HTML element.\n\n"
            # "You should use the following format, and do not output any explanation or any other contents: <meta name='description' content='YOUR ANSWER'>"
        )
        options = None
        answer = exp['answer']

    elif task_type == 'webqa':
        ques_type = 'webqa'
        question = (
            f"<image> {exp['question']}\nYou should directly tell me your answer in the fewest words possible, and do not output any explanation or any other contents.")
        options = None
        answer = exp['answer'].tolist()

    if isinstance(answer, str):
        answer = [answer]

    if options is not None:
        concated_options = option_refine(options)
    else:
        concated_options = None

    formated_one_sample = {"images_path": image_paths,
                           'question': question,
                           'options': options,
                           'concated_options': concated_options,
                           "answers": answer,
                           'ques_type': ques_type,
                           'image_type': image_type,
                           'ques_category': ques_category}

    formated_all_data.append(formated_one_sample)

print(f"prepared {len(formated_all_data)} samples for VisualWebBench dataset.")

write_jsonl('../eval_visualwebbench.jsonl', formated_all_data)
