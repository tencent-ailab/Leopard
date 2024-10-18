import requests
import torch
from PIL import Image
from io import BytesIO
import json
from transformers import AutoProcessor, AutoModelForVision2Seq
# from transformers.models.ideficts2 import IDEFiCTS2Processor, IDEFiCTS2Model
from transformers.image_utils import load_image
from tqdm import tqdm
import argparse
import os
import sys
import sys

sys.path.append('../models/')
from utils import read_jsonl, read_json, write_jsonl, write_json
from eval_utils import option_refine, symbols, format_acc, split_shard

from mmmu_utils import parse_multi_choice_response, parse_open_response, eval_multi_choice, eval_open


def load_model(args):
    processor = AutoProcessor.from_pretrained(args.checkpoint,
                                              size={'longest_edge': args.resolution, 'shortest_edge': 0},
                                              do_image_splitting=False)

    model = AutoModelForVision2Seq.from_pretrained(args.checkpoint,
                                                   torch_dtype=torch.float16).to(torch.device('cuda'))
    model.eval()
    return processor, model


def main(args):
    processor, model = load_model(args)
    eval_bench_name = args.dataset

    eval_jsonl_path = f"../eval_{eval_bench_name}.jsonl"

    formated_all_data = read_jsonl(eval_jsonl_path)
    formated_all_data = formated_all_data[:900]
    formated_all_data = split_shard(formated_all_data, args)

    # formated_all_data = formated_all_data[:200]
    matches, results = [], []
    total, correct_anls = 0, 0
    image_type_acc_dic = {}

    for exp in tqdm(formated_all_data):
        images = []
        img_messages = []
        if exp['images_path']:
            for ipath in exp['images_path']:
                image = Image.open(ipath).convert('RGB')
                images.append(image)
                img_messages.append({"type": "image"})
        elif exp['image_bytes']:
            for iimage in exp['image_bytes']:
                images.append(iimage)
                img_messages.append({"type": "image"})
        else:
            print('No image found')
        if len(images) > 8:
            continue
        question = exp['question']
        answers = exp['answers']

        if exp['options']:
            if args.setting == 'cot':
                instruction = 'First think step by step. Then answer with the letter.'
            else:
                instruction = 'Answer with the letter.'
            options = exp['options']
            concat_options = option_refine(options)
            prompt = f"{question}\nOptions: {concat_options} {instruction}"
        else:
            if args.setting == 'cot':
                instruction = 'First think step by step. Then answer with a single word or phrase.'
            else:
                instruction = 'Answer with a single word or phrase.'
            prompt = f"{question} {instruction}"

        DEFAULT_IMAGE_TOKEN = "<image>"
        image_token_count = prompt.count(DEFAULT_IMAGE_TOKEN)
        image_num = len(images)
        if image_token_count < image_num:
            prompt = DEFAULT_IMAGE_TOKEN * (image_num - image_token_count) + prompt

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        with torch.no_grad():
            processed_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=processed_prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(torch.device('cuda')) for k, v in inputs.items()}
            # Generate
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            content = generated_texts[0].split('Assistant: ')[-1]

            if exp['options']:
                chosen = parse_multi_choice_response(content, symbols[:len(options)],
                                                     {s: o for s, o in zip(symbols[:len(options)], options)})
                correct = eval_multi_choice(answers, chosen)

            else:
                chosen = parse_open_response(content)
                correct = eval_open(answers, chosen)

            matches.append(correct)
            results.append(
                {'correct': correct, 'chosen': chosen, 'gold': exp['answers'],
                 'raw': generated_texts, 'question': question,
                 'image_type': exp.get('image_type', None),
                 'multi_img': len(images) > 1,
                 'correct_anls': correct_anls})
            print(f"prompt: {prompt} | generated: {generated_texts} | correct: {correct} | "
                  f"{len(matches)} acc: {format_acc(sum(matches), len(matches))}")

            # print(results[-1])

    acc_dic = {'acc': format_acc(sum(matches), len(matches)),
               'correct': sum(matches),
               'total': len(matches)}

    for k, v in acc_dic.items():
        print(k, v)

    evaL_setting_name = args.setting
    json_details = os.path.join(args.checkpoint,
                                f"{args.shard}_res{args.resolution}_{evaL_setting_name}_{eval_bench_name}_shard_details.jsonl")

    print(f'saving to {json_details}')
    write_jsonl(json_details, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard",
                        type=int,
                        default=0,
                        help="shard id")
    parser.add_argument("--num_shards",
                        type=int,
                        default=8,
                        help="number of shards")
    parser.add_argument("-d",
                        "--dataset",
                        type=str,
                        help="eval dataset")
    parser.add_argument("-c",
                        "--checkpoint",
                        type=str,
                        help="eval checkpoint")
    parser.add_argument("-s",
                        "--setting",
                        type=str,
                        default='direct')
    parser.add_argument("--resolution",
                        type=int,
                        default=980)
    args = parser.parse_args()
    if 'apdcephfs_us' in args.checkpoint:
        args.location = 'us'
    else:
        args.location = 'qy'

    main(args)
