import argparse
import os
import torch
from typing import List, Optional, Tuple, Union
import math
from transformers import CLIPImageProcessor, LlavaForConditionalGeneration, SiglipImageProcessor, \
    AutoTokenizer, LlavaConfig
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
from transformers.activations import ACT2FN
from PIL import Image
from tqdm import tqdm
import sys
from rouge import Rouge
import torch.nn as nn

sys.path.append('../models/')

from utils import read_jsonl, read_json, write_jsonl, write_json
from eval_utils import expand2square, calculate_anls, option_refine, symbols, format_acc, split_shard, get_instruction, \
    eval_rouge, eval_multi_choice, eval_open, parse_multi_choice_response, parse_open_response, cut_img, retain_n_images

head = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
tail = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def allocate_patches(image_sizes, patch_size=364, patch_budget=50):
    # Calculate total available patches for each image
    patch_counts = []
    total_patches = 0

    for size in image_sizes:
        # Calculate how many patches the image can have without resizing
        height, width = size
        num_patches = round(height / patch_size) * round(width / patch_size)
        if num_patches == 1:
            num_patches = 0
        patch_counts.append(num_patches)
        total_patches += num_patches

    # If the total number of patches is within the budget
    if total_patches <= patch_budget:
        return patch_counts

    # Otherwise, scale down the patch count to fit within the budget
    scale_factor = patch_budget / total_patches
    scaled_patch_counts = [int(num_patches * scale_factor) for num_patches in patch_counts]

    # Make sure the scaled counts don't exceed the budget
    while sum(scaled_patch_counts) > patch_budget:
        excess = sum(scaled_patch_counts) - patch_budget
        for i in range(len(scaled_patch_counts)):
            if scaled_patch_counts[i] > 0:
                scaled_patch_counts[i] -= 1
                excess -= 1
            if excess == 0:
                break

    return scaled_patch_counts


def select_best_resolution(original_size, num_patches, patch_size=364):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        num_patches (int): The number of patches the image will be split into.

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    if num_patches == 0:
        return None
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")
    for row in range(1, num_patches + 1):
        for col in range(1, num_patches + 1):
            if row * col > num_patches or (row == 1 and col == 1):
                continue
            height = row * patch_size
            width = col * patch_size
            # Calculate the downscaled size to keep the aspect ratio
            scale = min(width / original_width, height / original_height)
            downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

            # Calculate effective and wasted resolutions
            effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
            wasted_resolution = (width * height) - effective_resolution

            if effective_resolution > max_effective_resolution or (
                    effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
                max_effective_resolution = effective_resolution
                min_wasted_resolution = wasted_resolution
                best_fit = (width, height)
    if best_fit == (patch_size, patch_size):
        return None
    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    if target_resolution == None:
        return None
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def pixel_shuffle(x, scale_factor=2):
    bsz, seq, embed_dim = x.size()
    height = width = int(seq ** 0.5)

    x = x.reshape(bsz, height, width, embed_dim)
    x = x.reshape(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor ** 2))
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(bsz, int(seq / (scale_factor ** 2)), embed_dim * (scale_factor ** 2))

    return x


class myLlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_config.hidden_size * 4, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        image_features = pixel_shuffle(image_features)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class myLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.multi_modal_projector = myLlavaMultiModalProjector(config)
        # self.vision_tower = Idefics2ViTModel(config)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            vision_feature_layer: Optional[int] = None,
            vision_feature_select_strategy: Optional[str] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                # image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                image_outputs = self.vision_tower(pixel_values)
                # print(self.vision_tower)
                # print(len( image_outputs.hidden_states))
                # print(vision_feature_layer)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.last_hidden_state

                # if vision_feature_select_strategy == "default":
                #     selected_image_feature = selected_image_feature[:, 1:]
                # elif vision_feature_select_strategy == "full":
                #     selected_image_feature = selected_image_feature
                # else:
                #     raise ValueError(
                #         f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                #     )
                image_features = self.multi_modal_projector(selected_image_feature)
                inputs_embeds = inputs_embeds.to(image_features.dtype)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def run_llava_local_inference(args):
    eval_bench_name = args.dataset
    eval_jsonl_path = f"../eval_{eval_bench_name}.jsonl"

    formated_all_data = read_jsonl(eval_jsonl_path)
    formated_all_data = split_shard(formated_all_data, args)
    if args.view:
        formated_all_data = formated_all_data[:200]

    llava = myLlavaForConditionalGeneration.from_pretrained(args.checkpoint, torch_dtype=torch.float32)
    llava.eval()
    llava.to(f'cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    image_processor = SiglipImageProcessor.from_pretrained('siglip-so400m-14-364-flash-attn2-navit')

    matches, results = [], []
    with torch.no_grad():
        results = []
        for exp in tqdm(formated_all_data):
            image_paths = exp['images_path']
            images = [Image.open(ipath).convert('RGB') for ipath in image_paths]
            image_num = len(images)
            budget = 50 - len(images)
            if budget > 0:
                num_patchs_per_images = allocate_patches([image.size for image in images], patch_budget=budget)
                best_resolutions = [select_best_resolution(image.size, num_patches) for image, num_patches in
                                    zip(images, num_patchs_per_images)]
                images_padded = [resize_and_pad_image(image, resolution) for image, resolution in
                                 zip(images, best_resolutions)]
                image_patches = [divide_to_patches(image, 364) if image is not None else [] for image in images_padded]
                num_patchs_per_images_real = [len(patches) for patches in image_patches]
                all_patches = []
                for origin, patches in zip(images, image_patches):
                    all_patches += [origin] + patches
                images = all_patches
            else:
                num_patchs_per_images_real = [1] * len(images)

            images = torch.cat(
                [image_processor.preprocess(image, return_tensors='pt')['pixel_values'] for image in images], dim=0)
            images = images.to(llava.device)
            # images = images.half()

            question = exp['question']
            answers = exp['answers']
            ques_type = exp['ques_type']
            instruction = get_instruction(args.setting, ques_type)
            DEFAULT_IMAGE_TOKEN = "<image>"
            image_token_count = question.count(DEFAULT_IMAGE_TOKEN)
            if image_token_count < image_num:
                add_img_token = (DEFAULT_IMAGE_TOKEN * (image_num - image_token_count))
                question = f"{add_img_token} {question}"

            elif image_token_count > image_num:
                question = retain_n_images(question, image_token_count - image_num)

            # if image_token_count < image_num:
            #     question = f"{DEFAULT_IMAGE_TOKEN * (image_num - image_token_count)}{question}"
            # prompt = f"{system_prompt}{head}{question}\n{instruction}{tail}"
            prompt = f"{head}{question}\n{instruction}{tail}"

            image_count = 1
            image_special_tokens = []
            for _ in range(image_token_count):
                this_per_image_token = 1
                if num_patchs_per_images_real:
                    this_per_image_token += num_patchs_per_images_real[image_count - 1]
                # print ('this_per_image_token', this_per_image_token)
                image_special_tokens.append(
                    f"image {image_count}: <|reserved_special_token_20|>{'<|reserved_special_token_195|>' * this_per_image_token}<|reserved_special_token_21|>")
                image_count += 1
            # image_special_tokens = [f"(image {i}: <Image><|reserved_special_token_195|></Image>)" for i in
            #                         range(1, len(images) + 1)]

            for i, image_special_token in enumerate(image_special_tokens):
                prompt = prompt.replace("<image>", image_special_token, 1)

            prompt = prompt.replace('\r\n\t\t\r\n\t\t', ' ')
            input_ids = tokenizer([prompt], return_tensors='pt', truncation=True, max_length=16384)['input_ids']
            attn_mask = input_ids != tokenizer.pad_token_id
            input_ids = input_ids.to(llava.device)
            attn_mask = attn_mask.to(llava.device)

            output_ids = llava.generate(input_ids, pixel_values=images, attention_mask=attn_mask,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=[128001, 128009],
                                        # eos_token_id=tokenizer.encode('<|eot_id|>'),
                                        max_new_tokens=128, use_cache=True)
            input_token_len = input_ids.shape[1]
            response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

            correct_anls = 0
            if ques_type == 'multiple-choice':
                chosen = parse_multi_choice_response(response, symbols[:len(exp['options'])],
                                                     {s: o for s, o in
                                                      zip(symbols[:len(exp['options'])], exp['options'])})
                correct = eval_multi_choice(answers, chosen)
            elif ques_type == 'open-ended':
                chosen = parse_open_response(response)
                correct = eval_open(answers, chosen)
                if args.dataset in ['mpdocvqa', 'dude', 'docvqa']:
                    formated_response = response.split('Answer: ')[-1].lower()
                    answers = [answer.lower() for answer in answers]

                    correct_anls = calculate_anls(gt=answers, pred=formated_response)
            elif ques_type == 'captioning':
                chosen = response
                correct = eval_rouge([answers[0].lower()], [response.lower()])
            elif ques_type == 'webqa':
                chosen = response
                rouge = Rouge(metrics=['rouge-1'])
                correct = max(
                    [rouge.get_scores([chosen.lower()], [gold.lower()], avg=True)['rouge-1']['f'] for gold in answers])

            matches.append(correct)
            results.append({'correct': correct, 'chosen': chosen, 'gold': exp['answers'],
                            'raw': response, 'question': question,
                            'image_type': exp.get('image_type', None),
                            'multi_img': len(images) > 1,
                            'correct_anls': correct_anls})
            if args.view:
                print(f"response: {response} | correct: {correct} | gt: {exp['answers']} |"
                      f"{len(matches)} acc: {format_acc(sum(matches), len(matches))}")
    acc_dic = {'acc': format_acc(sum(matches), len(matches)),
               'correct': sum(matches),
               'total': len(matches)}

    for k, v in acc_dic.items():
        print(k, v)

    evaL_setting_name = args.setting
    json_details = os.path.join(args.checkpoint,
                                f"{args.shard}_{evaL_setting_name}_{eval_bench_name}_shard_details.jsonl")

    print(f'saving to {json_details}')
    write_jsonl(json_details, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("-c", "--checkpoint")
    parser.add_argument("-d", "--dataset", default='mmmu')
    parser.add_argument("-s", "--setting", default='direct')
    parser.add_argument('-v', '--view', action='store_true')
    args = parser.parse_args()

    run_llava_local_inference(args)
