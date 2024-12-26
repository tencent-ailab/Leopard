# Copyright (c) 2023 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import io
import json
from typing import Dict, Optional, Sequence, List
import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import transformers
from megatron.training import get_args
from itertools import chain

from megatron_patch.tokenizer import get_special_token_id
from megatron_patch.data.llava import conversation as conversation_lib
from megatron_patch.data.idefics2.constants import DEFAULT_IMAGE_TOKEN, END_OF_UTTERANCE_TOKEN

# from megatron_patch.model.llava.clip_encoder import CLIPVisionTower
from transformers import Idefics2ImageProcessor
# from transformers.models.idefics2.modeling_idefics2 import Idefics2ImageProcessor

import wids
from functools import partial
from packaging import version
import tokenizers
from transformers import AutoProcessor
from .idefics2_image_processor import Idefics2ImageProcessorPad
from collections import Counter
import pickle

def rename_dict(d):
    new_d = {}
    role_dict = {'human':'user', 'gpt':'assistant'}
    new_d['role'] = role_dict[d['from']]
    if 'value' in d:
        new_d['content'] = d['value']
    else:
        new_d['content'] = d['content']
    # new_d['content'] = d['value']
    return new_d

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

IDEFICS2_CHAT_TEMPLATE = (
            "{% for message in messages %}"
                "{{message['role'].capitalize()}}"
                "{% if message['content'][0]['type'] == 'image' %}"
                    "{{':'}}"
                "{% else %}"
                    "{{': '}}"
                "{% endif %}"
                "{% for line in message['content'] %}"
                    "{% if line['type'] == 'text' %}"
                        "{{line['text']}}"
                    "{% elif line['type'] == 'image' %}"
                        "{{ '<image>' }}"
                    "{% endif %}"
                "{% endfor %}"
                "<end_of_utterance>\n"
            "{% endfor %}"

            "{% if add_generation_prompt %}"
                "{{ 'Assistant:' }}"
            "{% endif %}"
        )

def bytes_to_img(byte_data):
    #img_stream = io.BytesIO(byte_data)
    img = Image.open(byte_data)
    return img

def btyes_to_dict(byte_data):
    return eval(byte_data.read().decode('utf-8'))

def make_sample(sources, processor, data_args):
    # print (sources)
    original_sources = sources
    # has_image=True
    # if sources['has_image']: 
    if '.input_image' in sources:
        try:
            images = pickle.loads(sources['.input_image'].getvalue())
            images = [Image.open(image).convert('RGB') for image in images]
        except:
            try:
                # single images
                images = [sources['.input_image']]
                images = [Image.open(image).convert('RGB') for image in images]
            except:
                # when pickle fails,
                # and other unexpected scenarios
                images = []
                images = [None]
                print("??? bad img 1")
    else:
        # has_image=False
        images = [None]
        print("??? bad img 2")

    # sources = copy.deepcopy([e[".conversations"] for e in sources])
    sources = copy.deepcopy([btyes_to_dict(sources[".conversations"])])
    try:
        # FIXME:
        # the format of the previous version
        if 'from' in sources[0][0]:
            sources = [[rename_dict(d) for d in sources[0]]]
    except:
        sources=[[{'user':' ','content':' '},{'assistant':' ','content':' '}]]
        print('bad source')
    # if data_args.max_padding_length <= len(data_dict["input_ids"][0]):
    #     print ('The example is too long, skipping')
    #     return None

    # pretraining
    # assert len(sources[0]) % 2 == 0
    # assert all([sources[0][2*i]['role'] == 'user' for i in range(len(sources[0])//2)])
    # assert all([sources[0][2*i + 1]['role'] == 'assistant' for i in range(len(sources[0])//2)])

    # example = {"image": image, 'question':question, 'answer':answer}

    # number of images must be equal to the number of <image> tokens
    # print(len(images), question.count("<image>"))

    all_queries = [sources[0][2*i]['content'] for i in range(len(sources[0])//2)]
    all_answers = [sources[0][2*i + 1]['content'] for i in range(len(sources[0])//2)]

    if all([img is None for img in images]):
        # if all images are none, force remove all <image> tokens
        all_queries = [question.replace(DEFAULT_IMAGE_TOKEN, "") for question in all_queries]
        assert 0 == sum([question.count(DEFAULT_IMAGE_TOKEN) for question in all_queries])
    else:
        # assert sum([img is not None for img in images]) == sum([question.count(DEFAULT_IMAGE_TOKEN) for question in all_queries])
        if sum([img is not None for img in images]) > sum([question.count(DEFAULT_IMAGE_TOKEN) for question in all_queries]):
            too_add = sum([img is not None for img in images]) - sum([question.count(DEFAULT_IMAGE_TOKEN) for question in all_queries])
            all_queries[0] = f"{DEFAULT_IMAGE_TOKEN*too_add} {all_queries[0]}"

    texts = []

    messages = list(chain(*[
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": question}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer}
                        ]
                    }
                ]
                for question, answer in zip(all_queries, all_answers)
            ]
        ))

    text = processor.apply_chat_template(messages, add_generation_prompt=False)

    # if the number of images is larger than 
    if len(images) > data_args.max_image_num:
        images = images[:data_args.max_image_num]
        # remove the <image> tokens 
        tokens = text.split(DEFAULT_IMAGE_TOKEN)
        text = DEFAULT_IMAGE_TOKEN.join(tokens[:data_args.max_image_num + 1]) + "".join(tokens[data_args.max_image_num + 1:])
    if data_args.force_add_eos_token:
        text = text + processor.tokenizer.eos_token
    texts.append(text.strip())

    # FIXME:
    # processor doesn't support passing do_image_splitting as arguments.
    # manually set it for now.

    # change it to self.do_image_splitting
    processor.image_processor.do_image_splitting = data_args.do_image_splitting


    bad_image_flag = False
    if images[0]:
        try:
            batch = processor(text=texts, images=images, return_tensors="pt", padding='max_length', max_length=data_args.seq_length, truncation=True)
        except:
            bad_image_flag = True
            batch = processor(text=[text.replace("<image>", "") for text in texts], return_tensors="pt", padding='max_length', max_length=data_args.seq_length, truncation=True)
            print("bad case", images, texts)
            # Open the file in append mode
            if torch.distributed.get_rank() == 0:
                with open('bad_data_examples.txt', 'a') as file:
                    # Write the new line to the file
                    file.write(texts[0] + '\n')
    else:
        # no image at all
        batch = processor(text=texts, return_tensors="pt", padding='max_length', max_length=data_args.seq_length, truncation=True)

    processor.image_processor.do_image_splitting = False

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = data_args.image_token_id
    batch["labels"] = labels

    END_OF_UTTERANCE = processor.tokenizer.convert_tokens_to_ids(END_OF_UTTERANCE_TOKEN)
    # FIXME:
    # the reason that we do not use get_special_token_id: for idefics2-8b-base, the pre-trained tokenizer does not contain END_OF_UTTERANCE_TOKEN. 
    # consider fixing this in the future.

    batch['answer_mask'] = torch.cat( ([torch.tensor(get_answer_mask(batch["input_ids"][batch_id, :], END_OF_UTTERANCE)).unsqueeze(0) \
        for batch_id in range(batch["input_ids"].shape[0])]), dim=0)


    # FIXME:
    # tq: for simplicity only support one page now.
    # need to also specify do_image_splitting=False when loading the processor

    # pad the second axis of pixel_values to self.max_image_num

    if images[0] and not bad_image_flag:
        batch['pixel_values'] = torch.cat([batch['pixel_values'].to(dtype=torch.half if data_args.fp16 else torch.bfloat16),
                                            torch.zeros(1, data_args.max_image_num - len(images), data_args.num_channel, processor.image_processor.length, processor.image_processor.length).to(dtype=torch.half if data_args.fp16 else torch.bfloat16)], 1)
        # print("batch['pixel_attention_mask']", batch['pixel_attention_mask'].shape)
        # print("self.max_image_num - len(images)", self.max_image_num - len(images))
        batch['pixel_attention_mask'] = torch.cat([batch['pixel_attention_mask'],
                                            torch.zeros(1, data_args.max_image_num - len(images), processor.image_processor.length, processor.image_processor.length).to(torch.int64)], 1)
    else:
        batch['pixel_values'] = torch.zeros(1, data_args.max_image_num, data_args.num_channel, processor.image_processor.length, processor.image_processor.length).to(dtype=torch.half if data_args.fp16 else torch.bfloat16)
        batch['pixel_attention_mask'] = torch.zeros(1, data_args.max_image_num, processor.image_processor.length, processor.image_processor.length).to(torch.int64)

    return {key:val[0] for key, val in batch.items()}


class PackedShardListDataset(wids.ShardListDataset):
    def __init__(
        self,
        shards,
        max_length,
        cache_size=int(1e12),
        cache_dir=None,
        dataset_name=None,
        localname=None,
        transformations="PIL",
        keep=False,
        base=None,
        options=None,
    ):
        '''
            text_data_path: 
        '''
        super(PackedShardListDataset, self).__init__(shards, cache_size=cache_size, cache_dir=cache_dir, dataset_name=dataset_name, localname=localname, transformations=transformations, keep=keep, base=base, options=options)
        self.max_length = max_length
        # self.id2index = {}
        image_count = Counter()
        # self.has_image = []
        # for i in range(len(data)):
        #     if 'image' not in data[i] and 'images' not in data[i]:
        #         self.id2index[data[i]['id']] = i
        #         data[i]['uid'] = data[i]['id']
        #         self.has_image.append(False)
        #     else:
        #         image_dir = data[i]['image'].split('/')[0]
        #         image_count[image_dir] += 1
        #         self.id2index[f"{image_dir}_{image_count[image_dir]}"] = i
        #         data[i]['uid'] =f"{image_dir}_{image_count[image_dir]}"
        #         self.has_image.append(True)
        # print ("size of id2index", len(self.id2index))
        # num_processes = 32  
        # chunk_size = len(data) // num_processes + 1 
        # partial_packing = partial(packing_examples, tokenizer=tokenizer, max_length=max_length)
        # data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        # with Pool(num_processes) as pool:
        #     results = list(pool.imap(partial_packing, data_chunks))

        # self.packed_samples = [item for sublist in results for item in sublist[0]]
        # self.has_image = [item for sublist in results for item in sublist[1]]
        # self.total_length = len(self.packed_samples)
        # print ('Total packed samples', self.total_length)

    # def __len__(self):
    #     """Return the total number of samples in the dataset."""
    #     return self.total_length

    def __getitem__(self, index):
        """Return the sample corresponding to the given index."""
        shard, inner_idx, desc = self.get_shard(index)
        try:
            sample = shard[inner_idx]
        except:
            print (index, inner_idx, desc, 'length of shard', len(shard))

        self.check_cache_misses()

        sample["__dataset__"] = desc.get("dataset")
        sample["__index__"] = index
        sample["__shard__"] = desc["url"]
        sample["__shardindex__"] = inner_idx
        # sample["has_image"] = self.has_image[index]

        for transform in self.transformations:
            sample = transform(sample)
        return sample

def get_sft_dataset(data_path):



    # tokenizer = get_tokenizer()
    data_args = get_args()

    processor = AutoProcessor.from_pretrained(data_args.load, chat_template=IDEFICS2_CHAT_TEMPLATE, do_image_splitting=False)

    ### only for saving the new processor

    processor.image_processor.size['longest_edge'] = data_args.image_size
    if data_args.shortest_edge is not None:
        processor.image_processor.size['shortest_edge'] = data_args.shortest_edge
    else:
        processor.image_processor.size['shortest_edge'] = 0
    processor.tokenizer.padding_side = 'right'
    processor.save_pretrained(data_args.save)

    ###

    processor.image_processor = Idefics2ImageProcessorPad.from_pretrained(data_args.load, do_image_splitting=False)

    processor.image_processor.size['longest_edge'] = data_args.image_size
    if data_args.shortest_edge is not None:
        processor.image_processor.size['shortest_edge'] = data_args.shortest_edge
    else:
        processor.image_processor.size['shortest_edge'] = 0


    processor.image_processor.length = processor.image_processor.size['longest_edge']

    data_args.image_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)

    # FIXME: ftq: right padding is hardcoded.
    processor.tokenizer.padding_side = 'right'
    data_args.num_channel = 3


    partial_make_sample = partial(make_sample, processor=processor, data_args=data_args)
    shardlist = json.load(open(f"{data_path}/shardlist.json"))
    def convert_localname(url):
        return f"{data_path}/{url}"
    localname = {k['url']: f"{data_path}/{k['url']}" for k in shardlist}
    #train_dataset = wids.ShardListDataset(shardlist, localname=convert_localname, keep=True)
    #train_dataset.add_transform(partial_make_sample)
    train_dataset = PackedShardListDataset(shardlist, data_args.max_padding_length, localname=convert_localname, keep=True, transformations=partial_make_sample)
    return train_dataset


def preprocess_multimodal(sources: Sequence[str]) -> Dict:
    args = get_args()
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                # sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                # sentence['value'] = sentence['value'].strip()
                # if "mmtag" in conversation_lib.default_conversation.version:
                #     sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            # replace_token = DEFAULT_IMAGE_TOKEN
            # if args.mm_use_im_start_end:
            #     replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            # sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def get_answer_mask(input_ids, eos_token_id=32002):
    answer_mask = [0] * len(input_ids)

    # iterate over the indices of a and update b accordingly
    count = 0
    start = -1
    for i in range(len(input_ids)):
        if input_ids[i] == eos_token_id:
            count += 1
            if count % 2 == 1:
                start = i
            else:
                # "<end_of_utterance>\nAssistant:"
                # [32002, 28705,    13,  7226, 11143, 28747]
                for j in range(min(start+6, len(answer_mask)-1), i+1):
                    answer_mask[j] = 1

    # if in the end count is an odd number:
    # indicating that the answer may be truncated
    if count % 2 == 1:
        for j in range(min(start+6, len(answer_mask)-1), len(answer_mask)):
            answer_mask[j] = 1

    return answer_mask

class LazySupervisedDataset(torch.utils.data.Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str):
        super(LazySupervisedDataset, self).__init__()
        self.args = get_args()
        self.list_data_dict = json.load(open(data_path[0], "r"))
        self.do_image_splitting = self.args.do_image_splitting

        self.processor = AutoProcessor.from_pretrained(self.args.load, chat_template=IDEFICS2_CHAT_TEMPLATE, do_image_splitting=False)
        ### 

            ### only for saving the new processor

        self.processor.image_processor.size['longest_edge'] = self.args.image_size
        if self.args.shortest_edge is not None:
            self.processor.image_processor.size['shortest_edge'] = self.args.shortest_edge
        else:
            self.processor.image_processor.size['shortest_edge'] = 0
        self.processor.tokenizer.padding_side = 'right'
        self.processor.save_pretrained(self.args.save)

        self.processor.image_processor = Idefics2ImageProcessorPad.from_pretrained(self.args.load, do_image_splitting=False)
        self.processor.image_processor.length = self.processor.image_processor.size['longest_edge']
        self.max_image_num = self.args.max_image_num

        self.image_token_id = get_special_token_id(DEFAULT_IMAGE_TOKEN)

        # FIXME: ftq: right padding is hardcoded.
        self.processor.tokenizer.padding_side = 'right'
        self.max_len = self.args.seq_length
        self.num_channel = 3

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        # print("gook", sources)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            if isinstance(image_file, str):
                # only a single image file:
                image_files = [image_file]
            elif isinstance(image_file, list):
                image_files = image_file

            image_folder = self.args.image_folder
            images = []
            for image_file in image_files:
                try:
                    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                except:
                    image = Image.new('RGB', (self.processor.image_processor.length,
                                              self.processor.image_processor.length), color='white')
                images.append(image)
            # This put all inputs at the beginning
            # sources = preprocess_multimodal(
            #     copy.deepcopy([e["conversations"] for e in sources]))
            sources = copy.deepcopy([e["conversations"] for e in sources])
        else:
            images = [None]
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # process conversation:
        '''
        sources = [ [
                {
                    "from": "human",
                    "value": "Render a clear and concise summary of the photo.\n<image>"
                },
                {
                    "from": "gpt",
                    "value": "select luxury furniture 3 - inch gel memory foam mattress topper"
                }
                ] ]
        '''
        if 'from' in sources[0][0].keys():
            from_key = 'from'
        elif 'role' in sources[0][0].keys():
            from_key = 'role'
        else:
            raise NotImplementedError

        if 'value' in sources[0][0].keys():
            value_key = 'value'
        elif 'content' in sources[0][0].keys():
            value_key = 'content'
        else:
            raise NotImplementedError

        # role_dict = {'from':'human', 'role':'assistant'}

        # pretraining
        assert len(sources[0]) % 2 == 0
        # assert all([sources[0][2*i][from_key] == role_dict[from_key] for i in range(len(sources[0])//2)])
        # assert all([sources[0][2*i + 1][from_key] == role_dict.get('gpt', 'assistant') for i in range(len(sources[0])//2)])

        # example = {"image": image, 'question':question, 'answer':answer}

        # number of images must be equal to the number of <image> tokens
        # print(len(images), question.count("<image>"))

        all_queries = [sources[0][2*i][value_key] for i in range(len(sources[0])//2)]
        all_answers = [sources[0][2*i + 1][value_key] for i in range(len(sources[0])//2)]

        if all([img is None for img in images]):
            assert 0 == sum([question.count(DEFAULT_IMAGE_TOKEN) for question in all_queries])
        else:
            assert sum([img is not None for img in images]) == sum([question.count(DEFAULT_IMAGE_TOKEN) for question in all_queries])


        texts = []
        # images = []
        # for example in examples:
        # image = example["image"]
        # question = example["question"]
        # answer = example["answer"]
        messages = list(chain(*[
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": question}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer}
                        ]
                    }
                ]
                for question, answer in zip(all_queries, all_answers)
            ]
        ))
        text = self.processor.apply_chat_template(messages, add_generation_prompt=False)

        # if the number of images is larger than 
        if len(images) > self.max_image_num:
            images = images[:self.max_image_num]
            # remove the <image> tokens 
            tokens = text.split(DEFAULT_IMAGE_TOKEN)
            text = DEFAULT_IMAGE_TOKEN.join(tokens[:self.max_image_num + 1]) + "".join(tokens[self.max_image_num + 1:])

        texts.append(text.strip())
        # images.append([image])


        # FIXME:
        # processor doesn't support passing do_image_splitting as arguments.
        # manually set it for now.

        # change it to self.do_image_splitting
        self.processor.image_processor.do_image_splitting = self.do_image_splitting

        if images[0]:
            batch = self.processor(text=texts, images=images, return_tensors="pt", padding='max_length', max_length=self.max_len, truncation=True)
        else:
            # no image at all
            batch = self.processor(text=texts, return_tensors="pt", padding='max_length', max_length=self.max_len, truncation=True)

        # change it back to default (False)
        self.processor.image_processor.do_image_splitting = False

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels
        # batch['answer_mask'] = batch["attention_mask"].clone()

        END_OF_UTTERANCE = self.processor.tokenizer.convert_tokens_to_ids(END_OF_UTTERANCE_TOKEN)
        batch['answer_mask'] = torch.cat( ([torch.tensor(get_answer_mask(batch["input_ids"][batch_id, :], END_OF_UTTERANCE)).unsqueeze(0) \
          for batch_id in range(batch["input_ids"].shape[0])]), dim=0)

        # for i in range(batch['answer_mask'].shape[0]):
        #     idx = (batch['input_ids'][i] == END_OF_UTTERANCE).nonzero(as_tuple=True)[0][0]  # find the index of the UTTERANCE_IDX token in the sequence
        #     batch['answer_mask'][i][:idx+1] = 0  # set all the ids after UTTERANCE_IDX to 0


        # pad the second axis of pixel_values to self.max_image_num

        if images[0]:
            batch['pixel_values'] = torch.cat([batch['pixel_values'].to(dtype=torch.half if self.args.fp16 else torch.bfloat16),
                                                torch.zeros(1, self.max_image_num - len(images), self.num_channel, self.processor.image_processor.length, self.processor.image_processor.length).to(dtype=torch.half if self.args.fp16 else torch.bfloat16)], 1)
            # print("batch['pixel_attention_mask']", batch['pixel_attention_mask'].shape)
            # print("self.max_image_num - len(images)", self.max_image_num - len(images))
            batch['pixel_attention_mask'] = torch.cat([batch['pixel_attention_mask'],
                                                torch.zeros(1, self.max_image_num - len(images), self.processor.image_processor.length, self.processor.image_processor.length).to(torch.int64)], 1)
        else:
            batch['pixel_values'] = torch.zeros(1, self.max_image_num, self.num_channel, self.processor.image_processor.length, self.processor.image_processor.length).to(dtype=torch.half if self.args.fp16 else torch.bfloat16)
            batch['pixel_attention_mask'] = torch.zeros(1, self.max_image_num, self.processor.image_processor.length, self.processor.image_processor.length).to(torch.int64)

        return {key:val[0] for key, val in batch.items()}


from datasets import load_dataset
class LazyVQADataset(torch.utils.data.Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, split: str):
        super(LazyVQADataset, self).__init__()
        self.args = get_args()

        self.data = load_dataset(data_path[0], split=split)
        self.processor = AutoProcessor.from_pretrained(self.args.load, do_image_splitting=False)

        # change processor parameters based on args
        self.processor.image_processor = Idefics2ImageProcessorPad.from_pretrained(self.args.load, do_image_splitting=False)
        self.processor.image_processor.size['longest_edge'] = self.args.image_size
        if self.args.shortest_edge is not None :
            self.processor.image_processor.size['shortest_edge'] = self.args.shortest_edge
        else:
            self.processor.image_processor.size['shortest_edge'] = 0
        self.processor.image_processor.length = self.processor.image_processor.size['longest_edge']
        self.max_image_num = self.args.max_image_num

        self.image_token_id = get_special_token_id(DEFAULT_IMAGE_TOKEN)

        # FIXME: ftq: right padding is hardcoded.
        self.processor.tokenizer.padding_side = 'right'
        self.max_len = self.args.seq_length
        self.num_channel = 3

    def __len__(self):
        return len(self.data)

    @property
    def lengths(self):
        length_list = []
        for sample in self.data:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(len(sample['query']['en'].split() + sample['answers'][0].split()) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.data:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(len(sample['query']['en'].split() + sample['answers'][0].split()) + img_tokens)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.data[i]

        image = sources['image']
        question = sources['query']['en']
        answer = sources['answers'][0]

        # process conversation:
        '''
        sources = [ [
                {
                    "from": "human",
                    "value": "Render a clear and concise summary of the photo.\n<image>"
                },
                {
                    "from": "gpt",
                    "value": "select luxury furniture 3 - inch gel memory foam mattress topper"
                }
                ] ]
        '''

        texts = []
        images = []
        # for example in examples:
        # image = example["image"]
        # question = example["question"]
        # answer = example["answer"]
        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        images.append([image])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding='max_length', max_length=self.max_len, truncation=True)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels
        # answer mask: only mask the output of the assistant
        batch['answer_mask'] = batch["attention_mask"].clone()
        # for i in range(batch['answer_mask'].shape[0]):
        #     idx = (batch['input_ids'][i] == END_OF_UTTERANCE).nonzero(as_tuple=True)[0][0]  # find the index of the UTTERANCE_IDX token in the sequence
        #     batch['answer_mask'][i][:idx+1] = 0  # set all the ids after UTTERANCE_IDX to 0

        # FIXME:
        # tq: for simplicity only support one page now.
        # need to also specify do_image_splitting=False when loading the processor

        # pad the second axis of pixel_values to self.max_image_num
        batch['pixel_values'] = batch['pixel_values'].to(dtype=torch.half if self.args.fp16 else torch.bfloat16)

        return {key:val[0] for key, val in batch.items()}