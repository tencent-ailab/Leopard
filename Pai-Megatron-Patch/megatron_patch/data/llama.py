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

import numpy as np
import io
import copy
import json
import torch
try:
    from megatron import get_args
except:
    from megatron.training import get_args
from datasets import load_dataset
from tqdm import tqdm

from multiprocessing import Pool

from megatron_patch.tokenizer import get_tokenizer
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

"""
PROMPT_DICT = {
            'prompt_input':
            ('Below is an instruction that describes a task,'
             ' paired with an input that provides further context. '
             'Write a response that appropriately completes the request.\n\n'
             '### Instruction:\n{instruction}'
             '\n\n### Input:\n{input}\n\n### Response:'),
            'prompt_no_input':
            ('Below is an instruction that describes a task. '
             'Write a response that appropriately completes the request.\n\n'
             '### Instruction:\n{instruction}\n\n### Response:'),
        }

PROMPT_DICT = {
    'prompt_input': ('[INST]{instruction} {input}\n[/INST]\n'),
    'prompt_no_input':('[INST]{instruction}\n[/INST]\n'),
}

PROMPT_DICT = {
    'prompt_input': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction} {input}<|im_end|>\n<|im_start|>assistant\n',
    'prompt_no_input':'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n'
}
"""

PROMPT_DICT = {
    'prompt_input': ('{instruction} {input}'),
    'prompt_no_input': ('{instruction}'),
}

class CKDataset(torch.utils.data.Dataset):
    """A class for processing a LLama text dataset"""

    def __init__(self, path, max_padding_length, split='train'):
        args = get_args()
        self.tokenizer = get_tokenizer()
        self.IGNORE_INDEX = self.tokenizer.pad_token_id
        if "-Pretrain" in args.dataset:
            self.max_padding_length = max_padding_length + 1
        else:
            self.max_padding_length = max_padding_length

        self.bos_token = "<|im_start|>"
        self.eos_token = "<|im_end|>"
        
        raw_data = list()
        with open(path[0], 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    raw_data.append(json.loads(line)['messages'])
                except UnicodeDecodeError:
                    print(line)
                    pass
        print('start encoding')
        self.do_packing = False
        if self.do_packing:
            print ('Packing data to improve efficicency')

        num_processes = 32  # 进程数
        chunk_size = len(raw_data) // num_processes + 1 # 计算每个数据块的大小

        # 将数据分块
        data_chunks = [raw_data[i:i + chunk_size] for i in range(0, len(raw_data), chunk_size)]
        with Pool(num_processes) as pool:
            # map函数自动分发数据块到各个进程
            results = list(pool.imap(self.batch_process, data_chunks))

        # 关闭并等待进程结束已经由上下文管理器处理

        # 重新组织结果列表，过滤掉None项
        self.samples = [item for sublist in results for item in sublist if item[0] is not None]
        original_size = len(self.samples)
        if self.do_packing:
            self.samples = self.packing_examples(self.samples)

        print('  >> total number of samples: {}'.format(len(self.samples)))
        packed_size = len(self.samples)
        print(f'  >> original size: {original_size}, packed size: {packed_size}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample)

    def packing_examples(self, examples):
        packed_examples = []
        tmp_input_ids = np.array([], dtype=np.int64)
        tmp_labels = np.array([], dtype=np.int64)
        tmp_weights = np.array([], dtype=np.float32)
        tmp_preferences = np.array([], dtype=np.int64)
        for i, example in enumerate(examples):
            if len(tmp_input_ids) + len(example[0]) > self.max_padding_length:
                tmp_input_ids = np.concatenate([tmp_input_ids, np.array([self.tokenizer.pad_token_id]*(self.max_padding_length - len(tmp_input_ids)), dtype=np.int64)])
                tmp_labels = np.concatenate([tmp_labels, np.array([self.tokenizer.pad_token_id]*(self.max_padding_length - len(tmp_labels)), dtype=np.int64)])
                tmp_weights = np.concatenate([tmp_weights, np.array([0.0]*(self.max_padding_length - len(tmp_weights)), dtype=np.float32)]) 
                tmp_preferences = np.concatenate([tmp_preferences, np.array([0]*(self.max_padding_length - len(tmp_preferences)), dtype=np.int64)])
                packed_examples.append((tmp_input_ids, tmp_labels, tmp_weights, tmp_preferences))
                tmp_input_ids = np.array([], dtype=np.int64)
                tmp_labels = np.array([], dtype=np.int64)
                tmp_weights = np.array([], dtype=np.float32)
                tmp_preferences = np.array([], dtype=np.int64)
                if tmp_input_ids.dtype != np.int64 or tmp_labels.dtype != np.int64:
                    print (i) 
            tmp_input_ids = np.concatenate([tmp_input_ids, example[0]])
            tmp_labels = np.concatenate([tmp_labels, example[1]])
            tmp_weights = np.concatenate([tmp_weights, example[2]])
            tmp_preferences = np.concatenate([tmp_preferences, example[3]])
        if len(tmp_input_ids) > 0:
            tmp_input_ids = np.concatenate([tmp_input_ids, np.array([self.tokenizer.pad_token_id]*(self.max_padding_length - len(tmp_input_ids)), dtype=np.int64)])
            tmp_labels = np.concatenate([tmp_labels, np.array([self.tokenizer.pad_token_id]*(self.max_padding_length - len(tmp_labels)), dtype=np.int64)])
            tmp_weights = np.concatenate([tmp_weights, np.array([0.0]*(self.max_padding_length - len(tmp_weights)), dtype=np.float32)])
            tmp_preferences = np.concatenate([tmp_preferences, np.array([0]*(self.max_padding_length - len(tmp_preferences)), dtype=np.int64)])
            if tmp_input_ids.dtype != np.int64 or tmp_labels.dtype != np.int64:
                print ('this should not happen')
            packed_examples.append((tmp_input_ids, tmp_labels, tmp_weights, tmp_preferences))
        return packed_examples

    def batch_process(self, messages):
        tmp_results = list()
        for message in tqdm(messages):
            tmp_results.append(self.preprocess_single_message(message)) 
        return tmp_results
    
    def preprocess_single_message(self, message):
        tmp_input_ids = []
        tmp_labels = []
        tmp_weights = [] # float
        tmp_preferences = [] # int, 2 means positive, 1 means negative, 0 means others
        for turn in message:
            if "name" in turn and turn['name'] != '':
                turn_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(f"{self.bos_token}{turn['role']} name={turn['name']}\n{turn['content']}{self.eos_token}"))
            else:
                turn_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(f"{self.bos_token}{turn['role']}\n{turn['content']}{self.eos_token}"))
            tmp_input_ids.extend(turn_tokens)
            tmp_labels.extend(turn_tokens)
            if turn['role'] in ['玩家', 'system', "策略", "领域知识", "长期记忆", "user"]:
                # model input
                tmp_weights.extend([0.0] * len(turn_tokens))
                tmp_preferences.extend([0] * len(turn_tokens))
            else:
                # model output
                prefix_len = len(self.tokenizer.tokenize(f"{self.bos_token}{turn['role']}\n"))
                left_token_len = len(turn_tokens)-prefix_len
                tmp_weights.extend([0.0] * prefix_len)
                tmp_preferences.extend([0] * prefix_len)
                if 'loss' in turn:
                    tmp_weights.extend([float(turn['loss'])]*left_token_len)
                else:
                    tmp_weights.extend([1.0] * left_token_len)
                if 'label' in turn:
                    tmp_preferences.extend([int(turn['label'])] * left_token_len)
                else:
                    tmp_preferences.extend([2] * left_token_len)
        # tmp_input_ids.append(self.tokenizer.eos_token_id)
        # tmp_labels.append(self.tokenizer.eos_token_id)
        # tmp_weights.append(0.0)
        # tmp_preferences.append(0)
        if len(tmp_input_ids) > self.max_padding_length:
            # This instance is longer than the max_padding_length. We should directly remove it to avoid potential issues
            print('exceeding the max length, this example will be removed')
            return None, None, None, None
        # if not self.do_packing:
        #     # if we are not doing packing, we will pad every single instance
        #     tmp_input_ids.extend([self.tokenizer.pad_token_id]*(self.max_padding_length-len(tmp_input_ids)))
        #     tmp_labels.extend([self.tokenizer.pad_token_id]*(self.max_padding_length-len(tmp_labels)))
        #     tmp_weights.extend([0.0]*(self.max_padding_length-len(tmp_weights)))
        #     tmp_preferences.extend([0]*(self.max_padding_length-len(tmp_preferences)))
        #     assert len(tmp_input_ids) == len(tmp_labels) == len(tmp_weights) == len(tmp_preferences) == self.max_padding_length
        tmp_input_ids = np.array(tmp_input_ids, dtype=np.int64)
        tmp_labels = np.array(tmp_labels, dtype=np.int64)
        tmp_weights = np.array(tmp_weights, dtype=np.float32)
        tmp_preferences = np.array(tmp_preferences, dtype=np.int64)
        return tmp_input_ids, tmp_labels, tmp_weights, tmp_preferences

    def preprocess(self, examples):
        """
        Preprocess the data by tokenizing.
        Args:
            sources (List[str]): a list of source strings
            targets (List[str]): a list of target strings
            tokenizer (Tokenizer): a tokenizer object used for tokenization
        Returns:
            dict: a dictionary containing the input_ids and labels for the examples
        """
        input_ids = []
        labels = []
        weights = []
        preferences = []
        for message in examples['messages']:
            tmp_input_ids, tmp_labels, tmp_weights, tmp_preferences = self.preprocess_single_message(message)
            if tmp_input_ids is not None:
                input_ids.append(tmp_input_ids)
                labels.append(tmp_labels)
                weights.append(tmp_weights)
                preferences.append(tmp_preferences)

        return dict(input_ids=input_ids, labels=labels, weights=weights, preferences=preferences)

    def tokenize(self, strings, tokenizer):
        """
        Tokenize a list of strings.
        Args:
            strings (List[str]): a list of strings to be tokenized
            tokenizer (Tokenizer): a tokenizer object used for tokenization
        Returns:
            dict: a dictionary containing the input_ids and labels for the tokenized strings
        """

        tokenized_list = [
            tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_padding_length,
                truncation=True,
                add_special_tokens=False
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            (tokenized.input_ids != tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def gpt_convert_example_to_feature(self, sample):
        """
        Convert a single sample containing input_id, label and loss_mask into a format suitable for GPT training.
        """
        input_ids, labels, weights, preference = sample
        train_sample = {
            'input_ids': input_ids,
            'labels': labels,
            'weights': weights,
            'preferences': preference
        }

        return train_sample

class LLamaRawDataset(torch.utils.data.Dataset):
    """A class for processing a LLama text dataset"""

    def __init__(self, path, max_padding_length, split='train'):
        args = get_args()
        self.tokenizer = get_tokenizer()
        self.IGNORE_INDEX = self.tokenizer.pad_token_id
        if "-Pretrain" in args.dataset:
            self.max_padding_length = max_padding_length + 1
        else:
            self.max_padding_length = max_padding_length

        list_data_dict = load_dataset(
            'json',
            data_files=path[0],
            split=split,
        )

        train_dataset = list_data_dict.map(
            self.preprocess,
            batched=True,
            batch_size=3000,
            num_proc=16,
            remove_columns=list_data_dict.column_names,
            load_from_cache_file=False,
            desc="Running Encoding"
        )

        self.input_ids = np.array(train_dataset['input_ids'])
        self.labels = np.array(train_dataset['labels'])
        self.samples = []

        for inputs, labels in tqdm(zip(self.input_ids, self.labels)):
            if self.tokenizer.eos_token_id not in inputs: continue
            self.samples.append([inputs, labels])

        print('  >> total number of samples: {}'.format(len(self.samples)))

    def _make_r_io_base(self, f, mode: str):
        if not isinstance(f, io.IOBase):
            f = open(f, mode=mode, encoding='utf-8')
        return f

    def jload(self, f, mode='r'):
        """
        Load a .json file into a dictionary.
        Args:
            f: The file object or string representing the file path.
            mode: The mode in which to open the file (e.g., 'r', 'w', 'a').
        Returns:
            A dictionary containing the contents of the JSON file.
        """
        f = self._make_r_io_base(f, mode)
        jdict = json.load(f)
        f.close()
        return jdict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample)

    def preprocess(self, examples):
        """
        Preprocess the data by tokenizing.
        Args:
            sources (List[str]): a list of source strings
            targets (List[str]): a list of target strings
            tokenizer (Tokenizer): a tokenizer object used for tokenization
        Returns:
            dict: a dictionary containing the input_ids and labels for the examples
        """

        prompt_input, prompt_no_input = PROMPT_DICT[
            'prompt_input'], PROMPT_DICT['prompt_no_input']

        sources = []
        if 'input' not in examples:
            if 'instruction' in examples:
                for instruction in examples['instruction']:
                    sources.append(prompt_no_input.format_map({"instruction": instruction}))
            elif 'query' in examples:
                for query in examples['query']:
                    sources.append(prompt_no_input.format_map({"instruction": query}))
        else:
            if 'instruction' in examples:
                for instruction, minput in zip(examples['instruction'], examples['input']):
                    sources.append(prompt_input.format_map({"instruction": instruction, "input": minput}))
            elif 'query' in examples:
                for query, minput in zip(examples['query'], examples['input']):
                    sources.append(prompt_input.format_map({"instruction": query, "input": minput}))

        if 'output' in examples:
            key = 'output'
        elif 'content' in examples:
            key = 'content'
        elif 'response' in examples:
            key = 'response'

        targets = [
            example + self.tokenizer.eos_token
            for example in examples[key]
        ]

        examples_raw = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [
            self.tokenize(strings, self.tokenizer)
            for strings in (examples_raw, sources)
        ]
        input_ids = examples_tokenized['input_ids']
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels,
                                     sources_tokenized['input_ids_lens']):
            label[:source_len] = self.IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def tokenize(self, strings, tokenizer):
        """
        Tokenize a list of strings.
        Args:
            strings (List[str]): a list of strings to be tokenized
            tokenizer (Tokenizer): a tokenizer object used for tokenization
        Returns:
            dict: a dictionary containing the input_ids and labels for the tokenized strings
        """

        tokenized_list = [
            tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_padding_length,
                truncation=True,
                add_special_tokens=False
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            (tokenized.input_ids != tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def gpt_convert_example_to_feature(self, sample):
        """
        Convert a single sample containing input_id, label and loss_mask into a format suitable for GPT training.
        """
        input_ids, labels = sample
        train_sample = {
            'input_ids': input_ids,
            'labels': labels
        }

        return train_sample


class LLamaIdxMapDataset(torch.utils.data.Dataset):
    """LLAMA dataset class for mmap format data"""

    def __init__(self,
                 name,
                 data_prefix,
                 documents,
                 indexed_dataset,
                 num_samples,
                 seed,
                 max_padding_length,
                 return_doc_ids=False):

        args = get_args()
        self.tokenizer = get_tokenizer()
        self.max_padding_length = max_padding_length

        self.name = name
        self.indexed_dataset = indexed_dataset
        self.return_doc_ids = return_doc_ids
        self.split = args.split
        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Build index mappings.
        from megatron.data.gpt_dataset import _build_index_mappings
        try:
            self.doc_idx, self.sample_idx, self.shuffle_idx, self.index_prefix = \
                _build_index_mappings(self.name, data_prefix,
                                      documents, self.indexed_dataset.sizes,
                                      num_samples, self.max_padding_length, seed)
        except:
            self.doc_idx, self.sample_idx, self.shuffle_idx, self.desc, self.desc_hash = \
                _build_index_mappings(self.name, data_prefix,
                                      documents, self.indexed_dataset.sizes,
                                      self.split, num_samples, self.max_padding_length, seed,
                                      data_cache_path=None)

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        doc_ids = []

        if doc_index_f == doc_index_l:
            doc_ids.append(self.doc_idx[doc_index_f])
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            doc_ids.append(self.doc_idx[doc_index_f])
            sample_list = [
                self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                         offset=offset_f)
            ]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                doc_ids.append(self.doc_idx[i])
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            doc_ids.append(self.doc_idx[doc_index_l])
            sample_list.append(
                self.indexed_dataset.get(self.doc_idx[doc_index_l],
                                         length=offset_l + 1))
            sample = np.concatenate(sample_list)

        tokens = sample.tolist()
        sample = []
        sample.append(np.array(tokens))
        sample.append(np.array(tokens))

        return self.gpt_convert_example_to_feature(sample)

    def gpt_convert_example_to_feature(self, sample):
        input_ids, labels = sample
        loss_mask = np.ones(labels.shape, dtype=np.int64)
        loss_mask[labels == self.tokenizer.bos_token_id] = 0
        loss_mask[labels == self.tokenizer.pad_token_id] = 0
        train_sample = {
            'input_ids': input_ids,
            'labels': labels,
            'loss_mask': loss_mask
        }

        return train_sample