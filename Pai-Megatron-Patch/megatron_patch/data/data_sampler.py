"""Dataloaders."""


import random
import torch
import numpy as np
from torch.utils.data import Dataset
from megatron.training import get_args
from megatron.core import mpu
try:
    from megatron.data.data_samplers import MegatronPretrainingSampler, MegatronPretrainingRandomSampler
except:
    from megatron.legacy.data.data_samplers import MegatronPretrainingSampler, MegatronPretrainingRandomSampler

from transformers import AutoProcessor
from megatron_patch.data.idefics2.constants import END_OF_UTTERANCE

import random
import os
import pdb

class IdeficsDataCollator:
    def __init__(self, processor):
        self.args = get_args()
        self.processor = processor
        self.max_len = self.args.seq_length
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = example["image"]
            question = example["question"]
            answer = example["answer"]
            messages = [
                {
                    "role": "user",
                    "content": [
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

        # TODO: divisible by tp_size.
        # TODO: check split image. 
        # https://git.woa.com/kaixinma/Pai-Megatron-Patch/blob/main/megatron_patch/data/llava/mm_pretrain_dataset.py#L108
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True, max_length=self.max_len, truncation=True)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels
        batch['answer_mask'] = batch["attention_mask"].clone()
        for i in range(batch['answer_mask'].shape[0]):
            idx = (batch['input_ids'][i] == END_OF_UTTERANCE).nonzero(as_tuple=True)[0][0]  # find the index of the UTTERANCE_IDX token in the sequence
            batch['answer_mask'][i][:idx+1] = 0  # set all the ids after UTTERANCE_IDX to 0

        # FIXME:
        # tq: for simplicity only support one page now.
        # need to also specify do_image_splitting=False when loading the processor
        batch['pixel_values'] = batch['pixel_values'].to(dtype=torch.half if self.args.fp16 else torch.bfloat16)

        return batch

def build_pretraining_data_loader(dataset, consumed_samples):
    """Build dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding)
    elif args.dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        return dataset
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    # Torch dataloader.
    collate_fn = torch.utils.data.default_collate
    if args.dataset in ['Idefics2-Pretrain-Raw', 'Idefics2-DocVQA-test']:
        processor = AutoProcessor.from_pretrained(args.load, do_image_splitting=False)
        processor.tokenizer.padding_side = 'right'
        collate_fn =  IdeficsDataCollator(processor)
        
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       persistent_workers=True if args.num_workers > 0 else False,
                                       collate_fn=collate_fn
                                       )