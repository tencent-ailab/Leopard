# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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

from functools import partial
import torch

from megatron.core.enums import ModelType
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training import get_args
from megatron.training import get_timers
from megatron.core import tensor_parallel
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.legacy.data.data_samplers import MegatronPretrainingRandomSampler
from megatron.training.training import cyclic_iter
from megatron.core import mpu

from megatron_patch.data import build_pretrain_dataset_from_original
from megatron_patch.model.llava.gpt_model import GPTModel
from megatron_patch.tokenizer import build_tokenizer
from megatron_patch.tokenizer import get_tokenizer
from megatron_patch.training import pretrain
from megatron_patch.arguments import get_patch_args
from megatron_patch.data.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

def model_provider(pre_process=True, post_process=True):
    args = get_args()
    build_tokenizer(args)
    print ('Building LLaVA model, preprocess =', pre_process, 'postprocess =', post_process)
    config = core_transformer_config_from_args(get_args())
    config.variable_seq_lengths = True
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    text_keys = ['input_ids', 'labels']
    img_keys = ['image']
    if data_iterator is not None:
        data = next(data_iterator)
        while data is None:
            data = next(data_iterator)
    else:
        data = None
    #data_text = {'input_ids': data['input_ids'], 'labels': data['labels']}
    #data_image = {'image': data['image']}
    data_text = tensor_parallel.broadcast_data(text_keys, data, torch.int64)
    tokens = data_text['input_ids'].long()
    labels = data_text['labels'].long()
    
    data_b1 = tensor_parallel.broadcast_data(['weights'], data, torch.float32)
    loss_mask = data_b1['weights']
    #print (labels[:, :20])
    #if data is not None and data['image'] is not None:
    data_image = tensor_parallel.broadcast_data(img_keys, data, torch.bfloat16)
    images = data_image['image']
    #print ('tokens', tokens.shape, 'labels', labels.shape, 'images', images.shape)
    #else:
    #    images = None
    #    print ('tokens', tokens.shape, 'labels', labels.shape)
    # for tk in tokens:
    #     print (tokenizer.convert_ids_to_tokens([x for x in tk if x != IGNORE_INDEX]))
    #exit()
    # tokens = tokens_[:, :-1].contiguous()
    # labels = labels_[:, 1:].contiguous()
    #attention_mask = tokens.ne(IGNORE_INDEX)
    # Get the masks and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        labels,
        tokenizer.eos_token_id,
        args.reset_position_ids,
        args.reset_attention_mask,
        True)
    #print ('position ids', (position_ids == 1).sum(dim=1))
    # print ('loss mask', loss_mask.shape, 'position ids', position_ids.shape)
    # print (loss_mask[:, :20])
    # exit()
    return tokens, labels, loss_mask, attention_mask, position_ids, images

def loss_func(loss_mask, labels, output_tensor):
    #losses = output_tensor.float()
    logits = output_tensor[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    loss_mask = loss_mask[:, 1:].contiguous()
    #print ('loss mask', loss_mask.shape, loss_mask.sum(dim=1))

    #print ('logits', logits.shape, 'labels', labels.shape)
    losses = tensor_parallel.vocab_parallel_cross_entropy(
            logits.contiguous().float(), labels.contiguous())
    # if mpu.get_tensor_model_parallel_rank() == 0:
    #     print ('losses', losses.shape, losses[0, 735:740], 'labels', labels.shape, labels[0, 735:740])
    # print (losses.shape)
    # print (losses[0][loss_mask[0] == 1])
    # exit()
    loss_mask = loss_mask.view(-1).float()    
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    #exit()
    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()
    # args = get_args()
    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, images = get_batch(
        data_iterator)
    # num_patch = int((args.image_size / args.patch_size) ** 2)
    timers('batch-generator').stop()

    # new_labels = []
    # new_loss_mask = []
    # for i in range(len(tokens)):
    #     image_token_indices = torch.where(tokens[i] == IMAGE_TOKEN_INDEX)[0]
    #     curr_start = 0
    #     curr_label = []
    #     curr_loss_mask = []
    #     while image_token_indices.numel() > 0:
    #         index = image_token_indices[0]
    #         curr_label.append(labels[i, curr_start:curr_start+index])
    #         curr_label.append(torch.full((num_patch,), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
    #         curr_loss_mask.append(loss_mask[i, curr_start:curr_start+index])
    #         curr_loss_mask.append(torch.zeros(num_patch, device=labels.device, dtype=labels.dtype))
    #         curr_start = curr_start + index + 1
    #         image_token_indices = torch.where(tokens[i][curr_start:] == IMAGE_TOKEN_INDEX)[0]
    #     curr_label.append(labels[i, curr_start:])
    #     curr_loss_mask.append(loss_mask[i, curr_start:])
    #     new_labels.append(torch.cat(curr_label, dim=0)[:tokens.shape[1]])
    #     new_loss_mask.append(torch.cat(curr_loss_mask, dim=0)[:tokens.shape[1]])
    # total_label = torch.stack(new_labels, dim=0)
    # total_loss_mask = torch.stack(new_loss_mask, dim=0)

    output_tensor = model(tokens, position_ids, attention_mask,
                          images=images)

    return output_tensor, partial(loss_func, loss_mask, labels)


def mm_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        print ('batch empty')
        return None
    max_len = max([len(item['input_ids']) for item in batch])
    while max_len % 8 != 0:
        max_len += 1
    input_ids = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    weights = torch.zeros(len(batch), max_len, dtype=torch.float)
    preferences = torch.zeros(len(batch), max_len, dtype=torch.float)
    for i, item in enumerate(batch):
        input_ids[i, :len(item['input_ids'])] = item['input_ids']
        labels[i, :len(item['labels'])] = item['labels']
        weights[i, :len(item['weights'])] = item['weights']
        preferences[i, :len(item['preferences'])] = item['preferences']
    # input_ids = torch.stack([item['input_ids'] for item in batch])
    # labels = torch.stack([item['labels'] for item in batch])
    # weights = torch.stack([item['weights'] for item in batch])
    # preferences = torch.stack([item['preferences'] for item in batch])
    batch_images = [item['image'] for item in batch if 'image' in item and item['image'] is not None]
    if len(batch_images) > 0:
        images = torch.cat(batch_images, dim=0)
    else:
        images = torch.zeros(1, 10, dtype=torch.bfloat16)

    return {'input_ids': input_ids, 'labels': labels, 'weights': weights, 'preferences': preferences, 'image': images}

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = \
        build_pretrain_dataset_from_original(args.dataset)

    if args.dataloader_type in ['cyclic', 'single']:
        return train_ds, valid_ds, test_ds
    batch_sampler = MegatronPretrainingRandomSampler(
            train_ds,
            total_samples=len(train_ds),
            consumed_samples=0,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding)
    print ('Current data parallel rank is', mpu.get_data_parallel_rank(), 'world size', mpu.get_data_parallel_world_size())
    train_loader = torch.utils.data.DataLoader(train_ds,
                                        batch_sampler=batch_sampler,
                                        num_workers=args.num_workers,
                                        pin_memory=True,
                                        persistent_workers=True if args.num_workers > 0 else False,
                                        collate_fn = mm_collate_fn
                                        )
    train_iterator = iter(cyclic_iter(train_loader))
    return train_iterator, train_iterator, train_iterator


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_patch_args)
