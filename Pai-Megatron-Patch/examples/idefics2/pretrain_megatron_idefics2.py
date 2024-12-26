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
import pdb

from megatron.core.enums import ModelType
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training import get_args
from megatron.training import get_timers
from megatron.core import tensor_parallel
from megatron.training.utils import average_losses_across_data_parallel_group

from megatron_patch.data import build_pretrain_dataset_from_original
from megatron_patch.model.idefics2.gpt_model import GPTModel
from megatron_patch.tokenizer import build_tokenizer, get_special_token_id, get_image_token_rank_and_id
from megatron_patch.tokenizer import get_tokenizer
from megatron_patch.training import pretrain
from megatron_patch.arguments import get_patch_args
from megatron_patch.data.idefics2.constants import DEFAULT_IMAGE_TOKEN
# IGNORE_INDEX, IMAGE_TOKEN_INDEX, PAD_IDX, AROUND_IMAGE_TOKEN_INDEX, RELETIVE_IMAGE_TOKEN_ID
from transformers import AutoConfig

def model_provider(pre_process=True, post_process=True):
    args = get_args()
    build_tokenizer(args)
    print ('Building Idefics2 model, preprocess =', pre_process, 'postprocess =', post_process)
    config = core_transformer_config_from_args(get_args())
    idefics2_config = AutoConfig.from_pretrained(args.load)
    assert idefics2_config.text_config.max_position_embeddings == args.max_position_embeddings,\
        "max_position_embeddings should be the same as in the config"
    assert idefics2_config.text_config.rope_theta == args.rotary_base, \
        "rotary_theta should be the same as in the config"

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
    text_attention_keys = ['attention_mask']
    img_keys = ['pixel_values']
    img_mask_keys = ['pixel_attention_mask']
    # pdb.set_trace()

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    #data_text = {'input_ids': data['input_ids'], 'labels': data['labels']}
    #data_image = {'image': data['image']}
    data_text = tensor_parallel.broadcast_data(text_keys, data, torch.int64)
    data_image = tensor_parallel.broadcast_data(img_keys, data, torch.bfloat16 if args.bf16 else torch.float16)
    image_attention_masks = tensor_parallel.broadcast_data(img_mask_keys, data, torch.int64)[img_mask_keys[0]]
    tokens = data_text['input_ids'].long()
    labels = data_text['labels'].long()
    images = data_image[img_keys[0]]
    #print (labels[:, :20])
    #print ('tokens', tokens.shape, 'labels', labels.shape, 'images', images.shape)
    # tokens = tokens_[:, :-1].contiguous()
    # labels = labels_[:, 1:].contiguous()
    # attention_mask = tokens.ne(IGNORE_INDEX)
    # attention_mask = tokens.ne(IGNORE_INDEX)
    attention_mask = tensor_parallel.broadcast_data(text_attention_keys, data, torch.int64)[text_attention_keys[0]]
    answer_mask = tensor_parallel.broadcast_data(['answer_mask'], data, torch.int64)['answer_mask'] # mask for answers
    # Get the masks and postition ids.
    # _, loss_mask, position_ids = get_ltor_masks_and_position_ids(
    #     tokens,
    #     IGNORE_INDEX,
    #     args.reset_position_ids,
    #     args.reset_attention_mask,
    #     True)
    loss_mask = torch.ones(tokens.size(), dtype=torch.float, device=tokens.device)
    # if eod_mask_loss:
    # loss_mask[tokens == PAD_IDX] = 0.0 # for unk (padding tokens), don't calc loss.
    IMAGE_TOKEN_INDEX = get_special_token_id(DEFAULT_IMAGE_TOKEN)
    loss_mask[tokens == IMAGE_TOKEN_INDEX] = 0.0
    loss_mask[attention_mask == 0] = 0.0
    # loss_mask[tokens == 0] = 0.0 # unk
    
    # loss_mask[tokens == AROUND_IMAGE_TOKEN_INDEX] = 0.0
    
    # for idefics
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    
    if args.answer_loss_only:
        loss_mask = loss_mask * answer_mask
        
    return tokens, labels, loss_mask, attention_mask, position_ids, images, image_attention_masks

def loss_func(loss_mask, labels, pad_vocab_size, vocab_size, output_tensor):
    # if torch.distributed.get_rank() == 0 :
    #     torch.save(loss_mask, "debug_output/loss_mask.pt")
    #     torch.save(answer_mask, "debug_output/answer_mask.pt")

    # WARNING: 
    # mask out pad_vocab_size for the last gpu    
    # this is for test only, to check the loss of megatron and huggingface.

    # if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1: 
    #     # this operation is used to discard the logits of all the un-trained tokens (like the padded tokens for tensor parallel)
    #     output_tensor[:, :, -pad_vocab_size:] = -100
    # img_token_tp_rank, relative_image_token_id = get_image_token_rank_and_id(DEFAULT_IMAGE_TOKEN, gpu_num=torch.distributed.get_world_size(), vocab_size=vocab_size)

    # if torch.distributed.get_rank() == img_token_tp_rank: 
    #     # output_tensor[:, :, relative_image_token_id] = -100 # this is the ignore_index when calculating cross entropy.
    #     mask = torch.ones(output_tensor.shape[-1]).to(torch.long)
    #     mask[relative_image_token_id] = 0
    #     # output_tensor = output_tensor[:, :, relative_image_token_id]
    #     output_tensor = output_tensor[:, :, mask]

    # assert img_token_tp_rank * vocab_size//torch.distributed.get_world_size() + relative_image_token_id == get_special_token_id(DEFAULT_IMAGE_TOKEN)

    # if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1: 
    #     print("output_tensor", output_tensor[0, 4, -30:-20], output_tensor[0, 4, -20:])

    logits = output_tensor[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    loss_mask = loss_mask[:, 1:].contiguous()
    # need to mask out in advance.
    
    losses = tensor_parallel.vocab_parallel_cross_entropy(
            logits.contiguous().float(), labels.contiguous())
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()
    args = get_args()
    pad_vocab_size = args.pad_vocab_size
    vocab_size = args.padded_vocab_size
    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, images, image_attention_mask = get_batch(
        data_iterator)

    timers('batch-generator').stop()

    # print("in pretrain.py", images.shape)

    output_tensor = model(tokens, position_ids, attention_mask,
                          pixel_values=images, pixel_attention_mask=image_attention_mask)

    # return output_tensor, partial(loss_func, total_loss_mask, total_label)
    return output_tensor, partial(loss_func, loss_mask, labels, pad_vocab_size, vocab_size)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = \
        build_pretrain_dataset_from_original(args.dataset)
    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_patch_args)