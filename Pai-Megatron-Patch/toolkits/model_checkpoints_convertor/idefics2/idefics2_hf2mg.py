# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import argparse
import random
import json
import os
import re
import sys
import types
import numpy as np
import torch
import pdb
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

from transformers import AutoTokenizer, GPT2Config, MistralConfig, MistralForCausalLM, AutoConfig
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint
from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionConfig, Idefics2Config, Idefics2VisionTransformer, Idefics2Connector, Idefics2ForConditionalGeneration


def add_checkpointing_args(parser):
    parser.add_argument("--megatron-path", type=str, default=None, help="Base directory of Megatron repository")
    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        default="",
        help="Path to the config to convert.",
    )
    
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model name",
    )
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    return parser


def add_megatron_checkpoint_args(parser):
    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_data_parallel_size",
        type=int,
        default=1,
        help=(
            "The data parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficieny reasons. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    parser.add_argument(
        '--extra_num_vocabs',
        type=int,
        default=0,
    )

    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help=(
            "If True, use the distributed optimizer. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    return parser


def add_transformers_checkpoint_args(parser):
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "The name of the pre-trained tokenizer to save. "
            "If not None, the tokenizer will be saved. "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    return parser

transformers_to_megatron = {
    "self_attn.dense": "self_attention.dense",
    "mlp.dense_h_to_4h_1": "mlp.dense_h_to_4h_1",
    "mlp.dense_h_to_4h_2": "mlp.dense_h_to_4h_2",
    "mlp.dense_4h_to_h": "mlp.dense_4h_to_h",
}

tensor_parallel_params = [
    # megatron-lm layers to merge across tp ranks
    "self_attn.query.weight",
    "self_attn.key_value.weight",
    "self_attn.dense.weight",
    "mlp.dense_h_to_4h_1.weight",
    "mlp.dense_h_to_4h_2.weight",
    "mlp.dense_4h_to_h.weight"
]

tensor_parallel_params = [
    # megatron-lm layers to merge across tp ranks
    "self_attn.query_key_value.weight",
    "self_attn.query_key_value.bias",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "self_attn.query.weight",
    "self_attn.key_value.weight",
    "self_attn.dense.weight",
    "mlp.dense_h_to_4h_1.weight",
    "mlp.dense_h_to_4h_2.weight",
    "mlp.dense_4h_to_h.weight"
]

megatron_to_transformers = {
    "self_attention.dense": ".self_attn.o_proj.",
    "mlp.dense_h_to_4h": [".mlp.gate_proj.",".mlp.up_proj."],
    'dense_h_to_4h': [".gate_proj.",".up_proj."],
    "mlp.dense_4h_to_h": ".mlp.down_proj.",
    "input_norm":".input_layernorm.",
    "post_attention_norm":".post_attention_layernorm.",
    "self_attention.rotary_emb":".self_attn.rotary_emb.inv_freq"
}

tensor_parallel_params_mg = [
    # megatron-lm layers to merge across tp ranks
    "self_attention.query_key_value.weight",
    "self_attention.query.weight",
    "self_attention.key_value.weight",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
]

def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint. This function is taken from `convert_megatron_gpt2_checkpoint.py`
    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def megatron_to_transformers_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :] for compatibility with later versions
    of NVIDIA Megatron-LM. The inverse operation is performed inside Megatron-LM to read checkpoints:
    https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209 If param is the weight tensor of the
    self-attention block, the returned tensor will have to be transposed one more time to be read by HuggingFace GPT2.
    This function is taken from `convert_megatron_gpt2_checkpoint.py`
    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def transformers_to_megatron_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to the one compatible with respective NVIDIA Megatron-LM chekpoint versions. Input
    is [num_splits * num_heads * hidden_size, :] and output is [num_heads * hidden_size * num_splits, :] for version
    1.0 and [num_heads * num_splits * hidden_size, :] for version 2.0 and later. If param is the weight tensor of the
    self-attention block, the param needs to be already transposed before calling this function.
    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    # Input is [num_splits * num_heads * hidden_size, :]
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def merge_transformers_sharded_states_7b(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.
    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    state_dict = {}
    for i in range(1, num_checkpoints + 1):
        checkpoint_path = os.path.join(path, f"pytorch_model-{i:05d}-of-{num_checkpoints:05d}.bin")
        current_chunk = torch.load(checkpoint_path, map_location="cpu")
        state_dict.update(current_chunk)
    return state_dict


def merge_transformers_sharded_states_13b(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.
    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    state_dict = {}
    for i in range(0, num_checkpoints + 1):
        checkpoint_path = os.path.join(path, f"pytorch_model-{i:05d}-of-{num_checkpoints:05d}.bin")
        current_chunk = torch.load(checkpoint_path, map_location="cpu")
        state_dict.update(current_chunk)
    return state_dict


def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.
    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir_name))[0]
        checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.
    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split(".")
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d

def _init_embedding_weights(module):
    std = 0.02
    module.weight.data.normal_(mean=0.0, std=std)


def convert_checkpoint_from_transformers_to_megatron(args):
    """
    Convert a checkpoint from HuggingFace Transformers to Megatron-LM. This allows converted checkpoints with variable
    tensor parallelism and pipeline parallelism sizes. It takes as input a checkpoint from HuggingFace Transformers
    which can have multiple shards.
    Args:
        args (argparse.Namespace): the arguments to the script
    """
    os.makedirs(args.save_path, exist_ok=True)
    # Search in directory above this
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    # try:
    #     from megatron.tokenizer.tokenizer import _vocab_size_with_padding
    # except ModuleNotFoundError:
    #     print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
    #     exit(1)

    # # load the transformers model state dict and config
    # sub_dirs = [x for x in os.listdir(args.load_path) if x.startswith("pytorch_model")]
    # if len(sub_dirs) == 1:
    #     checkpoint_name = "pytorch_model.bin"
    #     state_dict = torch.load(os.path.join(args.load_path, checkpoint_name), map_location="cpu")
    # else:
    #     # if args.model_name == "mistral-7b":
    #     #     num_checkpoints = len(sub_dirs) - 1
    #     #     state_dict = merge_transformers_sharded_states_7b(args.load_path, num_checkpoints)
    #     if args.model_name == "mistral-7b":
    #         model = MistralForCausalLM.from_pretrained(args.load_path)
    #         state_dict = model.state_dict()

    # config = MistralConfig.from_pretrained(args.load_path)

    idefics2_model =  Idefics2ForConditionalGeneration.from_pretrained(
            args.load_path,
    )

    # if os.path.exists(args.config_path):
    try:
        print('loading config from', args.config_path)
        config = AutoConfig.from_pretrained(args.config_path)
    # else:
    except:
        print('loading config from', args.load_path)
        config = AutoConfig.from_pretrained(args.load_path)
    text_config = config.text_config


    # load idefics2 params to mistral
    model = MistralForCausalLM(text_config)
    model.model.load_state_dict(idefics2_model.model.text_model.state_dict(), strict=True)
    model.lm_head.load_state_dict(idefics2_model.lm_head.state_dict(), strict=True)
    state_dict = model.state_dict()

    internal_state_dict = {}
    for layer_id in range(text_config.num_hidden_layers):
        q_weight = state_dict['model.layers.'+str(layer_id)+'.self_attn.q_proj.weight']
        k_weight = state_dict['model.layers.' + str(layer_id) + '.self_attn.k_proj.weight']
        v_weight = state_dict['model.layers.' + str(layer_id) + '.self_attn.v_proj.weight']

        internal_state_dict['transformer.layers.'+str(layer_id)+'.self_attn.query.weight'] = q_weight
        internal_state_dict['transformer.layers.'+str(layer_id)+'.self_attn.key_value.weight'] = torch.cat((k_weight, v_weight))

        internal_state_dict['transformer.layers.' + str(layer_id) + '.self_attn.dense.weight'] =\
            state_dict['model.layers.' + str(layer_id) + '.self_attn.o_proj.weight']

        dense_h_to_4h_1_weight = state_dict[
            'model.layers.' + str(layer_id) + '.mlp.gate_proj.weight']

        dense_h_to_4h_2_weight = state_dict[
            'model.layers.' + str(layer_id) + '.mlp.up_proj.weight']

        internal_state_dict['transformer.layers.' + str(layer_id) + '.mlp.dense_h_to_4h_1.weight'] =\
            dense_h_to_4h_1_weight

        internal_state_dict['transformer.layers.' + str(layer_id) + '.mlp.dense_h_to_4h_2.weight'] =\
            dense_h_to_4h_2_weight

        internal_state_dict['transformer.layers.' + str(layer_id) + '.mlp.dense_4h_to_h.weight'] = state_dict[
            'model.layers.' + str(layer_id) + '.mlp.down_proj.weight']

        internal_state_dict['transformer.layers.' + str(layer_id) + '.input_layernorm.weight'] = state_dict[
            'model.layers.' + str(layer_id) + '.input_layernorm.weight']

        internal_state_dict['transformer.layers.' + str(layer_id) + '.post_attention_layernorm.weight'] = state_dict[
            'model.layers.' + str(layer_id) + '.post_attention_layernorm.weight']

    internal_state_dict["transformer.word_embeddings.weight"] = state_dict['model.embed_tokens.weight']
    internal_state_dict["transformer.final_layernorm.weight"] = state_dict['model.norm.weight']
    internal_state_dict["transformer.lm_head.weight"] = state_dict['lm_head.weight']

    # vision

    # vision_tower_path = "/apdcephfs_us/share_300814644/data/nlp/huggingface_models/idefics2-vision-model/vision_model.pth"
    # vision_tower_path = args.vision_tower_path
    vision_config = config.vision_config
    # vision_model = Idefics2VisionTransformer(vision_config)
    # vision_model.load_state_dict(torch.load(args.vision_tower))
    # vision_state_dict = torch.load(vision_tower_path)
    vision_state_dict = idefics2_model.model.vision_model.state_dict()

    for layer_id in range(vision_config.num_hidden_layers):
        # 1. self attention qkv
        q_vision_weight = vision_state_dict['encoder.layers.' + str(layer_id) + '.self_attn.q_proj.weight']
        k_vision_weight = vision_state_dict['encoder.layers.' + str(layer_id) + '.self_attn.k_proj.weight']
        v_vision_weight = vision_state_dict['encoder.layers.' + str(layer_id) + '.self_attn.v_proj.weight']
        internal_state_dict['vision_transformer.layers.' + str(layer_id) + '.self_attn.query_key_value.weight'] = torch.cat((q_vision_weight, k_vision_weight, v_vision_weight))
        q_vision_bias = vision_state_dict['encoder.layers.' + str(layer_id) + '.self_attn.q_proj.bias']
        k_vision_bias = vision_state_dict['encoder.layers.' + str(layer_id) + '.self_attn.k_proj.bias']
        v_vision_bias = vision_state_dict['encoder.layers.' + str(layer_id) + '.self_attn.v_proj.bias']
        # 2. self attention output layer
        internal_state_dict['vision_transformer.layers.' + str(layer_id) + '.self_attn.query_key_value.bias'] = torch.cat((q_vision_bias, k_vision_bias, v_vision_bias))
        internal_state_dict['vision_transformer.layers.' + str(layer_id) + '.self_attn.dense.weight'] = vision_state_dict['encoder.layers.' + str(layer_id) + '.self_attn.out_proj.weight']
        internal_state_dict['vision_transformer.layers.' + str(layer_id) + '.self_attn.dense.bias'] = vision_state_dict['encoder.layers.' + str(layer_id) + '.self_attn.out_proj.bias']
        # 3. layer norm 1
        internal_state_dict['vision_transformer.layers.' + str(layer_id) + '.self_attn.layer_norm.weight'] = vision_state_dict['encoder.layers.' + str(layer_id) + '.layer_norm1.weight']
        internal_state_dict['vision_transformer.layers.' + str(layer_id) + '.self_attn.layer_norm.bias'] = vision_state_dict['encoder.layers.' + str(layer_id) + '.layer_norm1.bias']
        # 4. two fc layers
        internal_state_dict['vision_transformer.layers.' + str(layer_id) + '.mlp.dense_h_to_4h.weight'] = vision_state_dict['encoder.layers.' + str(layer_id) + '.mlp.fc1.weight']
        internal_state_dict['vision_transformer.layers.' + str(layer_id) + '.mlp.dense_h_to_4h.bias'] = vision_state_dict['encoder.layers.' + str(layer_id) + '.mlp.fc1.bias']
        internal_state_dict['vision_transformer.layers.' + str(layer_id) + '.mlp.dense_4h_to_h.weight'] = vision_state_dict['encoder.layers.' + str(layer_id) + '.mlp.fc2.weight']
        internal_state_dict['vision_transformer.layers.' + str(layer_id) + '.mlp.dense_4h_to_h.bias'] = vision_state_dict['encoder.layers.' + str(layer_id) + '.mlp.fc2.bias']
        # 5. layer norm 2
        internal_state_dict['vision_transformer.layers.' + str(layer_id) + '.mlp.layer_norm.weight'] = vision_state_dict['encoder.layers.' + str(layer_id) + '.layer_norm2.weight']
        internal_state_dict['vision_transformer.layers.' + str(layer_id) + '.mlp.layer_norm.bias'] = vision_state_dict['encoder.layers.' + str(layer_id) + '.layer_norm2.bias']

    # embedding
    internal_state_dict['vision_transformer.patch_embedding.weight'] = vision_state_dict['embeddings.patch_embedding.weight']
    internal_state_dict['vision_transformer.patch_embedding.bias'] = vision_state_dict['embeddings.patch_embedding.bias']
    internal_state_dict['vision_transformer.position_embedding.weight'] = vision_state_dict['embeddings.position_embedding.weight']
    internal_state_dict['vision_transformer.post_layernorm.weight'] = vision_state_dict['post_layernorm.weight']
    internal_state_dict['vision_transformer.post_layernorm.bias'] = vision_state_dict['post_layernorm.bias']

    # Idefics2Perceiver

    # perceiver_config = Idefics2Config.from_pretrained(args.load_path).perceiver_config
    perceiver_config = config.perceiver_config
    # perceiver_state_dict = torch.load(args.connector_path)
    perceiver_state_dict = idefics2_model.model.connector.state_dict()

    # 0. latent: 64 x hidden_size nn.Params

    num_attn_head_q = perceiver_config.resampler_n_heads # 16
    num_attn_head_kv = perceiver_config.num_key_value_heads # 4
    num_repeat = num_attn_head_q//num_attn_head_kv # 4

    for layer_id in range(perceiver_config.resampler_depth):
        # 1. input_latents_norm
        internal_state_dict['perceiver_transformer.layers.' + str(layer_id) + '.input_latents_norm.weight'] = perceiver_state_dict['perceiver_resampler.layers.' + str(layer_id) + '.input_latents_norm.weight']
        # 2. input_context_norm
        internal_state_dict['perceiver_transformer.layers.' + str(layer_id) + '.input_context_norm.weight'] = perceiver_state_dict['perceiver_resampler.layers.' + str(layer_id) + '.input_context_norm.weight']
        # 3. self attention qkv
        q_connector_weight = perceiver_state_dict['perceiver_resampler.layers.' + str(layer_id) + '.self_attn.q_proj.weight']
        k_connector_weight = perceiver_state_dict['perceiver_resampler.layers.' + str(layer_id) + '.self_attn.k_proj.weight']
        v_connector_weight = perceiver_state_dict['perceiver_resampler.layers.' + str(layer_id) + '.self_attn.v_proj.weight']
        
        internal_state_dict['perceiver_transformer.layers.'+str(layer_id)+'.self_attn.query.weight'] = q_connector_weight

        #### FIXME:
        # FTQ: repeat k and v 4 times to comply with the tp size. Hard coded here.
        # internal_state_dict['perceiver_transformer.layers.'+str(layer_id)+'.self_attn.key_value.weight'] = \
        #     torch.cat((torch.cat([k_connector_weight.clone() for i in range(num_repeat)], dim=0),
        #                torch.cat([v_connector_weight.clone() for i in range(num_repeat)], dim=0) ))
        # internal_state_dict['perceiver_transformer.layers.'+str(layer_id)+'.self_attn.key_value.weight'] = \
        #     torch.cat((k_connector_weight[None, :, :].expand(num_repeat, 
        #                                                     k_connector_weight.shape[0],  
        #                                                      k_connector_weight.shape[1]).reshape(
        #                                                          k_connector_weight.shape[0] * num_repeat, 
        #                                                          k_connector_weight.shape[1] 
        #                                                     ),
        #                v_connector_weight[None, :,  :].expand(num_repeat, 
        #                                                     v_connector_weight.shape[0],  
        #                                                      v_connector_weight.shape[1]).reshape(
        #                                                          v_connector_weight.shape[0] * num_repeat, 
        #                                                          v_connector_weight.shape[1] 
        #                                                     ) ))
        

        k_connector_weight = k_connector_weight.view(num_attn_head_kv, 
                                k_connector_weight.shape[0]//num_attn_head_kv, 
                                k_connector_weight.shape[1]
                            )[:, None, :, :].expand(
                                    num_attn_head_kv,
                                    num_repeat,
                                    k_connector_weight.shape[0]//num_attn_head_kv, 
                                    k_connector_weight.shape[1]
                                ).reshape(-1, k_connector_weight.shape[1])
        v_connector_weight = v_connector_weight.view(num_attn_head_kv, 
                                v_connector_weight.shape[0]//num_attn_head_kv, 
                                v_connector_weight.shape[1]
                            )[:, None, :, :].expand(
                                    num_attn_head_kv,
                                    num_repeat,
                                    v_connector_weight.shape[0]//num_attn_head_kv, 
                                    v_connector_weight.shape[1]
                                ).reshape(-1, v_connector_weight.shape[1])
        

        internal_state_dict['perceiver_transformer.layers.'+str(layer_id)+'.self_attn.key_value.weight'] = \
            torch.cat((k_connector_weight,  v_connector_weight))
        

        # 4, 96, 4096



        internal_state_dict['perceiver_transformer.layers.' + str(layer_id) + '.self_attn.dense.weight'] =\
            perceiver_state_dict['perceiver_resampler.layers.' + str(layer_id) + '.self_attn.o_proj.weight']
        
        # 4. post attention layer norm
        internal_state_dict['perceiver_transformer.layers.'+str(layer_id)+'.post_attention_layernorm.weight'] = perceiver_state_dict['perceiver_resampler.layers.' + str(layer_id) + '.post_attention_layernorm.weight']
        
        # 4. two fc layers
        # internal_state_dict['perceiver_transformer.layers.' + str(layer_id) + '.mlp.gate_proj.weight'] = perceiver_state_dict['perceiver_resampler.layers.' + str(layer_id) + '.mlp.gate_proj.weight']
        # internal_state_dict['perceiver_transformer.layers.' + str(layer_id) + '.mlp.up_proj.weight'] = perceiver_state_dict['perceiver_resampler.layers.' + str(layer_id) + '.mlp.up_proj.weight']
        # internal_state_dict['perceiver_transformer.layers.' + str(layer_id) + '.mlp.down_proj.weight'] = perceiver_state_dict['perceiver_resampler.layers.' + str(layer_id) + '.mlp.down_proj.weight']
        dense_h_to_4h_1_weight = perceiver_state_dict[
            'perceiver_resampler.layers.' + str(layer_id) + '.mlp.gate_proj.weight']

        dense_h_to_4h_2_weight = perceiver_state_dict[
            'perceiver_resampler.layers.' + str(layer_id) + '.mlp.up_proj.weight']

        internal_state_dict['perceiver_transformer.layers.' + str(layer_id) + '.mlp.dense_h_to_4h_1.weight'] =\
            dense_h_to_4h_1_weight

        internal_state_dict['perceiver_transformer.layers.' + str(layer_id) + '.mlp.dense_h_to_4h_2.weight'] =\
            dense_h_to_4h_2_weight

        internal_state_dict['perceiver_transformer.layers.' + str(layer_id) + '.mlp.dense_4h_to_h.weight'] = perceiver_state_dict[
            'perceiver_resampler.layers.' + str(layer_id) + '.mlp.down_proj.weight']
        
    internal_state_dict['perceiver_transformer.final_layernorm.weight'] = perceiver_state_dict['perceiver_resampler.norm.weight']
    internal_state_dict['perceiver_transformer.latents'] = perceiver_state_dict['perceiver_resampler.latents']

    state_dict = internal_state_dict

    # Saving config and tokenzier files
    os.system("cp -rf "+args.load_path+"/*.json "+args.save_path)
    os.system("cp -rf " + args.load_path + "/tokeniz* " + args.save_path)

    # Saving the tracker file
    tracker_filepath = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    # create `release` dir in args.load_path
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)

    # megatron args
    megatron_args = {
        "orig_vocab_size": text_config.vocab_size,
        "hidden_size": text_config.hidden_size,
        "num_layers": text_config.num_hidden_layers,
        "num_attention_heads": text_config.num_attention_heads,
        "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
        "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size,
        "data_parallel_size": args.target_data_parallel_size,
        "make_vocab_size_divisible_by": args.make_vocab_size_divisible_by,
        "rank": 0,
        "tokenizer_type": "GPT2BPETokenizer",
    }

    margs = types.SimpleNamespace()
    for k, v in megatron_args.items():
        setattr(margs, k, v)

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    setattr(margs, "params_dtype", dtype)

    # Convert.
    print("Converting")
    output_state_dict = []
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append({})

    num_query_group = 8
    output_group_state_dict = []
    for i in range(num_query_group):
        output_group_state_dict.append({})

    # Embedding layer
    print("converting embedding layer")
    word_embedding = state_dict["transformer.word_embeddings.weight"].to(dtype)
    lm_head = state_dict["transformer.lm_head.weight"].to(dtype)
    orig_vocab_size = text_config.vocab_size
    #padded_vocab_size = _vocab_size_with_padding(orig_vocab_size, margs)
    padded_vocab_size = orig_vocab_size
    setattr(margs, "padded_vocab_size", padded_vocab_size)
    # Cut out extra padding we don't need
    if args.extra_num_vocabs == 0:
        full_word_embed = word_embedding
        full_lm_head = lm_head
    else:
        new_embeddings = torch.nn.Embedding(args.extra_num_vocabs, word_embedding.shape[1])
        # initialize all new embeddings (in particular added tokens)
        _init_embedding_weights(new_embeddings)
        full_word_embed = torch.cat([word_embedding, new_embeddings.weight])
        full_lm_head = torch.cat([lm_head, new_embeddings.weight])

    # Split into new tensor model parallel sizes
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_model_parallel_size, dim=0)

    # 
    mlp_gate_proj = torch.chunk(perceiver_state_dict['modality_projection.gate_proj.weight'], args.target_tensor_model_parallel_size, dim=0)
    mlp_up_proj = torch.chunk(perceiver_state_dict['modality_projection.up_proj.weight'], args.target_tensor_model_parallel_size, dim=0)
    mlp_down_proj = torch.chunk(perceiver_state_dict['modality_projection.down_proj.weight'], args.target_tensor_model_parallel_size, dim=1)


    for i in range(args.target_tensor_model_parallel_size):
        word_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.word_embeddings"
        )
        word_emb_dict["weight"] = out_word_embed[i]

        # vision
        # actually, these are not being chunked. Just copy them to each layer.
        vision_enc_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.vision_model")
        vision_enc_dict["conv1.weight"] = state_dict['vision_transformer.patch_embedding.weight']
        vision_enc_dict["conv1.bias"] = state_dict['vision_transformer.patch_embedding.bias']
        vision_enc_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.vision_model")
        vision_enc_dict["position_embeddings.weight"] = state_dict['vision_transformer.position_embedding.weight']
        vision_enc_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.vision_model")
        vision_enc_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.vision_model")
        vision_enc_dict["ln_post.weight"] = state_dict['vision_transformer.post_layernorm.weight']
        vision_enc_dict["ln_post.bias"] = state_dict['vision_transformer.post_layernorm.bias']

        # perceiver.
        perceiver_enc_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.perceiver")
        perceiver_enc_dict['modality_projection.dense_h_to_4h.weight'] = torch.cat(
                    [mlp_gate_proj[i].clone().to(dtype), mlp_up_proj[i].clone().to(dtype)], dim=0)
        perceiver_enc_dict['modality_projection.dense_4h_to_h.weight'] = mlp_down_proj[i].clone().to(dtype)
        

    out_lm_head = torch.chunk(full_lm_head, args.target_tensor_model_parallel_size, dim=0)
    for i in range(args.target_tensor_model_parallel_size):
        lm_head_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.lm_head"
        )
        lm_head_dict["weight"] = out_lm_head[i]

    # Transformer layers
    print("converting transformer layers")
    if text_config.num_hidden_layers % args.target_pipeline_model_parallel_size != 0:
        raise ValueError(
            f"Number of layers ({text_config.num_hidden_layers}) must be divisible by number of pipeline parallelism"
            f" ({args.target_pipeline_model_parallel_size})"
        )
    num_layers = text_config.num_hidden_layers // args.target_pipeline_model_parallel_size

    # KM: it could be tricky to do pipeline parallel, as the vision encoder should be in the first stage of the pipeline

    layer_re = re.compile("transformer.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    vision_layer_re = re.compile("vision_transformer.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    perceiver_layer_re = re.compile("perceiver_transformer.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    # The number of heads.
    heads = text_config.num_attention_heads
    # The hidden_size per head.
    hidden_size = 4096
    num_groups = 8
    head_dim = 128
    num_heads = 32
    vision_heads = vision_config.num_attention_heads # 16
    resampler_heads = perceiver_config.resampler_n_heads # 16
    kv_heads = perceiver_config.num_key_value_heads # 4
    
    #  hidden size per head
    hidden_size_per_head = text_config.hidden_size // text_config.num_attention_heads
    vision_hidden_size_per_head = vision_config.hidden_size // vision_config.num_attention_heads
    resampler_hidden_size_per_head = perceiver_config.resampler_head_dim

    for pp_rank in range(args.target_pipeline_model_parallel_size):
        layer_offset = pp_rank * num_layers
        if pp_rank > 0:
            output_state_dict = []
            for i in range(args.target_tensor_model_parallel_size):
                output_state_dict.append({})

            output_group_state_dict = []
            for i in range(num_query_group):
                output_group_state_dict.append({})

        for layer in range(num_layers):
            pp_layer_id = layer + layer_offset
            layers_to_copy = [
                layer_name
                for layer_name in state_dict.keys()
                if layer_name.startswith(f"transformer.layers.{pp_layer_id}.")
            ]
            ## Perceiver
            if pp_layer_id < perceiver_config.resampler_depth:
                assert pp_rank == 0
                perceiver_layers_to_copy = [
                    layer_name
                    for layer_name in state_dict.keys()
                    if layer_name.startswith(f"perceiver_transformer.layers.{pp_layer_id}.")
                ]
                for layer_name in perceiver_layers_to_copy:
                    m = perceiver_layer_re.match(layer_name)
                    # Stop if that's not a layer
                    if m is None:
                        break

                    # The index of the layer.
                    _ = int(m.group(1))
                    # The name of the operation.
                    op_name = m.group(2)
                    # Is it a weight or a bias?
                    weight = m.group(3)

                    params = state_dict[layer_name].to(dtype)

                    if op_name.startswith("self_attn.query") and weight == "weight":
                        # transformers stores D X (3*D) but Megatron-LM expects (3*D) X D.
                        params = transformers_to_megatron_fix_query_key_value_ordering(
                            params,
                            3.0,
                            1,
                            resampler_heads,
                            resampler_hidden_size_per_head,
                        )
                        layer_name = f"layers.{layer}.self_attention.query.{weight}"
                        # (1536, 4096)
                    elif op_name.startswith("self_attn.key_value") and weight == "weight" :
                        # transformers stores D X (3*D) but Megatron-LM expects (3*D) X D.
                        params = transformers_to_megatron_fix_query_key_value_ordering(
                            params,
                            3.0,
                            2,
                            resampler_heads,
                            resampler_hidden_size_per_head,
                        )
                        # WARNING: num attention head is resampler_heads instead of kv_heads!!
                        layer_name = f"layers.{layer}.self_attention.key_value.{weight}"

                    # elif op_name.startswith("self_attn.dense"):
                    #     layer_name = f"layers.{layer}.self_attention.linear_proj.{weight}"
                    # elif op_name.startswith("self_attn.layer_norm"):
                    #     if weight == "weight":
                    #         layer_name = f"layers.{layer}.self_attention.linear_qkv.layer_norm_weight"
                    #     else:
                    #         layer_name = f"layers.{layer}.self_attention.linear_qkv.layer_norm_bias"
                    elif op_name.startswith("input_latents_norm") or op_name.startswith("input_context_norm") or op_name.startswith("post_attention_layernorm"):
                        if op_name.startswith("input_latents_norm"):
                            out_name = "input_latents_norm"
                        elif op_name.startswith("input_context_norm"):
                            out_name = "input_context_norm"
                        elif op_name.startswith("post_attention_layernorm"):
                            out_name = "post_attention_layernorm"

                        layer_name = f"layers.{layer}.{out_name}.{weight}"
                        # pdb.set_trace()


                    elif weight == "weight":
                        out_name = transformers_to_megatron.get(op_name, None)
                        if out_name is None:
                            continue
                        #params = params.transpose(0, 1)
                        layer_name = f"layers.{layer}.{out_name}.{weight}"

                    else:
                        print ("this should not happen!!!")
                        exit()

                    if op_name + "." + weight in tensor_parallel_params:
                        dim = 1 if op_name in ["self_attn.dense", "mlp.dense_4h_to_h"] else 0
                        params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=dim)

                    for i in range(args.target_tensor_model_parallel_size):
                        params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.perceiver")
                        params_dict[layer_name] = (
                            params[i].clone() if (op_name + "." + weight in tensor_parallel_params) else params.clone()
                        )
                # pdb.set_trace()
            ## ViT
            if pp_layer_id < vision_config.num_hidden_layers:
                ### FIXME: 
                ### ftq: check for layer_name in state_dict.keys() later!!
                assert pp_rank == 0
                vision_layers_to_copy = [
                    layer_name
                    for layer_name in state_dict.keys()
                    if layer_name.startswith(f"vision_transformer.layers.{pp_layer_id}.")
                ]
                for layer_name in vision_layers_to_copy:
                    m = vision_layer_re.match(layer_name)
                    # Stop if that's not a layer
                    if m is None:
                        break

                    # The index of the layer.
                    _ = int(m.group(1))
                    # The name of the operation.
                    op_name = m.group(2)
                    # Is it a weight or a bias?
                    weight = m.group(3)

                    params = state_dict[layer_name].to(dtype)
                    if op_name.startswith("self_attn.query_key_value"):
                        # transformers stores D X (3*D) but Megatron-LM expects (3*D) X D.
                        params = transformers_to_megatron_fix_query_key_value_ordering(
                            params,
                            3.0,
                            3,
                            vision_heads,
                            vision_hidden_size_per_head,
                        )
                        layer_name = f"layers.{layer}.self_attention.linear_qkv.{weight}"
                    elif op_name.startswith("self_attn.dense"):
                        layer_name = f"layers.{layer}.self_attention.linear_proj.{weight}"
                    elif op_name.startswith("self_attn.layer_norm"):
                        if weight == "weight":
                            layer_name = f"layers.{layer}.self_attention.linear_qkv.layer_norm_weight"
                        else:
                            layer_name = f"layers.{layer}.self_attention.linear_qkv.layer_norm_bias"
                    elif op_name.startswith("mlp.dense_h_to_4h"):
                        layer_name = f"layers.{layer}.mlp.linear_fc1.{weight}"
                    elif op_name.startswith("mlp.dense_4h_to_h"):
                        layer_name = f"layers.{layer}.mlp.linear_fc2.{weight}"
                    elif op_name.startswith("mlp.layer_norm"):
                        if weight == "weight":
                            layer_name = f"layers.{layer}.mlp.linear_fc1.layer_norm_weight"
                        else:
                            layer_name = f"layers.{layer}.mlp.linear_fc1.layer_norm_bias"
                    else:
                        print ("this should not happen!!!")
                        exit()

                    if op_name + "." + weight in tensor_parallel_params:
                        dim = 1 if op_name in ["self_attn.dense", "mlp.dense_4h_to_h"] else 0
                        params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=dim)

                    for i in range(args.target_tensor_model_parallel_size):
                        params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.vision_model")
                        params_dict['transformer.'+layer_name] = (
                            params[i].clone() if (op_name + "." + weight in tensor_parallel_params) else params.clone()
                        )

            # Mistral

            for layer_name in layers_to_copy:
                m = layer_re.match(layer_name)
                # Stop if that's not a layer
                if m is None:
                    break

                # The index of the layer.
                _ = int(m.group(1))
                # The name of the operation.
                op_name = m.group(2)
                # Is it a weight or a bias?
                weight = m.group(3)

                params = state_dict[layer_name].to(dtype)
                # handle layernorm
                if op_name.startswith("input_layernorm") or op_name.startswith("post_attention_layernorm"):
                    out_name = "input_layernorm" if op_name.endswith("input_layernorm") else "post_attention_layernorm"
                    layer_name = f"layers.{layer}.{out_name}.{weight}"

                elif op_name.startswith("self_attn.rotary_emb"):
                    layer_name = f"layers.{layer}.self_attention.rotary_emb.inv_freq"

                # handle attention K, V, Q weights
                elif op_name.startswith("self_attn.query") and weight == "weight":
                    # transformers stores D X (3*D) but Megatron-LM expects (3*D) X D.
                    params = transformers_to_megatron_fix_query_key_value_ordering(
                        params,
                        3.0,
                        1,
                        heads,
                        hidden_size_per_head,
                    )
                    layer_name = f"layers.{layer}.self_attention.query.{weight}"

                elif op_name.startswith("self_attn.key_value") and weight == "weight":
                    # transformers stores D X (3*D) but Megatron-LM expects (3*D) X D.
                    params = transformers_to_megatron_fix_query_key_value_ordering(
                        params,
                        3.0,
                        2,
                        8,
                        hidden_size_per_head,
                    )
                    layer_name = f"layers.{layer}.self_attention.key_value.{weight}"

                # handle attention and mlp weights
                elif weight == "weight":
                    out_name = transformers_to_megatron.get(op_name, None)
                    if out_name is None:
                        continue
                    #params = params.transpose(0, 1)
                    layer_name = f"layers.{layer}.{out_name}.{weight}"

                # skip
                else:
                    continue

                if op_name + "." + weight in tensor_parallel_params:
                    dim = 1 if op_name in ["self_attn.dense", "mlp.dense_4h_to_h"] else 0
                    params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=dim)

                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = (
                        params[i].clone() if (op_name + "." + weight in tensor_parallel_params) else params.clone()
                    )

            if pp_layer_id < perceiver_config.resampler_depth:
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i],
                                                                "model.language_model.perceiver")

                    dense_h_to_4h_1_name = 'mlp.dense_h_to_4h_1.weight'
                    dense_h_to_4h_1_layer_name = f"layers.{layer}.{dense_h_to_4h_1_name}"
                    dense_h_to_4h_1_weight = params_dict[dense_h_to_4h_1_layer_name]

                    dense_h_to_4h_2_name = 'mlp.dense_h_to_4h_2.weight'
                    dense_h_to_4h_2_layer_name = f"layers.{layer}.{dense_h_to_4h_2_name}"
                    dense_h_to_4h_2_weight = params_dict[dense_h_to_4h_2_layer_name]

                    dense_h_to_4h_name = 'mlp.dense_h_to_4h.weight'
                    dense_h_to_4h_layer_name = f"layers.{layer}.{dense_h_to_4h_name}"

                    params_dict[dense_h_to_4h_layer_name] = torch.cat(
                    [dense_h_to_4h_1_weight, dense_h_to_4h_2_weight], dim=0)

                    del params_dict[dense_h_to_4h_1_layer_name]
                    del params_dict[dense_h_to_4h_2_layer_name]

            for i in range(args.target_tensor_model_parallel_size):

                params_dict = get_element_from_dict_by_path(output_state_dict[i],
                                                            "model.language_model.encoder")

                dense_h_to_4h_1_name = 'mlp.dense_h_to_4h_1.weight'
                dense_h_to_4h_1_layer_name = f"layers.{layer}.{dense_h_to_4h_1_name}"
                dense_h_to_4h_1_weight = params_dict[dense_h_to_4h_1_layer_name]

                dense_h_to_4h_2_name = 'mlp.dense_h_to_4h_2.weight'
                dense_h_to_4h_2_layer_name = f"layers.{layer}.{dense_h_to_4h_2_name}"
                dense_h_to_4h_2_weight = params_dict[dense_h_to_4h_2_layer_name]

                dense_h_to_4h_name = 'mlp.dense_h_to_4h.weight'
                dense_h_to_4h_layer_name = f"layers.{layer}.{dense_h_to_4h_name}"

                params_dict[dense_h_to_4h_layer_name] = torch.cat(
                [dense_h_to_4h_1_weight, dense_h_to_4h_2_weight], dim=0)

                del params_dict[dense_h_to_4h_1_layer_name]
                del params_dict[dense_h_to_4h_2_layer_name]

                query_name = 'self_attention.query.weight'
                query_layer_name = f"layers.{layer}.{query_name}"
                query_weight = params_dict[query_layer_name]

                kv_name = 'self_attention.key_value.weight'
                kv_layer_name = f"layers.{layer}.{kv_name}"
                kv_weight = params_dict[kv_layer_name]

                qkv_name = 'self_attention.query_key_value.weight'
                qkv_layer_name = f"layers.{layer}.{qkv_name}"

                # torch.Size([32 128, 4096])
                # (1, 512, 4096)
                group_query_weight = query_weight.view(num_groups // args.target_tensor_model_parallel_size, num_heads // num_groups * head_dim, hidden_size)
                # torch.Size(8, 256, 4096])
                group_kv_weight = kv_weight.view(num_groups // args.target_tensor_model_parallel_size, 2 * head_dim, hidden_size)

                group_qkv_weight = torch.cat([group_query_weight, group_kv_weight], dim=1)
                params_dict[qkv_layer_name] = group_qkv_weight.view(-1, hidden_size)

                del params_dict[query_layer_name]
                del params_dict[kv_layer_name]

        if pp_rank == args.target_pipeline_model_parallel_size - 1:
            # handle final layernorm
            for weight_or_bias in ["weight"]:
                params = state_dict[f"transformer.final_layernorm.{weight_or_bias}"].to(dtype)
                layer_name = f"final_layernorm.{weight_or_bias}"
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = params.clone()
            # handle final layernorm of perceiver:
            for weight_or_bias in ["weight"]:
                params = state_dict[f"perceiver_transformer.final_layernorm.{weight_or_bias}"].to(dtype)
                layer_name = f"final_layernorm.{weight_or_bias}"
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.perceiver")
                    params_dict[layer_name] = params.clone()
            # perceiver latents
            params = state_dict[f"perceiver_transformer.latents"].to(dtype)
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.perceiver")
                params_dict["latents"] = params.clone()
            # add the LM head
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.word_embeddings_for_head")
                params_dict["weight"] = out_word_embed[i].clone()

            # add the LM head
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.output_layer")
                params_dict["weight"] = out_lm_head[i].clone()

        # saving the state dict as per the tp_rank and pp_rank
        for tp_rank in range(args.target_tensor_model_parallel_size):
            output_state_dict[tp_rank]["checkpoint_version"] = 3.0
            output_state_dict[tp_rank]["args"] = margs
            checkpoint_dir = (
                f"mp_rank_{tp_rank:02d}"
                if args.target_pipeline_model_parallel_size == 1
                else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
            )

            checkpoint_name = "model_optim_rng.pt"
            checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            if args.print_checkpoint_structure:
                print(
                    f"Checkpoint structure of model state dict shard belonging to TP rank {tp_rank} and PP rank"
                    f" {pp_rank}:"
                )
                recursive_print(None, output_state_dict[tp_rank])
            torch.save(output_state_dict[tp_rank], checkpoint_path)


def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality. This greatly extends the functionality of
    `convert_megatron_gpt2_checkpoint.py`

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Load Megatron-LM checkpoint arguments from the state dict
    os.makedirs(args.save_path, exist_ok=True)
    sub_dirs = os.listdir(os.path.join(args.load_path))
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_checkpoint_name = [i for i in os.listdir(os.path.join(args.load_path, sub_dir)) if 'rng' in i][0]
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break
    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # Saving config and tokenzier files
    config_path = '/'.join(args.load_path.split('/')[:-1])
    os.system("cp -rf "+config_path+"/*.json " + args.save_path)
    os.system("cp -rf " + config_path + "/tokenizer.model " + args.save_path)
    import glob
    # if glob.glob(args.load_path+"/mp_rank*/distrib*"):
    # # if os.path.exists(args.load_path+"/mp_rank*/distrib*"):
    #     user_input = input("Optimizer states detected. Will remove distrib* files. yes (remove and continue) / no (stop programme): ")
    #     if user_input == 'yes':
    #         os.system("rm -rf "+args.load_path+"/mp_rank*/distrib*")
    #     else:
    #         raise RuntimeError("Optimizer states are not removed. Save files to another folder and re-run.")

    activation_function = "gelu"

    vocab_size = (
        megatron_args.padded_vocab_size
        if getattr(megatron_args, "orig_vocab_size", None) is None
        else megatron_args.orig_vocab_size
    )

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    intermediate_size_map = {4096:14336}
    config_llm = MistralConfig(
        vocab_size=vocab_size,
        hidden_size=megatron_args.hidden_size,
        num_hidden_layers=megatron_args.num_layers,
        num_attention_heads=megatron_args.num_attention_heads,
        num_key_value_heads=8,
        max_position_embeddings=32768,
        rms_norm_eps=1e-05,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        architectures=["MistralForCausalLM"],
        torch_dtype=dtype,
        max_sequence_length=2048,
        hidden_act="silu",
        intermediate_size=intermediate_size_map[megatron_args.hidden_size],
    )

    # config = AutoConfig.from_pretrained(config_path) # idefics2 config
    if os.path.exists(args.config_path):
    # try:
        print('loading config from', args.config_path)
        config = AutoConfig.from_pretrained(args.config_path)
    else:
    # except:
        print('loading config from', config_path)
        config = AutoConfig.from_pretrained(config_path)
    config.perceiver_config.num_key_value_heads *= 4 # FIXME: hard coded here.
    vision_config = config.vision_config
    perceiver_config = config.perceiver_config
    config.perceiver_config.num_key_value_heads = 16 # hard coded
    config.save_pretrained(args.save_path)

    output_state_dict = {}

    checkpoint_version = state_dict.get("checkpoint_version", 3.0)
    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Convert.
    print("Converting")

    # Embeddings
    print("Converting embeddings")
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)

    # Convert and store the position embeddings.
    position_embeddings = get_element_from_dict_by_path(
        tp_state_dicts[0], "model.language_model.embedding.position_embeddings.weight"
    )

    if position_embeddings:
        assert False # no position embedding
        output_state_dict["transformer.position_embeddings.weight"] = position_embeddings.to(dtype).clone()

    # Convert and store the word embeddings.
    word_embeddings = []
    word_embeddings_layernorm_weight = []
    word_embeddings_layernorm_bias = []

    # import pdb
    # pdb.set_trace()

    modality_projection_dense_h_to_4h = []
    modality_projection_dense_4h_to_h = []

    for tp_rank in range(tp_size):
        embeddings = get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.word_embeddings_for_head.weight"
            )
        # After training with megatron, word_embeddings is stored differently
        if type(embeddings) is dict:
            embeddings = get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.language_model.embedding.word_embeddings.weight"
            )
        word_embeddings.append(embeddings)
        dense_h_to_4h = tp_state_dicts[tp_rank]['model']['language_model']['perceiver']['modality_projection.dense_h_to_4h.weight']
        dense_4h_to_h = tp_state_dicts[tp_rank]['model']['language_model']['perceiver']['modality_projection.dense_4h_to_h.weight']
        modality_projection_dense_h_to_4h.append(dense_h_to_4h)
        modality_projection_dense_4h_to_h.append(dense_4h_to_h)

    word_embeddings = torch.cat(word_embeddings, dim=0)
    word_embeddings = word_embeddings.to(dtype)
    word_embeddings = word_embeddings[:-args.extra_num_vocabs, :]
    output_state_dict["model.text_model.embed_tokens.weight"] = word_embeddings.clone()

    modality_projection_dense_h_to_4h = torch.cat(modality_projection_dense_h_to_4h, dim=0)
    modality_projection_dense_4h_to_h = torch.cat(modality_projection_dense_4h_to_h, dim=1)
    # pdb.set_trace()
    # output_state_dict['model.connector.modality_projection.gate_proj.weight'], \
    #     output_state_dict['model.connector.modality_projection.up_proj.weight'] = \
    #     torch.split(modality_projection_dense_h_to_4h, modality_projection_dense_h_to_4h.size(0)//2, dim=0)
    params = modality_projection_dense_h_to_4h
    out_name = megatron_to_transformers['dense_h_to_4h']
    para_dict = {i:[] for i in out_name}
    for index, mat in enumerate(torch.split(params, params.shape[0]//tp_size)):
        for index_new, mat_sep in enumerate(torch.split(mat, mat.shape[0]//2)):
            para_dict[out_name[index_new]].append(mat_sep.clone())
    for name, para in para_dict.items():
        output_state_dict['model.connector.modality_projection' + name + "weight"] = torch.cat(para)

    output_state_dict['model.connector.modality_projection.down_proj.weight'] = modality_projection_dense_4h_to_h
    
    output_state_dict['model.connector.perceiver_resampler.latents'] = tp_state_dicts[0]['model']['language_model']['perceiver']['latents'].clone()

    # Reset the vocab size
    config.vocab_size = word_embeddings.shape[0]

    # Transformer Layers
    print("Converting transformer layers")
    # The number of heads.
    heads = config.text_config.num_attention_heads
    # The hidden_size per head.
    hidden_size_per_head = config.text_config.hidden_size // config.text_config.num_attention_heads
    num_layers = config.text_config.num_hidden_layers // pp_size

    for pp_rank in range(pp_size):
        if pp_size > 0:
            print(f"Converting pipeline parallel rank {pp_rank}")
            tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, pp_rank)

        # The transformer.

        path = (
            "model.language_model.transformer"
            if "transformer" in get_element_from_dict_by_path(tp_state_dicts[0], "model.language_model").keys()
            else "model.language_model.encoder"
        )
        print ('path', path)
        #print (tp_state_dicts[0]['model']['language_model']['encoder'].keys())
        # Extract the layers.
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
            if key.endswith('_extra_state'):
                continue
            # Match the name.
            m = layer_re.match(key)
            # Stop if that's not a layer
            if m is None:
                break

            # The index of the layer.
            layer_idx = int(m.group(1)) + pp_rank * num_layers
            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)

            # The name of the layer.
            layer_name = f"model.text_model.layers.{layer_idx}"
            print(layer_name, op_name, weight_or_bias)
            # import pdb
            # pdb.set_trace()
            if op_name + "." + weight_or_bias not in tensor_parallel_params_mg:
                params = val.to(dtype)
            else:
                # import pdb
                # pdb.set_trace()
                dim = 1 if op_name in ["self_attention.dense", "mlp.dense_4h_to_h"] else 0
                params = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("layernorm") and weight_or_bias == 'weight':
                ln_name = "input_layernorm" if op_name.startswith("input") else "post_attention_layernorm"
                output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = params.clone()

            # Transpose the bias.
            # Not applicable
            elif (
                op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
            ) and weight_or_bias == "bias":
                out_val = megatron_to_transformers_fix_query_key_value_ordering(
                    params, checkpoint_version, 3, heads, hidden_size_per_head
                )

                # Store. No change of shape.
                output_state_dict[layer_name + ".attn.c_attn.bias"] = out_val.clone()

            # Transpose the Q matrix for query for Llama70b.
            elif (
                op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
            ) and weight_or_bias == "weight":
                hidden_size = config.text_config.hidden_size 
                num_groups = 8
                head_dim = config.text_config.hidden_size // config.text_config.num_attention_heads
                num_heads = config.text_config.num_attention_heads

                tp_states = torch.chunk(params, args.target_tensor_model_parallel_size, dim=0)
                query_dim = num_heads // num_groups * head_dim
                kv_dim = 2 * head_dim
                tp_states = [i.view(num_groups//args.target_tensor_model_parallel_size, query_dim + kv_dim, hidden_size) for i in tp_states]
                # tp_dim = hidden_size // args.target_tensor_model_parallel_size
                query = torch.cat([i[:, :query_dim].reshape(-1, hidden_size) for i in tp_states])
                key_value = torch.cat([i[:, query_dim:].reshape(-1, hidden_size) for i in tp_states])

                out_val = megatron_to_transformers_fix_query_key_value_ordering(
                    query,
                    checkpoint_version,
                    1,
                    heads,
                    hidden_size_per_head,
                )
                output_state_dict[layer_name + f".self_attn.q_proj.weight"] = out_val.clone()
                out_val = megatron_to_transformers_fix_query_key_value_ordering(
                    key_value,
                    checkpoint_version,
                    2,
                    8,
                    hidden_size_per_head,
                )
                KV = {0:'k_proj',1:'v_proj'}
                for index, matrix in enumerate(torch.split(out_val, out_val.shape[0]//2, 0)):
                    output_state_dict[layer_name + f".self_attn.{KV[index]}.weight"] = matrix.clone()

            # Transpose the weights.
            elif weight_or_bias == "weight":
                if 'dense_h_to_4h' in op_name:
                    out_name = megatron_to_transformers[op_name]
                    para_dict = {i:[] for i in out_name}
                    for index, mat in enumerate(torch.split(params, params.shape[0]//tp_size)):
                        for index_new, mat_sep in enumerate(torch.split(mat, mat.shape[0]//2)):
                            para_dict[out_name[index_new]].append(mat_sep.clone())
                    for name, para in para_dict.items():
                        output_state_dict[layer_name + name + "weight"] = torch.cat(para)
                else:
                    out_name = megatron_to_transformers[op_name]
                    output_state_dict[layer_name + out_name + "weight"] = params.clone()

            # Copy the bias.
            # Ignore them
            elif weight_or_bias == "bias":
                pass

            # Copy the Rotary Embedding
            else:
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + out_name] = params.clone()
        
        print("converting ViT")
        # vision_model
        vision_layer_re = re.compile("vision_transformer.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], 'model.language_model.vision_model').items():
            if key.endswith('_extra_state'):
                continue
            if key == "conv1.weight":
                output_state_dict["model.vision_model.embeddings.patch_embedding.weight"] = val.to(dtype).clone()
            elif key == "conv1.bias":
                output_state_dict["model.vision_model.embeddings.patch_embedding.bias"] = val.to(dtype).clone()
            elif key == "position_embeddings.weight": 
                output_state_dict["model.vision_model.embeddings.position_embedding.weight"] = val.to(dtype).clone()
            elif key == "ln_post.weight":
                output_state_dict["model.vision_model.post_layernorm.weight"] = val.to(dtype).clone()
            elif key == "ln_post.bias":
                output_state_dict["model.vision_model.post_layernorm.bias"] = val.to(dtype).clone()
            elif key.startswith("transformer"):
                if any([name in key for name in ['self_attention.linear_qkv.weight', 'self_attention.linear_qkv.bias', 'self_attention.linear_proj.weight',
                'mlp.linear_fc1.weight', 'mlp.linear_fc1.bias', 'mlp.linear_fc2.weight']]):
                    dim = 1 if 'self_attention.linear_proj.weight' in key or 'mlp.linear_fc2.weight' in key else 0
                    params = torch.cat(
                        [val]
                        + [
                            get_element_from_dict_by_path(tp_state_dicts[tp_rank], 'model.language_model.vision_model')[key]
                            for tp_rank in range(1, tp_size)
                        ],
                        dim=dim,
                    ).to(dtype)
                else:
                    params = val.to(dtype)
                if 'self_attention.linear_qkv' in key and 'layer_norm' not in key:
                    out_val = megatron_to_transformers_fix_query_key_value_ordering(
                        params,
                        3.0,
                        3,
                        vision_config.num_attention_heads,
                        vision_config.hidden_size // vision_config.num_attention_heads,
                    )
                    QKV = {0:'q_proj',1:'k_proj',2:'v_proj'}
                    #print (key, out_val.shape)
                    layer_name = key.split('self_attention.linear_qkv')[0].split('.')[-2]
                    weight_or_bias = key.split('.')[-1]
                    if weight_or_bias == 'bias':
                        for index, matrix in enumerate(torch.split(out_val, out_val.shape[0]//3, 0)):
                            output_state_dict['model.vision_model.encoder.layers.' + layer_name + f".self_attn.{QKV[index]}.{weight_or_bias}"] = matrix.clone()
                    else: 
                        for index, matrix in enumerate(torch.split(out_val, out_val.shape[1], 0)):
                            output_state_dict['model.vision_model.encoder.layers.' + layer_name + f".self_attn.{QKV[index]}.{weight_or_bias}"] = matrix.clone()
                elif 'self_attention.linear_qkv' in key:
                    # handle the layer_norm 
                    layer_name = key.split('self_attention.linear_qkv')[0].split('.')[-2]
                    if key.endswith('weight'):
                        output_state_dict['model.vision_model.encoder.layers.' + layer_name + ".layer_norm1.weight"] = params.clone()
                    else:
                        output_state_dict['model.vision_model.encoder.layers.' + layer_name + ".layer_norm1.bias"] = params.clone()
                elif 'self_attention.linear_proj' in key:
                    layer_name = key.split('self_attention.linear_proj')[0].split('.')[-2]
                    print (key, '->', layer_name + "self_attn.dense.weight")
                    if key.endswith('weight'):
                        output_state_dict['model.vision_model.encoder.layers.' + layer_name + ".self_attn.out_proj.weight"] = params.clone()
                    else:
                        output_state_dict['model.vision_model.encoder.layers.' + layer_name + ".self_attn.out_proj.bias"] = params.clone()
                elif 'mlp.linear_fc1' in key and 'layer_norm' not in key:
                    layer_name = key.split('mlp.linear_fc1')[0].split('.')[-2]
                    if key.endswith('weight'):
                        output_state_dict['model.vision_model.encoder.layers.' + layer_name + ".mlp.fc1.weight"] = params.clone()
                    else:
                        output_state_dict['model.vision_model.encoder.layers.' + layer_name + ".mlp.fc1.bias"] = params.clone()
                elif 'mlp.linear_fc2' in key:
                    layer_name = key.split('mlp.linear_fc2')[0].split('.')[-2]
                    if key.endswith('weight'):
                        output_state_dict['model.vision_model.encoder.layers.' + layer_name + ".mlp.fc2.weight"] = params.clone()
                    else:
                        output_state_dict['model.vision_model.encoder.layers.' + layer_name + ".mlp.fc2.bias"] = params.clone()
                elif 'mlp.linear_fc1' in key:
                    layer_name = key.split('mlp.linear_fc1')[0].split('.')[-2]
                    if key.endswith('weight'):
                        output_state_dict['model.vision_model.encoder.layers.' + layer_name + ".layer_norm2.weight"] = params.clone()
                    else:
                        output_state_dict['model.vision_model.encoder.layers.' + layer_name + ".layer_norm2.bias"] = params.clone()
            else:
                print ('this should not happen', key)


        print("converting perceiver")
        resampler_heads = perceiver_config.resampler_n_heads # 16
        resampler_hidden_size_per_head = perceiver_config.resampler_head_dim 
        perceiver_layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
        # pdb.set_trace()
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], 'model.language_model.perceiver').items():
            if key.endswith('_extra_state'):
                continue
            # Match the name.
            m = perceiver_layer_re.match(key)
            # Stop if that's not a layer
            if m is None:
                continue

            # The index of the layer.
            layer_idx = int(m.group(1)) + pp_rank * num_layers
            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)

            # The name of the layer.
            layer_name = f"model.connector.perceiver_resampler.layers.{layer_idx}"
            print(layer_name, op_name, weight_or_bias)
            # import pdb
            # pdb.set_trace()
            if op_name + "." + weight_or_bias not in tensor_parallel_params_mg:
                params = val.to(dtype)
            else:
                # import pdb
                # pdb.set_trace()
                dim = 1 if op_name in ["self_attention.dense", "mlp.dense_4h_to_h"] else 0
                params = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"model.language_model.perceiver")[key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("input_latents_norm") and weight_or_bias == 'weight':
                output_state_dict[layer_name + "." + "input_latents_norm" + "." + weight_or_bias] = params.clone()
            elif op_name.endswith("input_context_norm") and weight_or_bias == 'weight':
                output_state_dict[layer_name + "." + "input_context_norm" + "." + weight_or_bias] = params.clone()
            elif op_name.endswith("post_attention_layernorm") and weight_or_bias == 'weight':
                output_state_dict[layer_name + "." + "post_attention_layernorm" + "." + weight_or_bias] = params.clone()
            elif op_name == 'self_attention.query' and weight_or_bias == "weight":
                out_val = megatron_to_transformers_fix_query_key_value_ordering(
                    params,
                    checkpoint_version,
                    1,
                    resampler_heads,
                    resampler_hidden_size_per_head,
                )
                output_state_dict[layer_name + f".self_attn.q_proj.weight"] = out_val.clone()
            elif op_name == 'self_attention.key_value' and weight_or_bias == "weight":
                out_val = megatron_to_transformers_fix_query_key_value_ordering(
                    params,
                    checkpoint_version,
                    2,
                    resampler_heads,
                    resampler_hidden_size_per_head,
                )
                KV = {0:'k_proj',1:'v_proj'}
                for index, matrix in enumerate(torch.split(out_val, out_val.shape[0]//2, 0)):
                    output_state_dict[layer_name + f".self_attn.{KV[index]}.weight"] = matrix.clone()
            # Transpose the Q matrix for query for Llama70b.
            # elif (
            #     op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
            # ) and weight_or_bias == "weight":
            #     hidden_size = config.hidden_size 
            #     num_groups = 8
            #     head_dim = config.hidden_size // config.num_attention_heads
            #     num_heads = config.num_attention_heads

            #     tp_states = torch.chunk(params, args.target_tensor_model_parallel_size, dim=0)
            #     query_dim = num_heads // num_groups * head_dim
            #     kv_dim = 2 * head_dim
            #     tp_states = [i.view(num_groups//args.target_tensor_model_parallel_size, query_dim + kv_dim, hidden_size) for i in tp_states]
            #     # tp_dim = hidden_size // args.target_tensor_model_parallel_size
            #     query = torch.cat([i[:, :query_dim].reshape(-1, hidden_size) for i in tp_states])
            #     key_value = torch.cat([i[:, query_dim:].reshape(-1, hidden_size) for i in tp_states])

            #     out_val = megatron_to_transformers_fix_query_key_value_ordering(
            #         query,
            #         checkpoint_version,
            #         1,
            #         heads,
            #         hidden_size_per_head,
            #     )
            #     output_state_dict[layer_name + f".self_attn.q_proj.weight"] = out_val.clone()
            #     out_val = megatron_to_transformers_fix_query_key_value_ordering(
            #         key_value,
            #         checkpoint_version,
            #         2,
            #         8,
            #         hidden_size_per_head,
            #     )
            #     KV = {0:'k_proj',1:'v_proj'}
            #     for index, matrix in enumerate(torch.split(out_val, out_val.shape[0]//2, 0)):
            #         output_state_dict[layer_name + f".self_attn.{KV[index]}.weight"] = matrix.clone()

            # Transpose the weights.
            elif weight_or_bias == "weight":
                if 'dense_h_to_4h' in op_name:
                    out_name = megatron_to_transformers[op_name]
                    para_dict = {i:[] for i in out_name}
                    for index, mat in enumerate(torch.split(params, params.shape[0]//tp_size)):
                        for index_new, mat_sep in enumerate(torch.split(mat, mat.shape[0]//2)):
                            para_dict[out_name[index_new]].append(mat_sep.clone())
                    for name, para in para_dict.items():
                        output_state_dict[layer_name + name + "weight"] = torch.cat(para)
                else:
                    out_name = megatron_to_transformers[op_name]
                    output_state_dict[layer_name + out_name + "weight"] = params.clone()

            # Copy the bias.
            # Ignore them
            elif weight_or_bias == "bias":
                pass

            # Copy the Rotary Embedding
            else:
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + out_name] = params.clone()
        
    # if config.text_config.num_hidden_layers != (layer_idx + 1):
    #     raise ValueError(f"Expected {config.num_hidden_layers} layers but found {layer_idx + 1}")

    ## perceiver final norm
    # may vary in the name
    try:
        output_state_dict['model.connector.perceiver_resampler.norm.weight'] = \
            get_element_from_dict_by_path(tp_state_dicts[0], 'model.language_model.perceiver')['final_layernorm.weight']
    except:
        output_state_dict['model.connector.perceiver_resampler.norm.weight'] = \
            get_element_from_dict_by_path(tp_state_dicts[0], 'model.language_model.perceiver')['final_norm.weight']


    # The final layernorm.
    print("Converting final layernorm")
    params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    try:
        output_state_dict["model.text_model.norm.weight"] = params["final_layernorm.weight"].to(dtype).clone()
    except:
        output_state_dict["model.text_model.norm.weight"] = params["final_norm.weight"].to(dtype).clone()

    # For LM head, transformers' wants the matrix to weight embeddings.
    print("Converting LM head")
    params = torch.cat([
                        get_element_from_dict_by_path(tp_state_dicts[i], 'model.language_model.output_layer.weight')
                        for i in range(tp_size)]
        )
    output_state_dict["lm_head.weight"] = params.to(dtype).clone()[:-args.extra_num_vocabs, :]


    # It should be done!
    print("Conversion from Megatron-LM to Transformers is done!")

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    model = Idefics2ForConditionalGeneration(config)
    model.load_state_dict(output_state_dict, strict=True)
    model.save_pretrained(args.save_path)

    # # Store the config to file.
    # print("Saving config")
    # config.save_pretrained(args.save_path)

    # Store the state_dict to file.
    # max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    # shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)

    # # Save the model
    # if not os.path.exists(args.save_path):
    #     os.system(f'mkdir -p {args.save_path}')
    # for shard_file, shard in shards.items():
    #     torch.save(shard, os.path.join(args.save_path, shard_file))

    # if index is None:
    #     print(f"Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    # else:
    #     save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
    #     # Save the index as well
    #     with open(save_index_file, "w", encoding="utf-8") as f:
    #         content = json.dumps(index, indent=2, sort_keys=True) + "\n"
    #         f.write(content)
    #     print(
    #         f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
    #         f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
    #         f"index located at {save_index_file}."
    #     )


def main():
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    parser = add_transformers_checkpoint_args(parser)
    args = parser.parse_args()
    if args.convert_checkpoint_from_megatron_to_transformers:
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        convert_checkpoint_from_transformers_to_megatron(args)


if __name__ == "__main__":
    main()
