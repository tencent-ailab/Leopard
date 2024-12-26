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

import torch.nn as nn
import re
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
import torch.nn.functional as F
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules
from transformers.activations import ACT2FN
from copy import deepcopy

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_vision_projector_megatron(language_transformer_config, vision_transformer_config, scale_factor=1, delay_load=False, **kwargs):
    vision_projection_type = "mlp"
    vision_projection_config = deepcopy(language_transformer_config)
    vision_projection_config.ffn_hidden_size = language_transformer_config.hidden_size if vision_transformer_config.ffn_hidden_size != 4304 else 4096
    vision_projection_config.activation_func = F.gelu
    vision_projection_config.add_bias_linear = True
    vision_projection_config.gated_linear_unit = False
    vision_projection_modules = MLPSubmodules(
            linear_fc1=ColumnParallelLinear,
            linear_fc2=RowParallelLinear,
        )
    vision_projection = MultimodalProjector(
            vision_projection_config,
            vision_projection_modules,
            vision_projection_type,
            vision_transformer_config.hidden_size * scale_factor # input size to the projection.
        )
    return vision_projection

def build_vision_projector_to_save():
    projector_type = 'mlp2x_gelu'
    mm_hidden_size = 4096
    hidden_size = 4096
    if projector_type == 'linear':
        return nn.Linear(mm_hidden_size, hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

if __name__ == '__main__':
    import torch
    torch.manual_seed(1234)
    mm_projector = build_vision_projector_to_save()
    torch.save(mm_projector.state_dict(), '/apdcephfs_sh2/share_300000800/user/kaixinma/models/clip-vit-large-patch14-336/mm_projector_8B_concat.pth')
    exit()
    import transformers 
    model_id = "/apdcephfs_us/share_300814644/data/nlp/huggingface_models/meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    print (prompt)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    print (tokenizer.tokenize(prompt))

