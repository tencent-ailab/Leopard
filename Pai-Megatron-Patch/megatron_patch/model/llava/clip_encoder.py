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

import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
#from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.models.vision.vit_layer_specs import _get_mlp_module_spec
from copy import deepcopy
import warnings
try:
    import cvcuda
    from megatron_patch.data.llava.cvcuda_image_processing_clip import CLIPCVCUDAImageProcessor
    warnings.warn("The cvcuda environment exists, use the cvcuda operator for preprocessing")
except:
    warnings.warn("The cvcuda environment does not exist. Install cvcuda and use it")

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from transformers.activations import ACT2FN
from transformers import Idefics2Config, SiglipConfig
from megatron_patch.model.idefics2.idefics_vision_tower import Idefics2ViTModel

# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_vit_layer_with_transformer_engine_spec() -> ModuleSpec:
    mlp = _get_mlp_module_spec(use_te=True)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={
                    "attn_mask_type": AttnMaskType.no_mask,
                },  # TODO: This should be no_mask when CI is upgraded
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower,
                 cvcuda_image_processing=False,
                 mm_vision_select_layer=-2,
                 mm_vision_select_feature='patch',
                 delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.cvcuda_image_processing = cvcuda_image_processing
        self.select_layer = mm_vision_select_layer
        self.select_feature = mm_vision_select_feature

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if self.cvcuda_image_processing:
            self.image_processor = CLIPCVCUDAImageProcessor.from_pretrained(self.vision_tower_name)
        else:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
class CLIPViTModelNoLast(CLIPViTModel):
    """CLIP ViT vision model.

    Args:
        transformer_config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        patch_dim (int): Image patch size.
        img_h (int): Input image height.
        img_w (int): Input image width.
        add_class_token (bool, optional): Include a class token. Defaults to True.
        class_token_len (int): Class token length. Defaults to 1 but 8 may be faster.
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        patch_dim: int = 14,
        img_h: int = 336,
        img_w: int = 336,
        add_class_token: bool = True,
        class_token_len: int = 1,
    ) -> None:
        super().__init__(transformer_config, transformer_layer_spec)

        self.visual_hidden_size = transformer_config.hidden_size
        self.patch_dim = patch_dim
        self.img_h = img_h
        self.img_w = img_w
        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w

        self.add_class_token = add_class_token
        self.class_token_len = class_token_len

        self.seq_length = self.num_patches + (self.class_token_len if self.add_class_token else 0)

        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=self.visual_hidden_size,
            kernel_size=self.patch_dim,
            stride=self.patch_dim,
            bias=False,
        )

        self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()

        self.position_embeddings = torch.nn.Embedding(self.seq_length, self.visual_hidden_size)

        self.add_class_token = add_class_token
        if self.add_class_token:
            self.class_token = torch.nn.Parameter(
                torch.randn(1, self.class_token_len, self.visual_hidden_size)
            )

        self.ln_pre = TENorm(
            config=self.config,
            hidden_size=self.visual_hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.model_type = ModelType.encoder_or_decoder

        self.transformer = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,          # KM: This is the only place we changed, to remove the final layer norm
        )

import torch.nn.functional as F
from megatron.core import mpu
class CLIPVisionTowerMegatron(MegatronModule):
    def __init__(self, vision_tower, language_transformer_config,
                 cvcuda_image_processing=False,
                 mm_vision_select_layer=-2,
                 mm_vision_select_feature='patch',
                 delay_load=False):
        vision_transformer_config = deepcopy(language_transformer_config)
        vision_transformer_config.num_layers = 23 * mpu.get_pipeline_model_parallel_world_size() # This is to trick the megatron pipeline parallel to get the correct number of layers
        vision_transformer_config.hidden_size = 1024
        vision_transformer_config.kv_channels = 64 
        vision_transformer_config.num_attention_heads = 16
        vision_transformer_config.num_query_groups = 16
        vision_transformer_config.ffn_hidden_size = 4096
        vision_transformer_config.activation_func = F.gelu
        vision_transformer_config.add_bias_linear = True
        vision_transformer_config.normalization = 'LayerNorm'
        vision_transformer_config.gated_linear_unit = False
        vision_transformer_config.sequence_parallel = False
        print ('vision_transformer_config', vision_transformer_config)
        super().__init__(vision_transformer_config)

        self.vision_tower_name = vision_tower
        self.cvcuda_image_processing = cvcuda_image_processing
        self.select_layer = mm_vision_select_layer
        self.select_feature = mm_vision_select_feature
    
        vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPViTModelNoLast(vision_transformer_config, vision_transformer_layer_spec)
        self.vision_config = vision_transformer_config
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs #.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out)#.to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images)
            image_features = self.feature_select(image_forward_outs)#.to(images.dtype)

        return image_features

    # @property
    # def dummy_feature(self):
    #     return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    # @property
    # def dtype(self):
    #     return self.vision_tower.dtype

    # @property
    # def device(self):
    #     return self.vision_tower.device

    # @property
    # def config(self):
    #     if self.is_loaded:
    #         return self.vision_tower.config
    #     else:
    #         return self.cfg_only

    # @property
    # def hidden_size(self):
    #     return self.config.hidden_size

    # @property
    # def num_patches(self):
    #     return (self.config.image_size // self.config.patch_size) ** 2


def get_siglip_vision_tower(load_path, config):
    if load_path.endswith('idefics2-8b'):
        vision_config = Idefics2Config.from_pretrained(load_path).vision_config
    else:
        vision_config = SiglipConfig.from_pretrained(load_path).vision_config

    vision_transformer_config = deepcopy(config)
    vision_transformer_config.num_layers = 27 * mpu.get_pipeline_model_parallel_world_size() # This is to trick the megatron pipeline parallel to get the correct number of layers
    vision_transformer_config.hidden_size = vision_config.hidden_size
    vision_transformer_config.kv_channels = vision_config.hidden_size // vision_config.num_attention_heads #  # kv_channels = hidden_size // num_attention_heads
    vision_transformer_config.num_attention_heads = vision_config.num_attention_heads
    vision_transformer_config.num_query_groups = vision_config.num_attention_heads # no group
    vision_transformer_config.ffn_hidden_size = vision_config.intermediate_size
    vision_transformer_config.activation_func = ACT2FN[vision_config.hidden_act]
    # vision_transformer_config.activation_func = F.gelu
    vision_transformer_config.normalization = 'LayerNorm'
    vision_transformer_config.gated_linear_unit = False
    vision_transformer_config.num_channels = vision_config.num_channels
    vision_transformer_config.layernorm_epsilon = vision_config.layer_norm_eps
    vision_transformer_config.bias_activation_fusion = False # ftq: what's this

    vision_transformer_config.add_qkv_bias = True # there is qkv bias
    vision_transformer_config.add_bias_linear = True # not sure.$ FIXME by FTQ
    vision_transformer_config.apply_rope_fusion = False
    vision_transformer_config.sequence_parallel = False
    # print ('vision_transformer_config', vision_transformer_config)

    vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
    #vision_transformer_layer_spec = get_idefics2_layer_with_transformer_engine_spec()
    # pdb.set_trace()
    vision_model = Idefics2ViTModel(vision_transformer_config, vision_transformer_layer_spec,
                                                    patch_dim = vision_config.patch_size, 
                                                    img_h=vision_config.image_size, img_w=vision_config.image_size) 
    return vision_model