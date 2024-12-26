from typing import Optional

import torch

from megatron.core import tensor_parallel
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
# from .transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

# Note: This is under development and is missing features like position embedding interpolation.
class Idefics2ViTModel(VisionModule):
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
        class_token_len: int = 1,
    ) -> None:
        super().__init__(config=transformer_config)

        self.visual_hidden_size = transformer_config.hidden_size
        self.patch_dim = patch_dim
        self.img_h = img_h
        self.img_w = img_w
        if self.img_h % self.patch_dim != 0:
            print ('img_h', self.img_h, 'patch_dim', self.patch_dim, 'does not divide evenly')
        if self.img_w % self.patch_dim != 0:
            print ('img_w', self.img_w, 'patch_dim', self.patch_dim, 'does not divide evenly')
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w

        self.class_token_len = class_token_len

        self.seq_length = self.num_patches

        self.conv1 = torch.nn.Conv2d(
            in_channels=transformer_config.num_channels,
            out_channels=self.visual_hidden_size,
            kernel_size=self.patch_dim,
            stride=self.patch_dim,
            # bias=False,
            padding="valid",
        )

        # self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()

        self.position_embeddings = torch.nn.Embedding(self.seq_length, self.visual_hidden_size)

        # self.ln_pre = TENorm(
        #     config=self.config,
        #     hidden_size=self.visual_hidden_size,
        #     eps=self.config.layernorm_epsilon,
        # )


        self.ln_post = TENorm(
            config=self.config,
            hidden_size=self.visual_hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.model_type = ModelType.encoder_or_decoder

        # Transformer + final layer norm (via post_process)
        # TODO: Follow-up changes will make pre and post_process configurable. They are needed for supporting pipeline parallelism.
        self.transformer = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False, # do not do post_process.
        )

        # Note: a final linear layer present in some implementations is omitted here. It can be added separately where needed.

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.transformer.set_input_tensor(input_tensor)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function of the CLIP ViT Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input data of shape [batch, img_h, img_w]
            attention_mask (torch.Tensor with dtype=bool): Attention mask to use. If none, all ones.

        Returns:
            x (torch.Tensor): output after final transformer block of shape [b, s, h].
        """
        batch_size, _, max_im_h, max_im_w = x.shape
        x = self.conv1(x)  # shape = [batch, hidden_size, grid, grid]
        

        # x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch, hidden_size, grid ** 2]
        x = x.flatten(2).transpose(1, 2) # [batch, grid ** 2, hidden_size]
        # x = x.permute(0, 2, 1)  # [batch, grid ** 2, hidden_size]
        
        # torch.save(x, "test_output/new_image_embeddings.pt")

        # get new position_ids

        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_dim, max_im_w // self.patch_dim

        if attention_mask is not None:
            boundaries = torch.arange(1 / self.num_patches_per_dim_h, 1.0, 1 / self.num_patches_per_dim_w)
            position_ids = torch.full(size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)
            for batch_idx, p_attn_mask in enumerate(attention_mask):
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

                fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
                fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

                bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_dim_h + bucket_coords_w).flatten()
                position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids
        else:
            position_ids = torch.arange(0, max_nb_patches_h * max_nb_patches_w).expand(batch_size, -1).cuda()

        position_ids = position_ids.to(self.position_embeddings.weight.device)
        x = x + self.position_embeddings(position_ids)
        # x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # [b, s, h] -> [s, b, h]
        if attention_mask is None:
            attention_mask = torch.ones(1, 1, x.shape[0], x.shape[0]).cuda()  # [1, 1, s, s]
            attention_mask = attention_mask < 0.5  # to bool
        
        attention_mask = attention_mask.view(attention_mask.shape[0], -1)

        # attention_mask = (
        #     _prepare_4d_attention_mask(attention_mask, x.dtype)
        #     if True
        #     else attention_mask
        # )

        if self.config.sequence_parallel:
            # x [grid ** 2, batch_size, hidden_size]
            x = tensor_parallel.scatter_to_sequence_parallel_region(x)

        x = self.transformer(x.contiguous(), attention_mask)
        x = x.permute(1, 0, 2)  # [s, b, h] -> [b, s, h]
        x = x.contiguous()
        if self.config.sequence_parallel:
            x = tensor_parallel.gather_from_tensor_model_parallel_region(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        ### layernorm
        x = self.ln_post(x)

        return x

# import torch
# # from megatron.core.models.vision.clip_vit_model import CLIPViTModel
# from megatron.core.transformer.custom_layers.transformer_engine import TENorm
# from megatron.core.transformer.enums import ModelType
# from megatron.core.transformer.spec_utils import ModuleSpec
# from megatron.core.transformer.transformer_block import TransformerBlock
# from megatron.core.transformer.transformer_config import TransformerConfig

# class CLIPViTModelNoLast(CLIPViTModel):
#     """CLIP ViT vision model.

#     Args:
#         transformer_config (TransformerConfig): Transformer config
#         transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
#         patch_dim (int): Image patch size.
#         img_h (int): Input image height.
#         img_w (int): Input image width.
#         add_class_token (bool, optional): Include a class token. Defaults to True.
#         class_token_len (int): Class token length. Defaults to 1 but 8 may be faster.
#     """

#     def __init__(
#         self,
#         transformer_config: TransformerConfig,
#         transformer_layer_spec: ModuleSpec,
#         patch_dim: int = 14,
#         img_h: int = 336,
#         img_w: int = 336,
#         add_class_token: bool = True,
#         class_token_len: int = 1,
#     ) -> None:
#         super().__init__(transformer_config, transformer_layer_spec)

#         self.visual_hidden_size = transformer_config.hidden_size
#         self.patch_dim = patch_dim
#         self.img_h = img_h
#         self.img_w = img_w
#         assert self.img_h % self.patch_dim == 0
#         assert self.img_w % self.patch_dim == 0
#         self.num_patches_per_dim_h = self.img_h // self.patch_dim
#         self.num_patches_per_dim_w = self.img_w // self.patch_dim
#         self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w

#         self.add_class_token = add_class_token
#         self.class_token_len = class_token_len

#         self.seq_length = self.num_patches + (self.class_token_len if self.add_class_token else 0)

#         self.conv1 = torch.nn.Conv2d(
#             in_channels=3,
#             out_channels=self.visual_hidden_size,
#             kernel_size=self.patch_dim,
#             stride=self.patch_dim,
#             bias=False,
#         )

#         self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()

#         self.position_embeddings = torch.nn.Embedding(self.seq_length, self.visual_hidden_size)

#         self.add_class_token = add_class_token
#         if self.add_class_token:
#             self.class_token = torch.nn.Parameter(
#                 torch.randn(1, self.class_token_len, self.visual_hidden_size)
#             )

#         self.ln_pre = TENorm(
#             config=self.config,
#             hidden_size=self.visual_hidden_size,
#             eps=self.config.layernorm_epsilon,
#         )

#         self.model_type = ModelType.encoder_or_decoder

#         self.transformer = TransformerBlock(
#             config=transformer_config,
#             spec=transformer_layer_spec,
#             pre_process=True,
#             post_process=False,          # KM: This is the only place we changed, to remove the final layer norm
#         )

# import torch.nn.functional as F
# class CLIPVisionTowerMegatron(MegatronModule):
#     def __init__(self, vision_tower, language_transformer_config,
#                  cvcuda_image_processing=False,
#                  mm_vision_select_layer=-2,
#                  delay_load=False):
#         vision_transformer_config = deepcopy(language_transformer_config)
#         vision_transformer_config.num_layers = 23
#         vision_transformer_config.hidden_size = 1024
#         vision_transformer_config.kv_channels = 64 
#         vision_transformer_config.num_attention_heads = 16
#         vision_transformer_config.num_query_groups = 16
#         vision_transformer_config.ffn_hidden_size = 4096
#         vision_transformer_config.activation_func = F.gelu
#         vision_transformer_config.add_bias_linear = True
#         vision_transformer_config.normalization = 'LayerNorm'
#         vision_transformer_config.gated_linear_unit = False
#         print ('vision_transformer_config', vision_transformer_config)
#         super().__init__(vision_transformer_config)

#         self.vision_tower_name = vision_tower
#         self.cvcuda_image_processing = cvcuda_image_processing
#         self.select_layer = mm_vision_select_layer
    
#         vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
#         self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
#         self.vision_tower = CLIPViTModelNoLast(vision_transformer_config, vision_transformer_layer_spec)
#         self.vision_config = vision_transformer_config
#         self.is_loaded = True

#     def forward(self, images):

#         image_forward_outs = self.vision_tower(images)

#         return image_features

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