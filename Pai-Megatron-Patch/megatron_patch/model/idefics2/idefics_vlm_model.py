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
import os
import torch
import pdb

try:
    from megatron import get_args
    from megatron.core import mpu, tensor_parallel
    from megatron.core.enums import ModelType
    from megatron.model.enums import AttnMaskType
    from megatron.model.module import MegatronModule
    from megatron.model.utils import get_linear_layer
    from megatron.model.utils import init_method_normal
    from megatron.model.utils import scaled_init_method_normal
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
except:
    from megatron.training import get_args
    from megatron.core import mpu, tensor_parallel
    from megatron.core.enums import ModelType
    from megatron.legacy.model.enums import AttnMaskType
    from megatron.legacy.model.module import MegatronModule
    from megatron.legacy.model.utils import get_linear_layer
    from megatron.legacy.model.utils import init_method_normal
    from megatron.legacy.model.utils import scaled_init_method_normal
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    # from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
    from .get_idefics2vit_layer_spec import get_idefics2_layer_with_transformer_engine_spec

from copy import deepcopy
from megatron_patch.model.mistral.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from megatron_patch.data.llava.constants import IMAGE_TOKEN_INDEX
# from .clip_encoder import CLIPVisionTower
# from .mm_projector_builder import build_vision_projector
from .transformer import ParallelTransformer
from .perceiver_transformer import Idefics2PerceiverParallelTransformer
import os 
from typing import Dict, List, Optional, Tuple, Union

from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionConfig, Idefics2Config, Idefics2VisionTransformer, Idefics2Connector
from transformers.activations import ACT2FN
from .idefics_vision_tower import Idefics2ViTModel as Idefics2ViTModelMegatron
from megatron_patch.model.llava.mm_projector_builder import build_vision_projector_megatron

def parallel_lm_logits(input_, word_embeddings_weight, parallel_output,
                       bias=None):
    """LM logits using word embedding weights."""
    args = get_args()
    # Parallel logits.
    if args.async_tensor_model_parallel_allreduce or\
            args.sequence_parallel:
        input_parallel = input_
        model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        async_grad_allreduce = args.async_tensor_model_parallel_allreduce and \
            model_parallel and not args.sequence_parallel
    else:
        input_parallel = tensor_parallel.copy_to_tensor_model_parallel_region(input_)
        async_grad_allreduce = False

    # Matrix multiply.
    logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
        input=input_parallel,
        weight=word_embeddings_weight,
        bias=bias,
        gradient_accumulation_fusion=args.gradient_accumulation_fusion,
        async_grad_allreduce=async_grad_allreduce,
        sequence_parallel=args.sequence_parallel)
    # Gather if needed.

    if parallel_output:
        return logits_parallel

    return tensor_parallel.gather_from_tensor_model_parallel_region(logits_parallel)


def get_language_model(config, num_tokentypes, add_pooler,
                       encoder_attn_mask_type,
                       add_encoder=True,
                       add_decoder=False,
                       decoder_attn_mask_type=AttnMaskType.causal,
                       pre_process=True, post_process=True):
    """Build language model and return along with the key to save."""
    args = get_args()
    if config.init_method is None:
        config.init_method = init_method_normal(config.init_method_std)

    if config.output_layer_init_method is None:
        config.output_layer_init_method = scaled_init_method_normal(config.init_method_std,
                                                                    config.num_layers)

    # Language model.
    language_model = TransformerLanguageModel(
        config,
        encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process
    )
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key


class Pooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method):
        super(Pooler, self).__init__()
        args = get_args()
        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)
        self.sequence_parallel = args.sequence_parallel


    def forward(self, hidden_states, sequence_index=0):
        # hidden_states: [s, b, h]
        # sequence_index: index of the token to pool.

        # gather data along sequence dimensions
        # same pooler is run on all tensor parallel nodes
        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False)

        pooled = hidden_states[sequence_index, :, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 config,
                 num_tokentypes=0):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = config.init_method
        self.num_tokentypes = num_tokentypes

        args = get_args()

        # Word embeddings (parallel).
        self.params_dtype = args.params_dtype
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            vocab_size, self.hidden_size, config=config, init_method=config.init_method)
        self._word_embeddings_key = 'word_embeddings'
        print ('Current Vocab Size', vocab_size)
        # Position embedding (serial).
        self.add_position_embedding = args.position_embedding_type == 'learned_absolute'
        if self.add_position_embedding:
            self.position_embeddings = torch.nn.Embedding(
                max_sequence_length, self.hidden_size)
            self._position_embeddings_key = 'position_embeddings'
            # Initialize the position embeddings.
            if args.perform_initialization:
                self.init_method(self.position_embeddings.weight)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes,
                                                           self.hidden_size)
            # Initialize the token-type embeddings.
            if args.perform_initialization:
                self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        self.fp32_residual_connection = args.fp32_residual_connection
        self.sequence_parallel = False #args.sequence_parallel         KM: llava can't have sequence parallel because we're building multimodal inputs later 
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        if self.add_position_embedding:
            self.position_embeddings.weight.data.fill_(0)
            self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            print('adding embedding for {} tokentypes'.format(num_tokentypes),
                  flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes,
                                                       self.hidden_size)
        # Initialize the token-type embeddings.
        args = get_args()
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        # Embeddings.
        words_embeddings = self.word_embeddings(torch.clamp(input_ids, min=0))  # to handle negative numbers check
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] \
            = self.word_embeddings.state_dict(prefix=prefix,
                                              keep_vars=keep_vars)
        if self.add_position_embedding:
            state_dict_[self._position_embeddings_key] \
                = self.position_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] \
                = self.tokentype_embeddings.state_dict(prefix=prefix,
                                                       keep_vars=keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] \
                        = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self.add_position_embedding:
            if self._position_embeddings_key in state_dict:
                state_dict_ = state_dict[self._position_embeddings_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'position_embeddings' in key:
                        state_dict_[key.split('position_embeddings.')[1]] \
                            = state_dict[key]
            self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if 'tokentype_embeddings' in key:
                        state_dict_[key.split('tokentype_embeddings.')[1]] \
                            = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_,
                                                          strict=strict)
            else:
                print('***WARNING*** expected tokentype embeddings in the '
                      'checkpoint but could not find it', flush=True)

import torch.nn.functional as F
# Idefics LLM
class TransformerLanguageModel(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 config,
                 encoder_attn_mask_type,
                 num_tokentypes=0,
                 add_encoder=True,
                 add_decoder=False,
                 decoder_attn_mask_type=AttnMaskType.causal,
                 add_pooler=False,
                 pre_process=True,
                 post_process=True):
        self.args = get_args()
        # TODO: passing share_embeddings_and_output_weights=False will not work correctly for T5 and embeddings will not be synced. Fix later for T5.
        if self.args.untie_embeddings_and_output_weights: assert not add_decoder
        super(TransformerLanguageModel, self).__init__(share_embeddings_and_output_weights=not self.args.untie_embeddings_and_output_weights)

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = config.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = config.init_method
        self.add_encoder = add_encoder
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_decoder = add_decoder
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.encoder_hidden_state = None
        self.add_retriever = self.args.retro_add_retriever
        self.untie_embeddings_and_output_weights = self.args.untie_embeddings_and_output_weights
        self.dtype = torch.half if self.args.fp16 else torch.bfloat16
        self.sliding_window = self.args.sliding_window

        # Embeddings.
        if self.pre_process:
            self.embedding = Embedding(self.hidden_size,
                                       self.args.padded_vocab_size,
                                       self.args.max_position_embeddings,
                                       self.args.hidden_dropout,
                                       config,
                                       self.num_tokentypes)
            self._embedding_key = 'embedding'

            self.vision_config = Idefics2Config.from_pretrained(self.args.vision_tower).vision_config

            vision_transformer_config = deepcopy(config)
            vision_transformer_config.num_layers = self.vision_config.num_hidden_layers * mpu.get_pipeline_model_parallel_world_size() # This is to trick the megatron pipeline parallel to get the correct number of layers
            vision_transformer_config.hidden_size = self.vision_config.hidden_size
            vision_transformer_config.kv_channels = self.vision_config.hidden_size // self.vision_config.num_attention_heads #  # kv_channels = hidden_size // num_attention_heads
            vision_transformer_config.num_attention_heads = self.vision_config.num_attention_heads
            vision_transformer_config.num_query_groups = self.vision_config.num_attention_heads # no group
            vision_transformer_config.ffn_hidden_size = self.vision_config.intermediate_size
            vision_transformer_config.activation_func = ACT2FN[self.vision_config.hidden_act]
            vision_transformer_config.normalization = 'LayerNorm'
            vision_transformer_config.gated_linear_unit = False
            vision_transformer_config.num_channels = self.vision_config.num_channels
            vision_transformer_config.layernorm_epsilon = self.vision_config.layer_norm_eps
            vision_transformer_config.bias_activation_fusion = False # ftq: what's this

            vision_transformer_config.add_qkv_bias = True # there is qkv bias
            vision_transformer_config.add_bias_linear = True # not sure.$ FIXME by FTQ
            vision_transformer_config.apply_rope_fusion = False
            vision_transformer_config.sequence_parallel = False

            # print ('vision_transformer_config', vision_transformer_config)

            # vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
            vision_transformer_layer_spec = get_idefics2_layer_with_transformer_engine_spec()
            # pdb.set_trace()
            self.vision_model = Idefics2ViTModelMegatron(vision_transformer_config, vision_transformer_layer_spec,
                                                        patch_dim = self.vision_config.patch_size, 
                                                        img_h=self.vision_config.image_size, img_w=self.vision_config.image_size)

            if self.args.freeze_clip_vision_tower:
                for param in self.vision_model.parameters():
                    param.requires_grad = False

            self.config_mm = Idefics2Config.from_pretrained(self.args.vision_tower)
            self.image_token_id  = self.config_mm.image_token_id

            perceiver_transformer_config = deepcopy(config)
            # add configs
            perceiver_transformer_config.num_layers = self.config_mm.perceiver_config.resampler_depth
            perceiver_transformer_config.intermediate_size = self.config_mm.text_config.intermediate_size
            # perceiver_transformer_config.hidden_size = self.vision_config.hidden_size # FIXME
            perceiver_transformer_config.kv_channels = self.config_mm.perceiver_config.resampler_head_dim # self.vision_config.hidden_size // self.vision_config.num_attention_heads #  # kv_channels = hidden_size // num_attention_heads
            perceiver_transformer_config.num_attention_heads = self.config_mm.perceiver_config.resampler_n_heads
            perceiver_transformer_config.num_query_groups = self.config_mm.perceiver_config.resampler_n_heads # no group
            # perceiver_transformer_config.ffn_hidden_size = self.vision_config.intermediate_size
            perceiver_transformer_config.activation_func = F.silu
            # perceiver_transformer_config.activation_func = F.gelu
            perceiver_transformer_config.normalization = 'RMSNorm' 
            perceiver_transformer_config.gated_linear_unit = True
            perceiver_transformer_config.layernorm_epsilon = self.config_mm.text_config.rms_norm_eps
            perceiver_transformer_config.bias_activation_fusion = False # ftq: what's this

            perceiver_transformer_config.add_qkv_bias = False # there is qkv bias
            perceiver_transformer_config.add_bias_linear = False # not sure.$ FIXME by FTQ
            perceiver_transformer_config.apply_rope_fusion = False
            perceiver_transformer_config.vision_hidden_size = self.config_mm.vision_config.hidden_size
            perceiver_transformer_config.text_hidden_size = self.config_mm.text_config.hidden_size
            perceiver_transformer_config.n_latents = self.config_mm.perceiver_config.resampler_n_latents

            perceiver_transformer_config.sequence_parallel = False # KM: it's nearly impossible to support sequence parallel in perceiver 
            self.connector = Idefics2PerceiverParallelTransformer(perceiver_transformer_config, self.args.model_type)

            if perceiver_transformer_config.text_hidden_size != self.hidden_size:
                print ('Size mismatch', perceiver_transformer_config.text_hidden_size, self.hidden_size)
                perceiver_transformer_config.hidden_size = perceiver_transformer_config.text_hidden_size
                perceiver_transformer_config.ffn_hidden_size = perceiver_transformer_config.text_hidden_size
                self.mm_projector = build_vision_projector_megatron(config, perceiver_transformer_config)
                for k, v in self.mm_projector.named_parameters():
                    print (k, v.shape)
            else:
                self.mm_projector = None

            if self.args.freeze_perceiver:
                for name, param in self.connector.named_parameters():
                    param.requires_grad = False
            
        if self.args.freeze_llm:
            for param in self.embedding.parameters():
                param.requires_grad = False

        # Rotary positional embeddings
        if self.args.use_rotary_position_embeddings:
            self.seq_length = self.args.seq_length
            rotary_dim = self.args.hidden_size // self.args.num_attention_heads \
                if self.args.kv_channels is None else self.args.kv_channels

            if self.args.rotary_percent < 1.0:
                rotary_dim = int(rotary_dim * self.args.rotary_percent)

            # partial rotary embeddings, which is better than full rotary
            # Wang and Komatsuzaki et al
            # https://github.com/kingoflolz/mesh-transformer-jax/
            self.rotary_pos_emb = RotaryEmbedding(
                rotary_dim,
                seq_len_interpolation_factor=self.args.rotary_seq_len_interpolation_factor, rotary_base=self.args.rotary_base, rotary_percent=self.args.rotary_percent,
            )
            self.use_rotary_position_embeddings = True
        elif self.args.use_mistral_rotary_position_embeddings:
            self.use_rotary_position_embeddings = False

        print ('Add_encoder', self.add_encoder, self.add_decoder)
        if self.add_encoder:
            self.encoder = ParallelTransformer(
                config,
                model_type=self.args.model_type if not self.args.retro_add_retriever \
                    else ModelType.retro_decoder,
                self_attn_mask_type=self.encoder_attn_mask_type,
                pre_process=self.pre_process,
                post_process=self.post_process,
            )
            self._encoder_key = 'encoder'

            if self.args.freeze_llm:
                for param in self.encoder.parameters():
                    param.requires_grad = False

        if self.post_process:
            if self.untie_embeddings_and_output_weights:
                self.output_layer = tensor_parallel.ColumnParallelLinear(
                    self.args.hidden_size,
                    self.args.padded_vocab_size,
                    config=config,
                    init_method=self.init_method,
                    bias=False) # Setting bias to False always to keep it consistent with embedding tying that also does not have a bias.
                self._output_layer_key = 'output_layer'

                if self.args.freeze_llm:
                    for param in self.output_layer.parameters():
                        param.requires_grad = False

    def encode_images(self, images):
        image_features = self.vision_tower(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""

        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        if self.add_encoder and self.add_decoder:
            assert len(input_tensor) == 1, \
                'input_tensor should only be length 1 for stage with both encoder and decoder'
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            assert len(input_tensor) == 1, \
                'input_tensor should only be length 1 for stage with only encoder'
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_decoder:
            if len(input_tensor) == 2:
                self.decoder.set_input_tensor(input_tensor[0])
                self.encoder_hidden_state = input_tensor[1]
            elif len(input_tensor) == 1:
                self.decoder.set_input_tensor(None)
                self.encoder_hidden_state = input_tensor[0]
            else:
                raise Exception('input_tensor must have either length 1 or 2')
        else:
            raise Exception('Stage must have at least either encoder or decoder')
    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.Tensor],
        image_hidden_states: Optional[torch.Tensor],
    ):
        """
        This method aims at merging the token embeddings with the image hidden states into one single sequence of vectors that are fed to the transformer LM.
        The merging happens as follows:
        - The text token sequence is: `tok_1 tok_2 tok_3 <fake_token_around_image> <image> <image> ... <image> <fake_token_around_image> tok_4`.
        - We get the image hidden states for the image through the vision encoder (and potentially the perceiver), and that hidden state is then projected into the text embedding space.
        We thus have a sequence of image hidden states of size (1, image_seq_len, hidden_dim), where 1 is for batch_size of 1 image and hidden_dim is the hidden_dim of the LM transformer.
        - The merging happens so that we obtain the following sequence: `vector_tok_1 vector_tok_2 vector_tok_3 vector_fake_tok_around_image {sequence of image_seq_len image hidden states} vector_fake_toke_around_image vector_tok_4`. That sequence is fed to the LM.
        - To fit the format of that sequence, `input_ids`, `input_embeds`, `attention_mask` are all 3 adapted to insert the image hidden states.
        """
        num_images, _, vision_hidden_size = image_hidden_states.shape
        special_image_token_mask = input_ids == self.image_token_id
        new_inputs_embeds = inputs_embeds.clone()
        reshaped_image_hidden_states = image_hidden_states.view(-1, vision_hidden_size)
        # print("inputs_merger")
        # print(new_inputs_embeds.shape, special_image_token_mask.shape, image_hidden_states.shape, reshaped_image_hidden_states.shape)
        new_inputs_embeds[special_image_token_mask] = reshaped_image_hidden_states
        return new_inputs_embeds
    
    def forward(self, enc_input_ids, enc_position_ids, enc_attn_mask,
                dec_input_ids=None, dec_position_ids=None, dec_attn_mask=None,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                enc_dec_attn_mask=None, tokentype_ids=None,
                inference_params=None,
                pooling_sequence_index=0,
                enc_hidden_states=None, output_enc_hidden=False, images=None,
                pixel_values=None, pixel_attention_mask=None
                ):
        # new inputs: pixel_values, pixel_attention_mask
        if self.pre_process:
            input_embeds = self.embedding(enc_input_ids, enc_position_ids,
                                            tokentype_ids=tokentype_ids)
            if images.shape[-1] > 100:
                input_embeds = input_embeds.permute(1, 0, 2).clone()
                if pixel_attention_mask is not None:
                    patch_size = self.vision_config.patch_size
                    patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
                    patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
                    patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()
                    # if mpu.get_tensor_model_parallel_rank() == 0:
                    #     print ('patch_attention_mask', patch_attention_mask.shape, patch_attention_mask)
                else:
                    patch_attention_mask = None
                image_features = self.vision_model(
                    x=images,
                    attention_mask=patch_attention_mask,
                )

                if mpu.get_tensor_model_parallel_rank() == 0 and image_features.isnan().any():
                    print ('after vision encoder', image_features.shape, image_features[:, :10, :10])

                # Modality projection & resampling
                image_features = self.connector(
                    image_features, attention_mask=patch_attention_mask.view(images.size(0), -1)
                ).permute(1, 0, 2).contiguous()

                if mpu.get_tensor_model_parallel_rank() == 0 and image_features.isnan().any():
                    print ('after perceriver', image_features.shape, image_features[:, :10, :10])

                if self.mm_projector:
                    image_features = self.mm_projector(image_features)
                
                if mpu.get_tensor_model_parallel_rank() == 0 and image_features.isnan().any():
                    print ('after projector', image_features.shape, image_features[:, :10, :10])

                image_mask = (enc_input_ids == IMAGE_TOKEN_INDEX)
                num_positions = image_mask.sum().item()
                assert image_features.shape[0] * image_features.shape[1] == num_positions, \
                f"The number of image features doesn't match the number of image tokens {image_features.shape} != {num_positions}"
                indices = torch.nonzero(image_mask, as_tuple=True)
                image_features = image_features.view(-1, image_features.shape[-1])
                input_embeds[indices] = image_features
                encoder_input = input_embeds.permute(1, 0, 2)
                # if mpu.get_tensor_model_parallel_rank() == 0:
                #     print ('encoder input', encoder_input.shape)
                # exit()
            else:
                encoder_input = input_embeds
        else:
            encoder_input = None

        # Retriever embedding.
        if self.add_retriever and self.pre_process:
            retriever_input = self.embedding(retriever_input_ids,
                                             retriever_position_ids,
                                             tokentype_ids=tokentype_ids)
        else:
            retriever_input = None

        # Rotary positional embeddings
        rotary_pos_emb = None
        if self.use_rotary_position_embeddings:
            if inference_params is not None:
                rotary_pos_emb = \
                    self.rotary_pos_emb(inference_params.max_sequence_length)
            else:
                rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
            rotary_pos_emb = rotary_pos_emb[:enc_input_ids.shape[1]]

        if self.pre_process and self.args.sequence_parallel:
            encoder_input = tensor_parallel.scatter_to_sequence_parallel_region(encoder_input)


        if enc_position_ids is None:
            past_key_values_length = 0
            seq_length = self.seq_length
            device = enc_input_ids.device\
                if enc_input_ids is not None else encoder_input.device
            position_ids = torch.arange(past_key_values_length,
                                        seq_length + past_key_values_length,
                                        dtype=torch.long,
                                        device=device)
            enc_position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # Run encoder.
        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output = self.encoder(
                    encoder_input,
                    enc_attn_mask,
                    retriever_input=retriever_input,
                    retriever_attn_mask=retriever_attn_mask,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb,
                    position_ids=enc_position_ids
                )
            else:
                encoder_output = self.encoder_hidden_state
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)

        # print("in language_model.py:: encoder_output", encoder_output.shape)
        
        # add pooler is false

        if self.post_process:
            if self.add_pooler:
                pooled_output = self.pooler(encoder_output,
                                            pooling_sequence_index)

        # output_enc_hidden refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output
            else:
                return encoder_output

        # Decoder embedding.
        if self.pre_process:
            decoder_input = self.embedding(dec_input_ids,
                                           dec_position_ids)
        else:
            decoder_input = None

        # Run decoder.
        decoder_output = self.decoder(
            decoder_input,
            dec_attn_mask,
            encoder_output=encoder_output,
            enc_dec_attn_mask=enc_dec_attn_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb)

        # print("in language_model.py:: decoder_output", decoder_output.shape)

        if self.add_pooler and self.post_process:
            return decoder_output, encoder_output, pooled_output
        else:
            return decoder_output, encoder_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        if self.pre_process:
            state_dict_[self._embedding_key] \
                = self.embedding.state_dict_for_save_checkpoint(prefix=prefix,
                                                                keep_vars=keep_vars)
            state_dict_['vision_model'] = self.vision_model.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)
            state_dict_['perceiver'] = self.connector.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)
            if self.mm_projector is not None:
                state_dict_['mm_projector'] = self.mm_projector.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)
        if self.add_encoder:
            state_dict_[self._encoder_key] \
                = self.encoder.state_dict_for_save_checkpoint(prefix=prefix,
                                                              keep_vars=keep_vars)
        if self.post_process:
            if self.add_pooler:
                state_dict_[self._pooler_key] \
                    = self.pooler.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
            if self.untie_embeddings_and_output_weights:
                state_dict_[self._output_layer_key] \
                    = self.output_layer.state_dict(prefix=prefix, keep_vars=keep_vars)

        if self.add_decoder:
            state_dict_[self._decoder_key] \
                = self.decoder.state_dict_for_save_checkpoint(prefix=prefix,
                                                              keep_vars=keep_vars)
        
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        args = get_args()
        # Embedding.
        if self.pre_process:
            if self._embedding_key in state_dict:
                state_dict_ = state_dict[self._embedding_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if '_embeddings' in key:
                        state_dict_[key] = state_dict[key]
            self.embedding.load_state_dict(state_dict_, strict=strict)

            state_dict_vision = state_dict['vision_model']
            state_dict_vision = {k:v for k, v in state_dict_vision.items()}

            for k, v in self.vision_model.state_dict().items():
                if k.endswith('_extra_state'):
                    state_dict_vision[k] = v
            self.vision_model.load_state_dict(state_dict_vision, strict=strict)

            ### connector:
            state_dict_connector = state_dict['perceiver']
            state_dict_connector = {k:v for k, v in state_dict_connector.items()}
            for k, v in self.connector.state_dict().items():
                if k.endswith('_extra_state'):
                    state_dict_connector[k] = v
            self.connector.load_state_dict(state_dict_connector, strict=strict)
            if self.mm_projector is not None:
                state_dict_mm_projector = state_dict['mm_projector']
                self.mm_projector.load_state_dict(state_dict_mm_projector, strict=strict)
            else:
                print ('MM projector is None, thus skip loading weights')

        # Encoder.
        if self.add_encoder:
            if self._encoder_key in state_dict:
                state_dict_ = state_dict[self._encoder_key]
            # For backward compatibility.
            elif 'transformer' in state_dict:
                state_dict_ = state_dict['transformer']
            else:
                # For backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'transformer.' in key:
                        state_dict_[key.split('transformer.')[1]] = state_dict[key]

            # For backward compatibility.
            state_dict_self_attention = {}
            for key in state_dict_.keys():
                if '.attention.' in key:
                    state_dict_self_attention[key.replace(".attention.",
                        ".self_attention.")] = state_dict_[key]
                else:
                    state_dict_self_attention[key] = state_dict_[key]
            state_dict_ = state_dict_self_attention

            if args.transformer_impl == "transformer_engine":
                self.encoder.load_state_dict(state_dict_, strict=False)
            else:
                for k, v in self.encoder.state_dict().items():
                    if k not in state_dict_ and k.endswith('inv_freq'):
                        print (f'{k} not found')
                        state_dict_[k] = v
                self.encoder.load_state_dict(state_dict_, strict=strict)

        # Pooler.
        if self.post_process:
            if self.add_pooler:
                assert 'pooler' in state_dict, \
                    'could not find data for pooler in the checkpoint'
                self.pooler.load_state_dict(state_dict[self._pooler_key],
                                            strict=strict)
            if self.untie_embeddings_and_output_weights:
                assert 'output_layer' in state_dict, \
                    'could not find data for output_layer in the checkpoint'
                self.output_layer.load_state_dict(state_dict[self._output_layer_key],
                                                  strict=strict)
        # Decoder.
        if self.add_decoder:
            assert 'decoder' in state_dict, \
                'could not find data for pooler in the checkpoint'
            self.decoder.load_state_dict(state_dict[self._decoder_key],
                                         strict=strict)
