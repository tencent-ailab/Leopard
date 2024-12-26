# Idefics updates


### Aug 1:

How to perform multi-node training:

Remember to use the `--accumulate-allreduce-grads-in-fp32` argument. This is becuase the optimizer performs the computation in fp32, while the default communication channel uses the specified dtype (bf16 in our case) for parameters, which may cost a loss of numerical precision.

### July 31:

How to resume training from a checkpoint?

- Simply use the checkpoint on a new dataset

1. In the output path, rename/copy the folder name of the specific iteration (e.g., `iter_0005000`) to `release`.
2. enter the directory `xxx/checkpoint` (which stores the output checkpoints by steps like `iter_0005000`), execute `touch latest_checkpointed_iteration.txt && cat release > latest_checkpointed_iteration.txt`.


- Resume training on the same dataset (e.g., for the case when training breaks down for certain reasons)

1. Here you need to load the optimizer states from the checkpoint you want to resume by removing the argument `--no-load-optim`
2. Currently the dataloader (defined in Megatron-LM) does not support indexing to a specific step, so the current hack is to add several lines in `Megatron-LM/megatron/legacy/data/data_samplers.py`:

```python
class MegatronPretrainingRandomSampler:

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, data_sharding):
        ...
        self.init_data_size = consumed_samples # New line 1: for continue training
        ...


    def __iter__(self):
        ...

        if self.data_sharding:
            ...
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = \
                torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_total = idx_range_total[self.init_data_size:]
            idx_range_active = idx_range_total[full_bucket_offset:] # New line 2: for continue training
            idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]

        ...
```

A TODO would be integrating this data sampler to the Pai-Megatron.

### July 10:

#### How to run idefics2 

1. Convert huggingface checkpoint.

- If using the original Idefics2 (with Mistral-v0.1-instruct as the backbone LLM)

Need to specify the `idefics2-8b` checkpoint.

```bash
sh toolkits/model_checkpoints_convertor/idefics2/model_convertor.sh
```

- If using Idefics2-Llama3

Need to specify the path to the Llama-x checkpoint, the path to the SigLip checkpoint. The perceiver is randomly initialized.

```bash
sh toolkits/model_checkpoints_convertor/idefics2_llama3/model_convertor.sh
```

2. Prepare webdatase.

If the json line file can take the form of Llava-Pretrain/Instruct:

```
    {"id": "id", 
    "image": "relative/path/to/image", 
    "conversations": 
        [
            {"from": "human", 
            "value": "Recognize all the text.\n<image>"}, 

            {"from": "gpt", 
            "value": "conversation"}]
    }, 
```

Then checkout the `prepare_llava_instruct_new` function at `toolkits/pretrain_data_preprocessing/move_bulk_data.py`.

3. Pre-traing / Alignment of image and text.


- Idefics2-llama3:

```
sh examples/idefics2/pretrain_idefics2_llama3.sh
```

- Useful params:

```bash
    --image-size 378 # Longest edge of images.
    --shortest-edge 378 # pad to shortest edges.
    --max-image-num # maximum number of images per example.
    --freeze-llm # to freeze the backbone language model during training
    --freeze-clip-vision-tower # to freeze the ViT
    --freeze-perceiver # to freeze the perceiver
```

Can freeze the llm and the ViT to be simple.

4. Fine-tuning



- Idefics2-llama3:

```
sh examples/idefics2/finetune_idefics2_llama3.sh
```

- Useful params:

Unfreeze all parameters

Other optional arguments:

```bash
    --answer-loss-only # to only compute the loss of the answer tokens. Used for SFT
```

5. Convert megatron checkpoint to huggingface

Idefics2

```bash
cd toolkits/model_checkpoints_convertor/idefics2
sh mg2hf_base_llava_instruct.sh
```

Idefics2-llama3:

```bash
cd toolkits/model_checkpoints_convertor/idefics2-llama3
sh mg2hf.sh
```

6. Inference with huggingface.

```python

import torch
import json
from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionConfig, Idefics2Config, Idefics2VisionTransformer, Idefics2Connector, Idefics2ForConditionalGeneration
from transformers import AutoConfig, AutoProcessor, AutoModel, SiglipModel

model_path = "path-to-model"

processor = AutoProcessor.from_pretrained(
    model_path,
    do_image_splitting=False
)

config = AutoConfig.from_pretrained(model_path)

model = Idefics2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)

model = model.cuda()

image_folder = "/apdcephfs_us/share_300814644/data/nlp/pretrain_data"
arxiv_data = json.load(open("/apdcephfs_us/share_300814644/data/nlp/pretrain_data/pretrain_arxiv_ocr_v1_head_new_2.json"))

from PIL import Image
import os

texts = []
images = []
for i in range(0, 1):
    example = arxiv_data[i]

    image = Image.open(os.path.join(image_folder, example['image'])).convert('RGB')
    question = example['conversations'][0]['value']
    answer = example['conversations'][1]['value']

    messages = [
        {
            "role": "user",
            "content": [
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
    text = processor.apply_chat_template(messages, add_generation_prompt=False)
    texts.append(text.strip())
    images.append([image])


inputs = processor(text=texts, images=images, return_tensors="pt",  padding='max_length', max_length=2048, truncation=True)

labels = inputs["input_ids"].clone()
inputs["labels"] = labels
inputs['pixel_values'] = inputs['pixel_values']
inputs = {key:val.cuda() for key, val in inputs.items()}


with torch.no_grad():
    output = model(**inputs)

print(output.loss)
    
```


#### Explanations of hyperparameters:

1. checkpoint convertor:

- extra vocab size:

2. model training:




#### June 8:


```
1. examples/idefics
├── debug.sh
├── idefics2.md
├── pretrain_megatron_idefics2.py
├── pretrain_megatron_idefics2_dev.py
```

```
2. megatron_patch/model/idefics
├── get_idefics2vit_layer_spec.py
├── gpt_model.py
├── idefics_vision_tower.py
├── __init__.py
├── language_model_dev.py
├── language_model.py
├── perceiver_transformer.py
├── rotary_pos_embedding.py
└── transformer.py
```

#### How to run:

1. Eval mode:

`sh test_eval.sh`

with `--freeze-clip-vision-tower`,  `--freeze-llm`, and `--freeze-perceiver`. Also `--skip-train`.

2. Training:

`sh test_train.sh`

Removing `--freeze-llm` and `--freeze-perceiver`.

Currently, the ViT module is not trainable, possibly due to the TERowParallelLinear. Change it to Non-transformer-engine later.


#### checkpoints:

Idefics2-8b w/o instruction tuning: /apdcephfs_us/share_300814644/data/nlp/megatron_models/idefics2-8b-base/idefics2-mistral-megatron-tp-8-pp-1

Idefics2-8b after instruction tuning: /apdcephfs_us/share_300814644/data/nlp/megatron_models/idefics2-8b-instruct/idefics2-mistral-megatron-tp-8-pp-1





Some differences
1. megatron_patch/data: New DataCollator in megatron_patch/data_sampler.py. Use the Idefics2Processor directly to collate batches. New mm_pretrain_dataset.py. Change the LazySupervisedDataset such that all processors are done by the new data collator in data_sampler.py
2. megatron_patch/model/idefics2: new gpt_model and language_model, adapted from transformers.Idefics2Model. 
3. examples/idefics2: pretrain_megatron_idefics2.py. New pretrain interface. 


- current only support `do_image_splitting=False` for the processor. meaning only one image is fed to the vision model. Here, `num_patch` may need to be changed in the future to support `do_image_splitting`.

#### Flash-attention in Perceiver

The attention in perceiver (perceiver_transformer.py) is a cross attention between the query `latents` (64 tokens) and key/value of `image_hidden_states` (4900 tokens). 

The core function of flash attention we use is `flash_attn_varlen_func`. 

Though `flash_attn_varlen_func` don't take attention_mask as an input, it supports taking the concatenated tensor after removing all padded token and a `cu_seqlens_q` `cu_seqlens_k` indicating sequence lenghts as input (see the padding in [BERT](https://github.com/Dao-AILab/flash-attention/issues/127)).

A visual illustration and documentation of how flash_attn_varlen_func works: 

https://xtuner.readthedocs.io/zh-cn/latest/acceleration/varlen_flash_attn.html


#### multi-image support:

In json:

- The key "image" should be a list of images.
- A new arguments: max_image_num, indicating the maximum number of images allowed. 


Possible errors:
1. The padding that handles different images numbers, is done in language_model.py. It selects the images where all pixels are 0 as padded images, and these images does not go through the ViT. Change it to a hard image_padding param later.
2. If the number of images cannot correspond to the number of image tokens // 64, there could be errors in input_merger of language_model.py (e.g., when some image tokens fall out of the max length.)