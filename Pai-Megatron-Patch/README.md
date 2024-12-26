## Training
The training code is build on top of Megatron. 
To train the model, make sure you have installed Megatron first. 
Please follow the instruction in the Megatron repository to install Megatron.

Prepare the megatron model by downloading from TBD. (LLaVA and Idefics2)

### LLaVA
Training the model by running the following command:
```bash
cd examples/llava/
# You need to modify the .sh file to specify your own path to the model and data
bash train_multiimg_llava_siglip.sh
```

### Idefics2
```bash
cd examples/idefics2/
# You need to modify the sh file to specify your own path to the model and data
bash train_multiimg_idefics2.sh
```

## After Training
Convert the model from megatron format to huggingface format
by running the following command:

### For LLaVA model

```bash
cd toolkits/model_checkpoints_convertor/llava
bash hf2megatron_convertor_llava.sh
```
Script Variables
This script requires the following variables as input parameters.

- MEGATRON_PATH: The path to the Megatron-LM repository. E.g., Megatron-LM-240603/
- SOURCE_CKPT_PATH: The megatron model path
- TARGET_CKPT_PATH: Path for the converted huggingface model checkpoint.
- TP/PP: 8/1
- MN: llava
- EXTRA_VOCAB_SIZE: 0
- mg2hf: true
- VISION: siglip


### For Idefics2 model
```bash
cd toolkits/model_checkpoints_convertor/idefics2
# You need to modify the .sh file to specify your own path 
bash mg2hg_idefics_multiimg.sh
```