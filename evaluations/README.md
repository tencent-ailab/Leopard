# Evaluations

To evaluate the performance of a model on a benchmark:
1. Prepare the evaluation environment.
2. Prepare the benchmark dataset.
3. Run the evaluation script.

---
## Evaluation Environment
1. Follow the instructions in [LLaVA](https://github.com/haotian-liu/LLaVA) repository to set up the evaluation environment.
2. Install the required packages.
```bash
# Make sure you are currently in evaluations/ directorty
pip install -r ../requirements.txt
```


## Text-rich Multi-Image Benchmarks

### MP-DocvVQA

1. Download the image.tar.gz and question-answer.zip from https://rrc.cvc.uab.es/?ch=17&com=downloads. (Note: Registration is required.)
2. Extract the image.tar.gz into mpdocvqa/images forder.
3. Unzip the question-answer.zip, move val.json into mpdocvqa/ folder.
4. Run load_mpdocvqa.py to prepare the dataset.
```bash
cd mpdocvqa/ && python load_mpdocvqa.py
```

### DUDE
1. Run load_dude.py to prepare the dataset. Data will be downloaded from huggingface Datasets.
```bash
cd dude/ && python load_dude.py
```

### SlideVQA
1. Follow the instructions in https://github.com/nttmdlab-nlp/SlideVQA to download the dataset.
2. Run load_slidevqa.py to prepare the dataset.
```bash
cd slidevqa/ && python load_slidevqa.py
```

### MultiChartQA
1. Download the dataset (the data/ folder) from https://github.com/Zivenzhu/Multi-chart-QA/tree/main into multichartqa/data/ folder.
2. Run load_multichartqa.py to prepare the dataset. 
```bash
cd multichartqa/ && python load_multichartqa.py
```

### MultiHiertt
1. Download dev.json from https://drive.google.com/drive/folders/1ituEWZ5F7G9T9AZ0kzZZLrHNhRigHCZJ into multihiertt/ folder.
2. Run load_multihiertt.py to prepare the dataset. 
```bash
cd multihiertt/ && python load_multihiertt.py
```

---

## Text-rich Single Image Benchmarks

### TextVQA
1. Download the 'TextVQA_0.5.1_val.json' and images from https://textvqa.org/dataset/.
2. Unzip the images into textvqa/images/ folder.
3. Run load_textvqa.py to prepare the dataset. 
```bash
cd textvqa/ && python load_textvqa.py
```

### DocVQA
1. Download the val_v1.0_withQT.json and images from https://rrc.cvc.uab.es/?ch=17&com=downloads. (Note: Registration is required.)
2. Unzip the images into docvqa/images/ folder.
3. Run load_docvqa.py to prepare the dataset. 
```bash
cd docvqa/ && python load_docvqa.py
```

### VisualWebBench
1. Download the dataset files (*.parquert) from https://huggingface.co/datasets/visualwebbench/VisualWebBench.
2. Run load_visualwebbench.py to prepare the dataset. 
```bash
cd visualwebbench/ && python load_visualwebbench.py
```

---

## General Benchmarks

### MIRB
1. Download the dataset files (*.parquert) from https://huggingface.co/datasets/VLLMs/MIRB/tree/main.
2. Run load_mirb.py to prepare the dataset. 
```bash
cd mirb/ && python load_mirb.py
```

### MIBench
TBD

### MMMU
1. Download the dataset files (*.parquert) from https://huggingface.co/datasets/MMMU/MMMU.
2. Run load_mmmu.py to prepare the dataset. 
```bash
cd mmmu/ && python load_mmmu.py
```

### MathVista
1. Download the testmini-00000-of-00001-725687bf7a18d64b.parquet file and images.zip from https://huggingface.co/datasets/AI4Math/MathVista.
2. Unzip the images into mathvista/images folder.
3. Run load_mathvista.py to prepare the dataset. 
```bash
cd mathvista/ && python load_mathvista.py
```

### ScienceQA
1. Download the dataset files (*.parquert) from https://huggingface.co/datasets/ScienceQA/ScienceQA.
2. Run load_scienceqa.py to prepare the dataset. 
```bash
cd scienceqa/ && python load_scienceqa.py
```

---


## Evaluation Script
To evaluate Leopard-LLaVA model:
```bash
# Make sure you are currently in the evaluations/ directory
cd models/ && bash run_eval_llava_siglip_multiimg.sh direct $MODEL_PATH
```


To evaluate Leopard-Idefics model:
```bash
# Make sure you are currently in the evaluations/ directory
cd models/ && bash run_eval_idefics2_multiimg.sh direct $MODEL_PATH
```

The scripts will eval the performance of the model on all benchmark datasets.
