# LEOPARD <img src="figures/leopard.png" alt="" width="28" height="28">: A Vision Language Model for Text-Rich Multi-Image Tasks
<center><img src="figures/intro.png" alt="Auto-Instruct Illustration" width="" height=""></center>

This is the repository for Leopard, a MLLM that is specifically designed to handle complex vision-language tasks involving multiple text-rich images. In real-world applications, such as presentation slides, scanned documents, and webpage snapshots, understanding the inter-relationships and logical flow across multiple images is crucial.

The code, data, and model checkpoints will be released in one month. Stay tuned!

<p align="center">
  <a href="https://arxiv.org/abs/2410.01744">
    <img src="https://img.shields.io/badge/Paper-000000?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper">
  </a>
  <a href="https://github.com/tencent-ailab/Leopard">
    <img src="https://img.shields.io/badge/Code-181717?style=for-the-badge&logo=github&logoColor=white" alt="Code">
  </a>
  <a href="https://huggingface.co/datasets/wyu1/Leopard-Instruct">
    <img src="https://img.shields.io/badge/Dataset-orange?style=for-the-badge" alt="Dataset">
  </a>

  <a href="https://huggingface.co/wyu1/Leopard-LLaVA">
    <img src="https://img.shields.io/badge/Models-ff6f00?style=for-the-badge&logo=huggingface&logoColor=white" alt="Models">
  </a>

</p>


### Updates

---
- [x] [2024-10-19]. Evaluation code for Leopard-LLaVA and Leopard-Idefics2 is available.
- [x] [2024-10-30]. We release the checkpoints of Leopard-LLaVA and Leopard-Idefics2. 


---

### Key Features:

---
- A High-quality Instruction-Tuning Data: LEOPARD leverages a curated dataset of approximately 1 million high-quality multimodal instruction-tuning samples specifically designed for tasks involving multiple text-rich images.
- Adaptive High-Resolution Multi-image Encoding: An innovative multi-image encoding module dynamically allocates visual sequence lengths based on the original aspect ratios and resolutions of input images, ensuring efficient handling of multiple high-resolution images.
- Superior Performance: LEOPARD demonstrates strong performance across text-rich, multi-image benchmarks and maintains competitive results in general-domain evaluations.


---
<center><img src="figures/model.png" alt="Auto-Instruct Illustration" width="" height=""></center>

### Evaluation

---
For evaluation, please refer to the [Evaluation](evaluation/README.md) section. 

### Model Zoo

---
We provide the checkpoints of Leopard-LLaVA and Leopard-Idefics2 on Huggingface.

- [Leopard-LLaVA](https://huggingface.co/wyu1/Leopard-LLaVA)
- [Leopard-Idefics2](https://huggingface.co/wyu1/Leopard-Idefics2)


