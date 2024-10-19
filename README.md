# LEOPARD <img src="figures/leopard.png" alt="" width="28" height="28">: A Vision Language Model for Text-Rich Multi-Image Tasks

This is the repository for Leopard, a MLLM that is specifically designed to handle complex vision-language tasks involving multiple text-rich images. In real-world applications, such as presentation slides, scanned documents, and webpage snapshots, understanding the inter-relationships and logical flow across multiple images is crucial.

The code, data, and model checkpoints will be released in one month. Stay tuned!

<center><img src="figures/intro.png" alt="Auto-Instruct Illustration" width="" height=""></center>
---

### Key Features:

- A High-quality Instruction-Tuning Data: LEOPARD leverages a curated dataset of approximately 1 million high-quality multimodal instruction-tuning samples specifically designed for tasks involving multiple text-rich images.
- Adaptive High-Resolution Multi-image Encoding: An innovative multi-image encoding module dynamically allocates visual sequence lengths based on the original aspect ratios and resolutions of input images, ensuring efficient handling of multiple high-resolution images.
- Superior Performance: LEOPARD demonstrates strong performance across text-rich, multi-image benchmarks and maintains competitive results in general-domain evaluations.


---
<center><img src="figures/model.png" alt="Auto-Instruct Illustration" width="" height=""></center>

### Evaluation
For evaluation, see the [Evaluation](evaluation/README.md) section. 

