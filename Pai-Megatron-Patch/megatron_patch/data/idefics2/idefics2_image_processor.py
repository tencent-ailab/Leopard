from transformers import Idefics2ImageProcessor, Idefics2Processor
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from transformers.utils import TensorType, is_vision_available, logging

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import PaddingMode, pad, resize, to_channel_dimension_format
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)

from transformers.models.idefics2.image_processing_idefics2 import make_pixel_mask

import numpy as np

class Idefics2ProcessorNew(Idefics2Processor):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Idefics2ImageProcessorPad"
    tokenizer_class = "AutoTokenizer"


class Idefics2ImageProcessorPad(Idefics2ImageProcessor):
    def pad(
        self,
        images: List[np.ndarray],
        constant_values: Union[float, Iterable[float]] = 0,
        return_pixel_mask: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        For a list of images, for each images, pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width.
        For each sample in the batch, pads the sample with empty images to the max_number of images per sample in the batch. Optionally returns a pixel mask.

        Args:
            images (`np.ndarray`):
                List of list of images to pad. Pads to the largest height and width in the batch.
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value to use for the padding if `mode` is `"constant"`.
            return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether to return a pixel mask.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # pad_size = get_max_height_width(images, input_data_format=input_data_format)
        pad_size = (self.length, self.length)

        batch_size = len(images)
        max_num_images = max(len(images_) for images_ in images)
        input_data_format = (
            infer_channel_dimension_format(images[0][0]) if input_data_format is None else input_data_format
        )
        data_format = input_data_format if data_format is None else data_format

        def empty_image(size, input_data_format):
            if input_data_format == ChannelDimension.FIRST:
                return np.zeros((3, *size), dtype=np.uint8)
            elif input_data_format == ChannelDimension.LAST:
                return np.zeros((*size, 3), dtype=np.uint8)
            raise ValueError("Invalid channel dimension format.")

        padded_images_list = [
            [empty_image(pad_size, data_format) for _ in range(max_num_images)] for _ in range(batch_size)
        ]
        padded_masks = [[np.zeros(pad_size) for _ in range(max_num_images)] for _ in range(batch_size)]

        for batch_idx in range(batch_size):
            for sample_idx, image in enumerate(images[batch_idx]):
                padded_images_list[batch_idx][sample_idx] = self._pad_image(
                    image,
                    pad_size,
                    constant_values=constant_values,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                padded_masks[batch_idx][sample_idx] = make_pixel_mask(
                    image, output_size=pad_size, input_data_format=input_data_format
                )
        padded_masks = padded_masks if return_pixel_mask else None
        return padded_images_list, padded_masks

