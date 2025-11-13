#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Image Captioning Module for AI Marketplace Platform

Uses Salesforce/blip2-opt-2.7b for high-quality image descriptions.
Provides fast image captioning for multimodal chat functionality.

Usage:
    from image_caption import ImageCaptioner

    captioner = ImageCaptioner()
    caption = captioner.caption_image("path/to/image.jpg")
    print(f"Image description: {caption}")
"""

import os
import io
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from typing import Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageCaptioner:
    """
    BLIP-2 Image Captioning with Salesforce/blip2-opt-2.7b

    Generates high-quality descriptions of images for multimodal AI interactions.
    """

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._model_loaded = False

    def _load_model(self):
        """Load BLIP-2 model and processor on first use"""
        if self._model_loaded:
            return

        try:
            logger.info(f"Loading image captioning model: {self.model_name}")

            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_name)

            # Load model with appropriate dtype
            if self.device.type == "cuda":
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                )
                self.model = self.model.to(self.device)

            # Set to eval mode
            self.model.eval()

            logger.info(f"Image captioning model loaded on {self.device}")
            self._model_loaded = True

        except Exception as e:
            logger.error(f"Failed to load image captioning model: {e}")
            raise

    def _prepare_image(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """
        Prepare image for captioning

        Args:
            image_input: Path to image file, bytes, or PIL Image

        Returns:
            PIL Image
        """
        if isinstance(image_input, str):
            # Load from file path
            image = Image.open(image_input)
        elif isinstance(image_input, bytes):
            # Load from bytes
            image = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            # Already a PIL Image
            image = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def caption_image(
        self,
        image_input: Union[str, bytes, Image.Image],
        max_length: int = 50,
        num_beams: int = 5
    ) -> str:
        """
        Generate caption for image

        Args:
            image_input: Path to image file, bytes, or PIL Image
            max_length: Maximum caption length
            num_beams: Number of beams for beam search

        Returns:
            Generated caption as string
        """
        # Load model on first use
        self._load_model()

        try:
            # Prepare image
            image = self._prepare_image(image_input)

            # Process image
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False
                )

            # Decode caption
            caption = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

            logger.info(f"Generated caption: {caption}")
            return caption

        except Exception as e:
            logger.error(f"Failed to generate caption: {e}")
            return "Unable to generate image description"

    def caption_with_context(
        self,
        image_input: Union[str, bytes, Image.Image],
        context: Optional[str] = None,
        max_length: int = 50,
        num_beams: int = 5
    ) -> str:
        """
        Generate caption with optional context/prompt

        Args:
            image_input: Path to image file, bytes, or PIL Image
            context: Optional context or prompt
            max_length: Maximum caption length
            num_beams: Number of beams for beam search

        Returns:
            Generated caption with context
        """
        # Load model on first use
        self._load_model()

        try:
            # Prepare image
            image = self._prepare_image(image_input)

            # Prepare prompt (optional)
            prompt = context if context else None

            # Process with prompt if provided
            if prompt:
                # For conditional captioning
                inputs = self.processor(
                    image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)
            else:
                # For unconditional captioning
                inputs = self.processor(image, return_tensors="pt").to(self.device)

            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False
                )

            # Decode caption
            caption = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

            logger.info(f"Generated caption with context '{context}': {caption}")
            return caption

        except Exception as e:
            logger.error(f"Failed to generate caption with context: {e}")
            return "Unable to generate image description"

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "loaded": self._model_loaded,
            "model_type": "BLIP-2 Vision-Language Model"
        }

# Global instance for reuse
_image_captioner = None

def get_image_captioner() -> ImageCaptioner:
    """Get or create global image captioner instance"""
    global _image_captioner
    if _image_captioner is None:
        _image_captioner = ImageCaptioner()
    return _image_captioner

def caption_image(image_input: Union[str, bytes, Image.Image]) -> str:
    """
    Convenience function for image captioning

    Args:
        image_input: Path to image file, bytes, or PIL Image

    Returns:
        Generated caption as string
    """
    captioner = get_image_captioner()
    return captioner.caption_image(image_input)

def caption_image_with_context(
    image_input: Union[str, bytes, Image.Image],
    context: Optional[str] = None
) -> str:
    """
    Convenience function for image captioning with context

    Args:
        image_input: Path to image file, bytes, or PIL Image
        context: Optional context or prompt

    Returns:
        Generated caption as string
    """
    captioner = get_image_captioner()
    return captioner.caption_with_context(image_input, context)