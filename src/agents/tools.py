from langchain.tools import tool
from PIL import Image
import base64
import io
from typing import Any


@tool
def image_captioning(image_path: str) -> str:
    """Generate a detailed caption for the given image path."""
    # Plugs into the VisionLanguageModel.generate() pipeline
    return f"[caption for {image_path}]"


@tool
def visual_grounding(image_path: str, query: str) -> str:
    """
    Locate and return bounding box coordinates for objects described by the query.
    Returns: JSON string with bounding boxes.
    """
    return f"[grounding result for '{query}' in {image_path}]"


@tool
def object_detection(image_path: str) -> str:
    """
    Detect all objects in the image and return their labels and confidence scores.
    Returns: JSON list of detected objects.
    """
    return f"[detected objects in {image_path}]"


@tool
def vqa(image_path: str, question: str) -> str:
    """
    Answer a natural language question about the given image.
    Used for visual question answering (VQA) tasks.
    """
    return f"[answer to '{question}' about {image_path}]"


@tool
def image_to_base64(image_path: str) -> str:
    """Convert a local image file to a base64 string for API payloads."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


AGENT_TOOLS = [image_captioning, visual_grounding, object_detection, vqa, image_to_base64]
