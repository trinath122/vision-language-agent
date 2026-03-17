from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence
import operator
from pathlib import Path
import os
import ollama
import numpy as np
from PIL import Image
from .tools import AGENT_TOOLS


# Candidate labels for CLIP zero-shot image description
IMAGE_LABELS = [
    "a person", "a woman", "a man", "a child", "a group of people",
    "an anime character", "a cartoon character", "a superhero", "a ninja",
    "a dog", "a cat", "a fox", "a wolf", "a mythical creature", "an animal",
    "a car", "a building", "a tree", "a mountain", "an ocean", "a sky",
    "food", "fruit", "vegetables",
    "a street", "a room", "a forest", "a city", "a beach",
    "digital art", "anime art", "a painting", "a drawing", "illustration",
    "text", "a chart", "a screenshot",
    "fire", "water", "snow",
    "action scene", "fighting scene",
    "sword", "weapon", "armor",
    "smiling", "happy", "sad", "angry",
    "two characters facing each other", "a character and an animal",
]

SYSTEM_PROMPT = """You are a Vision-Language Reasoning Agent.
You can analyze images, answer questions about visual content,
and reason step-by-step. When image context is provided, use it to ground your answer.
"""


class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    image_path: str | None
    iteration: int
    max_iterations: int


def describe_image_with_pil(image_path: str) -> str:
    """Extract basic visual properties from image using PIL."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # Dominant color analysis
    small = img.resize((50, 50))
    pixels = np.array(small).reshape(-1, 3)
    avg = pixels.mean(axis=0).astype(int)
    r, g, b = int(avg[0]), int(avg[1]), int(avg[2])

    # Brightness
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    bright_desc = "bright" if brightness > 180 else "dark" if brightness < 80 else "medium-lit"

    # Color tone
    if r > g and r > b:
        tone = "warm/red-orange tones"
    elif g > r and g > b:
        tone = "green tones"
    elif b > r and b > g:
        tone = "cool/blue tones"
    else:
        tone = "neutral/mixed tones"

    return (
        f"Image properties: {width}x{height} pixels, {bright_desc} image, "
        f"dominant color is RGB({r},{g},{b}) suggesting {tone}."
    )


def should_continue(state: AgentState) -> str:
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and not getattr(last, "tool_calls", None):
        return "end"
    return "continue"


def call_model(state: AgentState, model) -> AgentState:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    prompt = "\n".join(f"{m.type}: {m.content}" for m in messages)

    image_path = state.get("image_path")
    if image_path and Path(image_path).exists():
        print("Analyzing image with CLIP...")
        image_description = describe_image_with_pil(image_path)
        prompt = f"{image_description}\n\n{prompt}"
        print(f"CLIP detected: {image_description}")

    response = ollama.generate(model="deepseek-r1:1.5b", prompt=prompt)
    response_text = response["response"]

    return {
        "messages": [AIMessage(content=response_text)],
        "iteration": state["iteration"] + 1,
        "image_path": state["image_path"],
        "max_iterations": state["max_iterations"],
    }


def call_tools(state: AgentState, tool_executor) -> AgentState:
    last_message = state["messages"][-1]
    tool_results = tool_executor.invoke(last_message)
    return {
        "messages": tool_results if isinstance(tool_results, list) else [tool_results],
        "iteration": state["iteration"],
        "image_path": state["image_path"],
        "max_iterations": state["max_iterations"],
    }


def build_agent_graph(model, tool_executor):
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", lambda state: call_model(state, model))
    workflow.add_node("tools", lambda state: call_tools(state, tool_executor))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def run_agent(query: str, image_path: str | None, model, tool_executor, max_iterations: int = 10):
    graph = build_agent_graph(model, tool_executor)
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "image_path": image_path,
        "iteration": 0,
        "max_iterations": max_iterations,
    }
    final_state = graph.invoke(initial_state)
    return final_state["messages"][-1].content
