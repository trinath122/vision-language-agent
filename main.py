"""
Vision-Language Agentic Reasoning System
Entry point for running the agent interactively.
"""
import argparse
from src.agents import run_agent, AGENT_TOOLS


def main():
    parser = argparse.ArgumentParser(description="Vision-Language Agentic Reasoning System")
    parser.add_argument("--query", type=str, required=True, help="Natural language query")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--max-iter", type=int, default=10)
    args = parser.parse_args()

    print(f"\nQuery: {args.query}")
    if args.image:
        print(f"Image: {args.image}")

    response = run_agent(
        query=args.query,
        image_path=args.image,
        model=None,
        tool_executor=AGENT_TOOLS,
        max_iterations=args.max_iter,
    )

    print(f"\nAgent Response:\n{response}")


if __name__ == "__main__":
    main()
