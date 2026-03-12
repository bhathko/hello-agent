"""
Main entry point for the Hello-Agent application.
"""
import argparse

# Import example runners
from examples.transformer_structure import main as run_example_1
from examples.action_thought_observe import main as run_example_2
from examples.classic_agent_build import main as run_example_3
from examples.basic_agent_tool import main as run_example_4


def main():
    """Main application entry point with chapter selection."""
    parser = argparse.ArgumentParser(description="Hello-Agent: A Chapter-by-Chapter AI Exploration")
    parser.add_argument(
        "--chapter", 
        type=int, 
        choices=[1, 2, 3, 4], 
        default=4, 
        help="The example sample to run (default: 4)"
    )
    
    args = parser.parse_args()

    if args.chapter == 1:
        run_example_1()
    elif args.chapter == 2:
        run_example_2()
    elif args.chapter == 3:
        run_example_3()
    elif args.chapter == 4:
        run_example_4()
    else:
        print(f"Example {args.chapter} not found.")


if __name__ == "__main__":
    main()
