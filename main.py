"""
Main entry point for the Hello-Agent application.
"""
import argparse
import sys
import os

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import example runners
from example_1_transformer_structure.run import main as run_example_1
from example_2_action_thought_observe.run import main as run_example_2
from example_3_classic_agent_build.run import main as run_example_3


def main():
    """Main application entry point with chapter selection."""
    parser = argparse.ArgumentParser(description="Hello-Agent: A Chapter-by-Chapter AI Exploration")
    parser.add_argument(
        "--chapter", 
        type=int, 
        choices=[1, 2, 3], 
        default=3, 
        help="The example sample to run (default: 3)"
    )
    
    args = parser.parse_args()

    if args.chapter == 1:
        run_example_1()
    elif args.chapter == 2:
        run_example_2()
    elif args.chapter == 3:
        run_example_3()
    else:
        print(f"Example {args.chapter} not found.")


if __name__ == "__main__":
    main()
