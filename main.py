"""
Main entry point for the Hello-Agent application.
"""
import argparse
import sys
import os

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import chapter runners
from chapter_1_transformer_structure.run import main as run_chapter_1
from chapter_2_action_thought_observe.run import main as run_chapter_2


def main():
    """Main application entry point with chapter selection."""
    parser = argparse.ArgumentParser(description="Hello-Agent: A Chapter-by-Chapter AI Exploration")
    parser.add_argument(
        "--chapter", 
        type=int, 
        choices=[1, 2], 
        default=2, 
        help="The chapter sample to run (default: 2)"
    )
    
    args = parser.parse_args()

    if args.chapter == 1:
        run_chapter_1()
    elif args.chapter == 2:
        run_chapter_2()
    else:
        print(f"Chapter {args.chapter} not found.")


if __name__ == "__main__":
    main()
