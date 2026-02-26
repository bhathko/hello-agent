import sys
import os

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chapter_2_action_thought_observe.sample import Chapter2Sample

def main():
    """Main execution point for Chapter 2 sample."""
    sample = Chapter2Sample()
    sample.run()

if __name__ == "__main__":
    main()
