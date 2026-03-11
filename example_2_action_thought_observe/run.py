import sys
import os

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from example_2_action_thought_observe.sample import Example2Sample

def main():
    """Main execution point for Chapter 2 sample."""
    sample = Example2Sample()
    sample.run()

if __name__ == "__main__":
    main()
