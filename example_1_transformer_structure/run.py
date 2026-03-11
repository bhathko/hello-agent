import sys
import os

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from example_1_transformer_structure.simple_transformer import TransformerSample

def main():
    """Run Chapter 1 sample."""
    sample = TransformerSample()
    sample.run()

if __name__ == "__main__":
    main()
