import sys
import os

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from example_3_classic_agent_build.sample import Example3Sample

def main():
    """Main execution point for Chapter 4 sample."""
    sample = Example3Sample()
    sample.run()

if __name__ == "__main__":
    main()
