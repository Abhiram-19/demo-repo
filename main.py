import traceback
import sys
from visualizer import FM3QVisualizer

def main():
    try:
        # Create and run the visualizer
        print("Initializing FM3Q Visualizer...")
        vis = FM3QVisualizer(size=8, cell_size=60)
        vis.run()
    except Exception as e:
        print(f"Uncaught exception: {e}")
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())