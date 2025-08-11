import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    print("Attempting to import SanskritPostProcessor...")
    from src.post_processors.sanskrit_post_processor import SanskritPostProcessor
    print("Import successful!")
except Exception as e:
    print("An error occurred during import:")
    import traceback
    traceback.print_exc()
