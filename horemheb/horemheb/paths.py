from pathlib import Path

# Use __file__ to get the location of this file, then navigate up two directories
temp_sense = Path(__file__).parent.parent.parent

notebooks = temp_sense / "notebooks"
data_dir = temp_sense / "data"