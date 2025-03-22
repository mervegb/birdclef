import os
import shutil

input_dir = "data/processed/spectrograms"
output_dir = "data/processed/spectrograms_sorted"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".png"):
        _, species = file.split("_", 1)
        species = species.replace(".png", "")
        species_dir = os.path.join(output_dir, species)
        os.makedirs(species_dir, exist_ok=True)
        shutil.copy(os.path.join(input_dir, file), os.path.join(species_dir, file.split("_")[0] + ".png"))