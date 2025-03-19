from pydub import AudioSegment
import os

RAW_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Convert an OGG file to WAV format
def convert_ogg_to_wav(input_path, output_path):
    audio = AudioSegment.from_ogg(input_path)
    audio.export(output_path, format="wav")


# Convert all OGG files
for file in os.listdir(RAW_DIR):
    if file.endswith(".ogg"):
        input_file = os.path.join(RAW_DIR, file)
        output_file = os.path.join(PROCESSED_DIR, file.replace(".ogg", ".wav"))
        convert_ogg_to_wav(input_file, output_file)


print("Conversion complete.")