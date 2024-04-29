import os
from pydub import AudioSegment

def convert_mp3_to_wav(mp3_file, wav_file):
    # Load MP3 file
    audio = AudioSegment.from_mp3(mp3_file)
    # Export as WAV
    audio.export(wav_file, format="wav")

def convert_flac_to_wav(flac_file, wav_file):
    # Load FLAC file
    audio = AudioSegment.from_file(flac_file, format="flac")
    # Export as WAV
    audio.export(wav_file, format="wav")

def main():
    source_folder = 'source'  # Folder where audio files are located
    output_folder = 'wav'  # Folder to save WAV files
    total_files_processed = 0

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all audio files in the source directory
    files = [f for f in os.listdir(source_folder) if f.endswith('.mp3') or f.endswith('.flac')]
    total_files = len(files)

    # Convert each file to WAV
    for file in files:
        source_path = os.path.join(source_folder, file)
        if file.endswith('.mp3'):
            wav_filename = file.replace('.mp3', '.wav')
            convert_mp3_to_wav(source_path, os.path.join(output_folder, wav_filename))
        elif file.endswith('.flac'):
            wav_filename = file.replace('.flac', '.wav')
            convert_flac_to_wav(source_path, os.path.join(output_folder, wav_filename))
        total_files_processed += 1
        print(f"Conversion complete: {wav_filename}")

    print(f"Total files processed: {total_files_processed} out of {total_files}")

if __name__ == "__main__":
    main()
