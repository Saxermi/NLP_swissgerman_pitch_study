import os
import pandas as pd
from aubio import source, pitch
import time
import concurrent.futures
from tqdm import tqdm

def process_wav_file(filepath):
    # Parameters for pitch detection
    win_size = 4096  # window size
    hop_size = 512  # hop size

    # Initialize source with correct sample rate
    s = source(filepath, 0, hop_size)  # setting samplerate to 0 lets aubio detect it from the file
    samplerate = s.samplerate

    # Initialize pitch detection object
    pitch_o = pitch("yin", win_size, hop_size, samplerate)
    pitch_o.set_unit("Hz")
    pitch_o.set_tolerance(0.8)

    pitches = []
    confidences = []

    # Pitch detection loop
    while True:
        samples, read = s()
        p = pitch_o(samples)[0]
        confidence = pitch_o.get_confidence()
        pitches.append(p)
        confidences.append(confidence)
        if read < hop_size:
            break

    return filepath, pitches, confidences

def collect_data(results, all_data):
    filepath, pitches, confidences = results
    df = pd.DataFrame({
        "File": [os.path.basename(filepath)] * len(pitches),
        "Pitch": pitches,
        "Confidence": confidences
    })
    all_data = pd.concat([all_data, df], ignore_index=True)
    return all_data

def main():
    subfolder = 'wav/15'
    
    if not os.path.exists(subfolder):
        print(f"The folder '{subfolder}' does not exist.")
        return

    files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.wav')]

    all_data = pd.DataFrame()
    start_time_all = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map the function to the files and wrap with tqdm for a progress bar
        results = list(tqdm(executor.map(process_wav_file, files), total=len(files), desc="Processing Files"))
        for result in results:
            all_data = collect_data(result, all_data)

    all_data.to_csv('all_pitches_py_comparison.csv', index=False)
    total_duration = time.time() - start_time_all
    print(f"Total processing time for all files: {total_duration:.2f} seconds. Data saved to 'all_pitches.csv'.")

if __name__ == "__main__":
    main()

