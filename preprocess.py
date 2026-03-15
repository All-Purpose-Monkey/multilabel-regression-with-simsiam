import os
import glob
import numpy as np
import soundfile as sf
import librosa
import csv
import io
import configparser
from scipy.signal import stft
import datetime
import matplotlib.pyplot as plt

def stitch_resample_6s(house_number, house_channels, dat_folder, output_folder):
    """
    Stitch and resample UK-DALE appliance channel .dat files into a single
    timestamp-aligned CSV with 6-second resolution.

    Workflow:
    1. Load each channel_{n}.dat file (timestamp, power).
    2. Bucket timestamps into 6-second windows using integer flooring.
    3. Average all samples that fall within the same 6-second bucket.
    4. Perform a global outer join across all channels so every timestamp
       appears once.
    5. Save the resulting table as house{n}_stitched.csv.

    Args:
        house_number (int):
            House ID used for locating channel files.

        house_channels (list[int]):
            Channel numbers to include (e.g. appliances + aggregate).

        dat_folder (str):
            Directory containing the channel_{n}.dat files.

        output_folder (str):
            Directory where the stitched CSV will be saved.

    Returns:
        np.ndarray:
            Combined array of shape (N, channels+1)
            where column 0 = timestamp (int64) and remaining columns
            are appliance power values (float, NaN allowed).

    Notes:
        - Timestamps are preserved as exact int64 Unix times.
        - Missing channel readings remain NaN.
        - 6-second resolution aligns with STFT segmentation used
          in the NILM training pipeline.
    """
        
    dat_folder = os.path.expanduser(dat_folder)
    output_folder = os.path.expanduser(output_folder)
    output_path = os.path.join(output_folder, f"house{house_number}_stitched.csv")
    os.makedirs(output_folder, exist_ok=True)

    if os.path.exists(output_path):
        print(f"File already exists, skipping: {output_path}")
    else:
        channel_arrays = []
        all_timestamps = []

        for ch in house_channels:
            path = os.path.join(dat_folder, f"house{house_number}_channel{ch}.dat")
            data = np.loadtxt(path)

            ts = data[:, 0].astype(np.int64)   # keep timestamps as int64
            vals = data[:, 1]

            # ---- GLOBAL 6 SECOND BUCKET ----
            bucket_ts = (ts // 6) * 6

            # ---- AVERAGE VALUES IN SAME BUCKET ----
            unique_ts, inverse = np.unique(bucket_ts, return_inverse=True)
            sums = np.bincount(inverse, weights=vals)
            counts = np.bincount(inverse)
            mean_vals = sums / counts

            channel_arrays.append((unique_ts, mean_vals))
            all_timestamps.append(unique_ts)

        # ---- GLOBAL OUTER JOIN INDEX ----
        all_timestamps = np.unique(np.concatenate(all_timestamps))

        # Keep timestamp column as int64 separately
        timestamps_int64 = all_timestamps.astype(np.int64)
        stitched_vals = np.full((len(all_timestamps), len(house_channels)), np.nan)  # floats for channels

        # ---- FAST JOIN FOR CHANNELS ----
        for col_idx, (ts, vals) in enumerate(channel_arrays):
            idx = np.searchsorted(all_timestamps, ts)
            stitched_vals[idx, col_idx] = vals

        # ---- SAVE CSV ----
        
        header = ['timestamp'] + [f'ch{ch}' for ch in house_channels]

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(timestamps_int64)):
                row = [timestamps_int64[i]] + stitched_vals[i].tolist()
                writer.writerow(row)

        print("Saved:", output_path)
        print("Shape:", stitched_vals.shape)
        print("Timestamps are exact int64 values, channels are float NaN safe.")

def chop_flac(flac_folder, cfg_file, output_base, sample_rate=16000, window_sec=6, hop_samples=1028, n_fft=2048):
    """
    Chop FLAC files into 6-second calibrated segments, compute STFT, and save.

    All partial segments at the end of a file are padded with zeros to maintain uniform size.

    Folder structure:
    output_base/<clean_flac_name_without_ext>/<segment_idx>/stft_voltage.npy
    output_base/<clean_flac_name_without_ext>/<segment_idx>/segment.npy
    """

    

    # -----------------------------
    # Read calibration.cfg
    # -----------------------------
    ADC_HALF_RANGE = 2**31
    config = configparser.ConfigParser()
    cfg_path = os.path.expanduser(cfg_file)
    read_files = config.read(cfg_path)
    if not read_files:
        raise FileNotFoundError(f"Calibration file not found: {cfg_path}")

    volts_per_adc_step = float(config["Calibration"]["volts_per_adc_step"])
    amps_per_adc_step = float(config["Calibration"]["amps_per_adc_step"])
    print(f"Calibration loaded:\nVolts per ADC step: {volts_per_adc_step}\nAmps per ADC step: {amps_per_adc_step}")

    # Expand paths
    flac_folder = os.path.expanduser(flac_folder)
    output_base = os.path.expanduser(output_base)
    os.makedirs(output_base, exist_ok=True)

    # -----------------------------
    # Find FLAC files
    # -----------------------------
    flac_files = sorted(glob.glob(os.path.join(flac_folder, "*.flac")))
    print("Found FLAC files:", flac_files)

    for flac_file in flac_files:
        filename = os.path.basename(flac_file.split(".")[0])

        file_folder = os.path.join(output_base, filename)
        os.makedirs(file_folder, exist_ok=True)
        if os.path.exists(file_folder):
            print(f"Folder already exists, skipping: {file_folder}")
            continue
        else:
            print("Processing:", filename)


            # -----------------------------
            # Load audio using PySoundFile only
            # -----------------------------
            audio, sr = sf.read(flac_file)
            if sr != sample_rate:
                raise ValueError(f"Expected sample rate {sample_rate}, got {sr}")
            if audio.ndim < 2 or audio.shape[1] < 2:
                raise ValueError("Expected at least 2 channels (Voltage, Current)")

            print(f"Audio shape after loading: {audio.shape}, sample rate: {sr}")

            # -----------------------------
            # Apply calibration
            # -----------------------------
            voltage = volts_per_adc_step * ADC_HALF_RANGE * audio[:, 0]
            current = amps_per_adc_step * ADC_HALF_RANGE * audio[:, 1]
            calibrated_audio = np.column_stack([voltage, current])
            print(f"Calibrated audio shape: {calibrated_audio.shape}")

            # -----------------------------
            # Chop into 6-second segments
            # -----------------------------
            win_samples = window_sec * sr
            num_segments = int(np.ceil(len(calibrated_audio) / win_samples))
            print(f"Number of segments to generate: {num_segments}")

            for seg_idx in range(num_segments):
                start = seg_idx * win_samples
                stop = start + win_samples
                segment = calibrated_audio[start:stop, :]

                # Pad last segment if shorter than win_samples
                if segment.shape[0] < win_samples:
                    pad_len = win_samples - segment.shape[0]
                    segment = np.pad(segment, ((0, pad_len), (0, 0)), mode='constant')

                # -----------------------------
                # Compute STFT on voltage channel
                # -----------------------------
                stft_volt = np.abs(librosa.stft(
                    segment[:, 0],
                    n_fft=n_fft,
                    hop_length=hop_samples,
                    window='hann'
                ))

                # -----------------------------
                # Save segment
                # -----------------------------
                file_num = int(filename)  # convert filename string to number

                # folder for this FLAC file
                

                # timestamp-based filename
                timestamp = file_num + (seg_idx + 1) * 6

                np.save(os.path.join(file_folder, f"{timestamp}.npy"), stft_volt)
                # np.save(os.path.join(seg_folder, "segment.npy"), segment)

            print(f"Processed {filename}: {num_segments} segments saved\n")

def mash_that(stitched_csv_path, stft_base):
    """
    Aligns stitched .dat DataFrame with timestamped STFT folders to produce
    X and y ready for ML.
    
    Args:
        stitched_dat (pd.DataFrame): Output from stitch_n_chop_dat(), index = timestamps
        stft_base (str): Base folder containing STFT segment folders (nested by timestamps)
        window_sec (int): Duration of each segment (default 6s)
        
    Returns:
        X (list of np.ndarray): List of STFT arrays
        y (list of np.ndarray): List of corresponding labels from CSV
    """
    # --- Read CSV into a dictionary ---
    stitched_labels = {}
    with open(os.path.expanduser(stitched_csv_path), "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            timestamp = int(float(row[0]))  # convert timestamp to int
            labels = [float(x) for x in row[1:]]  # rest of columns as float
            stitched_labels[timestamp] = labels

    X, y = [], []

    stft_base = os.path.expanduser(stft_base)
    # --- Traverse nested folder structure ---
    for root_folder in sorted(os.listdir(stft_base)):
        root_path = os.path.join(stft_base, root_folder)
        if not os.path.isdir(root_path):
            continue

        for stft_file_name in sorted(os.listdir(root_path)):
            if not stft_file_name.endswith(".npy"):
                continue

            stft_file_path = os.path.join(root_path, stft_file_name)

            try:
                timestamp = int(float(stft_file_name.replace(".npy", "")))
            except ValueError:
                continue  # skip non-numeric filenames

            # Only include timestamps that exist in CSV
            if timestamp not in stitched_labels:
                continue

            # Load STFT
            stft = np.load(stft_file_path)

            # Get label for this timestamp
            label = np.array(stitched_labels[timestamp])

            # Append
            X.append(stft)
            y.append(label)

    return X, y

def label_segmentor(csv_path="~/project/house_1/house1_stitched.csv", output_folder="~/project", target_date=(2013, 9, 16), end_date=None):
    """
    Extract rows from a stitched UK-DALE label CSV for a specific date
    or a range of dates and save them to a new CSV.

    Args:
        csv_path (str):
            Path to the stitched label CSV file. Must contain timestamps
            in the first column (Unix epoch seconds).

        output_folder (str):
            Folder where the segmented CSV will be saved.

        target_date (tuple or datetime.date):
            Start date of extraction. Can be:
            - tuple format: (YYYY, MM, DD)
            - datetime.date object

        end_date (tuple or datetime.date, optional):
            End date for extraction (inclusive). If None, only the
            target_date is extracted.

    Output:
        Saves a CSV named:
            test_label_<startdate>_<enddate>.csv
        or
            test_label_<date>.csv

    Notes:
        - If the output file already exists, the function will skip processing.
        - Useful for creating evaluation subsets aligned with STFT segments.
    """

    csv_path = os.path.expanduser(csv_path)
    output_folder = os.path.expanduser(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    # Convert input dates
    if isinstance(target_date, tuple):
        start_date = datetime.date(*target_date)
    else:
        start_date = target_date

    if end_date is None:
        end_date = start_date
    else:
        if isinstance(end_date, tuple):
            end_date = datetime.date(*end_date)

    # Output filename
    if start_date == end_date:
        output_name = f"test_label_{start_date}.csv"
    else:
        output_name = f"test_label_{start_date}_to_{end_date}.csv"

    output_path = os.path.join(output_folder, output_name)

    # Check if file already exists
    if os.path.exists(output_path):
        print(f"File already exists, skipping: {output_path}")
        return output_path

    # Load CSV
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)

    timestamps = data[:, 0].astype(int)

    # Convert timestamps to dates
    dates = np.array([
        datetime.datetime.utcfromtimestamp(ts).date()
        for ts in timestamps
    ])

    # Create mask for date range
    mask = (dates >= start_date) & (dates <= end_date)
    data_filtered = data[mask]

    # Header reconstruction
    header = ["timestamp"] + [f"ch{i}" for i in range(1, data.shape[1])]

    # Save CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_filtered)

    print(
        f"Saved {data_filtered.shape[0]} rows "
        f"for {start_date} → {end_date} "
        f"to {output_path}"
    )

    return output_path

def extract_day_to_csv(data, target_date, output_folder, filename_prefix="test_label"):

    """
    Extract rows from a target day and save to CSV.
    
    Args:
        data (np.ndarray): 2D array, column 0 = timestamps
        target_date (datetime.date): the day to extract
        output_folder (str): folder to save CSV
        filename_prefix (str): prefix for CSV file
    """
    os.makedirs(os.path.expanduser(output_folder), exist_ok=True)
    
    timestamps = data[:, 0].astype(int)
    dates = np.array([datetime.datetime.utcfromtimestamp(ts).date() for ts in timestamps])
    
    mask = dates == target_date
    data_target_day = data[mask]
    
    output_path = os.path.expanduser(os.path.join(output_folder, f"{filename_prefix}_{target_date}.csv"))
    header = ['timestamp'] + [f'ch{i}' for i in range(1, data.shape[1])]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_target_day)
    
    print(f"Saved {data_target_day.shape[0]} rows for {target_date} → {output_path}")
    return output_path

def plot_stft_example(stft_file= "~/house_1/stft_segments/2013/wk38/1379304000/1379304012.npy"):

    stft_file = os.path.expanduser(stft_file)

    # Load the STFT
    stft_volt = np.load(stft_file)

    print("STFT shape:", stft_volt.shape)  # (frequency_bins, time_frames)

    # Convert to dB for better visualization
    stft_db = 20 * np.log10(stft_volt + 1e-6)  # add epsilon to avoid log(0)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.imshow(stft_db, origin='lower', aspect='auto', cmap='magma')
    plt.colorbar(label='dB')
    plt.xlabel('Time frames')
    plt.ylabel('Frequency bins')
    plt.title('Voltage STFT')
    plt.show()


