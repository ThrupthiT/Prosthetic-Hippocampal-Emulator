# src/feature_extraction.py
import os
import numpy as np
import pandas as pd
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def choose_channel_name(raw, preferred="Cz"):
    chs = raw.ch_names
    # prefer Cz, else pick first EEG channel
    if preferred in chs:
        return preferred
    # fallback heuristics
    for cand in ["Cz", "FCz", "Fz", "C3", "C4"]:
        if cand in chs:
            return cand
    return chs[0]

def windowed_spectrogram_and_bandpower(raw, out_subj_dir,
                                       subject_id, rec_name,
                                       channel=None,
                                       win_sec=2.0, step_sec=1.0,
                                       fmin=1, fmax=45,
                                       nperseg_seconds=0.5,
                                       image_size=(224,224)):
    """
    raw: preprocessed MNE Raw object (filtered)
    out_subj_dir: folder to save images and csv (outputs/S001/)
    subject_id: string like 'S001'
    rec_name: file basename like 'S001R01'
    """
    sfreq = raw.info['sfreq']
    nperseg = int(nperseg_seconds * sfreq)
    noverlap = nperseg // 2

    ch_name = channel if channel else choose_channel_name(raw)
    picks = [ch_name]
    data, times = raw.get_data(picks=picks, return_times=True)  # shape (n_ch=1, n_samples)
    data = data[0]  # single channel array
    total_seconds = data.shape[0] / sfreq

    # prepare output dirs
    img_dir = os.path.join(out_subj_dir, "images")
    ensure_dir(img_dir)
    csv_path = os.path.join(out_subj_dir, "bandpower.csv")
    rows = []

    win_samples = int(win_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    idx = 0
    win_i = 0
    while idx + win_samples <= data.shape[0]:
        seg = data[idx: idx + win_samples]
        t0 = idx / sfreq
        t1 = (idx + win_samples) / sfreq

        # compute spectrogram (frequencies x times)
        freqs, times_spec, Sxx = signal.spectrogram(seg, fs=sfreq,
                                                    nperseg=nperseg, noverlap=noverlap,
                                                    scaling='density', mode='psd')
        # restrict freq range
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs_sel = freqs[freq_mask]
        Sxx_sel = Sxx[freq_mask, :]

        # convert to log power (dB)
        Sxx_db = 10 * np.log10(Sxx_sel + 1e-12)

        # normalize to 0-255
        Smin, Smax = Sxx_db.min(), Sxx_db.max()
        img_arr = (Sxx_db - Smin) / (Smax - Smin + 1e-9)  # 0-1
        img_arr = (img_arr * 255).astype(np.uint8)  # uint8

        # Convert to PIL Image (freq x time). We rotate to have time on x-axis:
        img = Image.fromarray(img_arr)
        img = img.resize(image_size[::-1])  # note: spectrogram shape (freq,time) => PIL expects (width, height)
        # convert to RGB (3-channel) if needed
        img = img.convert("RGB")

        # save image
        fname = f"{subject_id}_{rec_name}_ch-{ch_name}_win-{win_i:05d}.png"
        out_path = os.path.join(img_dir, fname)
        img.save(out_path)

        # compute band powers in this window (theta,beta,gamma)
        # Use Welch PSD on the segment for bandpower
        freqs_w, psd = signal.welch(seg, fs=sfreq, nperseg=nperseg)
        def band_power_from_psd(psd, freqs, a, b):
            mask = (freqs >= a) & (freqs <= b)
            return np.trapz(psd[mask], freqs[mask])

        theta_bp = band_power_from_psd(psd, freqs_w, 4, 8)
        beta_bp = band_power_from_psd(psd, freqs_w, 13, 30)
        gamma_bp = band_power_from_psd(psd, freqs_w, 30, 45)

        rows.append({
            "subject": subject_id,
            "recording": rec_name,
            "channel": ch_name,
            "window_index": win_i,
            "start_time": t0,
            "end_time": t1,
            "theta_power": theta_bp,
            "beta_power": beta_bp,
            "gamma_power": gamma_bp,
            "image_path": out_path,
            "label": ""  # fill if you have ground truth
        })

        win_i += 1
        idx += step_samples

    # append CSV
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_new = pd.DataFrame(rows)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(rows)} images to {img_dir} and updated CSV at {csv_path}")

# Example usage (call after you have raw_filtered and subject/rec file info):
# out_subj_dir = os.path.join(OUTPUT_DIR, "S001")
# windowed_spectrogram_and_bandpower(raw_filtered, out_subj_dir, subject_id="S001", rec_name="S001R01")
