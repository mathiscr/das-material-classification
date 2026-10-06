import os

import matplotlib.pyplot as plt
import numpy as np

from preprocessing import (
    apply_filter,
    load_extract_data,
    plot_waterfall,
)

FILTER_ORDER = 4
LOW_RESOL_FACTOR = 5
OUT_DIR = "data_das/output_figures"
OUT_DIR_NPZ = "data_das/npz"

# Distance (m) of each material along the fibre
MATERIAL_DISTANCES = {"asphalt": 484, "sand": 146, "technical_chamber": 178}

# Config names found in the file path -> material
CONFIG_TO_MATERIAL = {
    "asphalt": ("Config7", "Config9bis", "Config8"),
    "sand": ("Config9", "Config12"),
    "technical_chamber": ("Config10",),
}


def get_distance(material):
    """Return the distance in meters of the given material (None if unknown)."""
    return MATERIAL_DISTANCES.get(material.lower())


def get_material(remote_file):
    """Return the material based on the remote file path."""
    for material, configs in CONFIG_TO_MATERIAL.items():
        if any(config in remote_file for config in configs):
            return material
    return "unknown material"


def extract_event_window(diff_phi, r_search, padding_left=0, padding_right=0):
    """
    Return the index of the event (max of |signal|) in channel r_search,
    ignoring the first `padding_left` and the last `padding_right` samples.
    """
    channel = diff_phi[r_search, :]
    idx_end = channel.shape[0] - padding_right
    return np.argmax(np.abs(channel[padding_left:idx_end])) + padding_left


def plot_event_timeseries(t_axis, signal, title, save_path, t_start, t_end):
    """Plot the time series of the channel and highlight the event window."""
    plt.plot(t_axis, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Phase (Rad)")
    plt.axvspan(t_start, t_end, color="orange")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Time series saved: {save_path}")
    plt.close()


def save_event_from_file(path_file, material, t1=-0.1, t2=0.4, plot=False):
    """
    Extract the event of the file and save it as an npz.
    """
    distance = get_distance(material)
    file_base_name = os.path.splitext(os.path.basename(path_file))[0]

    diff_phi, t_axis, r_axis, Fe_DAS, tcode, spatial_res = load_extract_data(path_file)
    diff_phi_filtered, desc = apply_filter(
        diff_phi, Fe_DAS, ftype="bandpass", order=FILTER_ORDER, bp_low=10, bp_high=200
    )

    dx = spatial_res * LOW_RESOL_FACTOR
    r_search = int(distance / dx)
    window_size = int((t2 - t1) / tcode)
    event_idx = extract_event_window(diff_phi_filtered, r_search, 0, window_size)

    idx_t1 = int(t1 / tcode)
    idx_t2 = int(t2 / tcode)

    # Shift the window if it starts before the beginning of the signal
    if event_idx + idx_t1 < 0:
        correction = -(event_idx + idx_t1)
        idx_t1 += correction
        idx_t2 += correction

    start, end = event_idx + idx_t1, event_idx + idx_t2

    if plot:
        filename = os.path.basename(path_file)

        # Waterfall
        plot_waterfall(
            diff_phi_filtered,
            t_axis,
            r_axis,
            f"{filename}\n{desc} (Butterworth order {FILTER_ORDER})",
            os.path.join(OUT_DIR, f"waterfall_{file_base_name}.png"),
            event_idx * tcode + t1,  # zoom t start (s)
            event_idx * tcode + t2,  # zoom t end (s)
            distance - 50,  # zoom r start (m)
            distance + 50,  # zoom r end (m)
        )
        plt.close()

        # Time series
        plot_event_timeseries(
            t_axis,
            diff_phi_filtered[r_search, :],
            f"{filename}\n{desc} TIME SERIES OF THE EVENT",
            os.path.join(OUT_DIR, f"timeseries_{file_base_name}.png"),
            t_axis[start],
            t_axis[end],
        )

    # Save the npz
    save_path_npz = os.path.join(OUT_DIR_NPZ, f"{file_base_name}.npz")
    os.makedirs(os.path.dirname(save_path_npz), exist_ok=True)
    np.savez(
        save_path_npz,
        time_ax=t_axis[start:end],
        time_series=diff_phi_filtered[r_search, start:end],
        material=material,
    )


if __name__ == "__main__":
    save_event_from_file(
        "data_das/raw_data/DAS_test.mat", material="asphalt", plot=True
    )
