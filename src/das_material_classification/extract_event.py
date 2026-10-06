import os

import matplotlib.pyplot as plt
import numpy as np

from preprocessing import (
    LOW_RESOL_FACTOR,
    apply_filter,
    load_extract_data,
    plot_waterfall,
)

# Asphalt: 484m
# sand: 146m
# Technical chamber: 178m

FILTER_ORDER = 4
LOW_RESOL_FACTOR = 5
OUT_DIR = r"Data_DAS/output_figures"
OUT_DIR_NPZ = r"Data_DAS/npz"


def get_distance(material):
    """
    Return the distance in meters based on the material.
    """
    material_distances = {"asphalt": 484, "sand": 146, "technical_chamber": 178}
    return material_distances.get(
        material.lower()
    )  # Default to technical_chamber if material not found


def get_material(remote_file):
    """
    Return the material based on the remote file path.
    """
    if (
        "Config7" in remote_file
        or "Config9bis" in remote_file
        or "Config8" in remote_file
    ):
        return "asphalt"
    if "Config9" in remote_file or "Config12" in remote_file:
        return "sand"
    if "Config10" in remote_file:
        return "technical_chamber"
    return "unknown material"


def extract_event_window(diff_phi, r_search, padding_left=0, padding_right=0):
    """
    Return the first index of the event in diff_phi in channel r_search of siwe
    window_size
    """
    # Definition of the channel we will work on
    channel = diff_phi[r_search, :]

    idx_end = (
        channel.shape[0] - padding_right if padding_right > 0 else channel.shape[0]
    )
    idx_event = np.argmax(np.abs(channel[padding_left:idx_end])) + padding_left
    return idx_event


def save_event_from_file(path_file, material, t1=-0.1, t2=0.4, plot=False):
    """
    Save the event of the file
    """
    distance = get_distance(material)

    # Extract the base name of the file without extension
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

    if event_idx + idx_t1 < 0:
        correction = -(event_idx + idx_t1)
        idx_t2 += correction
        idx_t1 += correction

    safe_desc = desc.replace(" ", "_").replace("-", "-").replace(".", "p")
    if plot:
        save_path_waterfall = os.path.join(OUT_DIR, f"waterfall_{file_base_name}.png")
        title_waterfall = (
            f"{os.path.basename(path_file)}\n{desc} (Butterworth order {FILTER_ORDER})"
        )
        zoom_t_start = event_idx * tcode + t1  # seconds
        zoom_t_end = event_idx * tcode + t2  # seconds
        zoom_r_start = distance - 50  # meters
        zoom_r_end = distance + 50  # meters
        plot_waterfall(
            diff_phi_filtered,
            t_axis,
            r_axis,
            title_waterfall,
            save_path_waterfall,
            zoom_t_start,
            zoom_t_end,
            zoom_r_start,
            zoom_r_end,
        )
        plt.close()
        save_path_timeseries = os.path.join(OUT_DIR, f"timeseries_{file_base_name}.png")

        title_timeseries = (
            f"{os.path.basename(path_file)}\n{desc} TIME SERIES OF THE EVENT)"
        )
        plt.plot(t_axis, diff_phi_filtered[r_search, :])
        plt.title = title_timeseries
        plt.xlabel("Time (s)")
        plt.ylabel("Phase (Rad)")
        plt.axvspan(
            t_axis[event_idx + idx_t1], t_axis[event_idx + idx_t2], color="orange"
        )
        os.makedirs(os.path.dirname(save_path_timeseries), exist_ok=True)
        plt.savefig(save_path_timeseries, dpi=150, bbox_inches="tight")
        print(f"Time series saved: {save_path_timeseries}")
        # plt.show()
        plt.close()

    # Savee of the npz
    save_path_npz = os.path.join(OUT_DIR_NPZ, f"{file_base_name}.npz")
    os.makedirs(os.path.dirname(save_path_npz), exist_ok=True)
    np.savez(
        save_path_npz,
        time_ax=t_axis[event_idx + idx_t1 : event_idx + idx_t2],
        time_series=diff_phi_filtered[
            r_search, event_idx + idx_t1 : event_idx + idx_t2
        ],
        material=material,
    )


save_event_from_file("../../data_das/raw_data/DAS_test.mat", material="asphalt", plot=True)
