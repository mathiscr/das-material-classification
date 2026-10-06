# %%
# ============================================================
#  DAS Differential Phase Extraction + Bandpass Filter + Waterfall Plot
#  All configurable parameters are centralized here. Modify and run.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
import os

# ============================================================
#  *** User-configurable parameters (only modify here) ***
# ============================================================

# --- Data path ---
DATA_PATH = r"data/DAS_Simu_3.mat"

# --- Output path (where to save figures; set to None to display only) ---
OUT_DIR = r'Data_DAS/output_figures'
OUT_DIR_RAW = r'Data_DAS/RAW_output_figures'

# --- Channel selection ---
LOW_RESOL_FACTOR = 5       # Spatial downsampling factor (1 = keep full 0.5 m resolution, 5 = Pruvost's 2.5 m gauge)

# --- Filter settings ---
FILTER_TYPE = 'bandpass'   # 'highpass' or 'bandpass'
FILTER_ORDER = 4           # Butterworth filter order

# Highpass mode (used when FILTER_TYPE='highpass')
HP_CUTOFF = 3.0            # Hz (Pruvost's paper uses 3 Hz)

# Bandpass mode (used when FILTER_TYPE='bandpass')
BP_LOW = 10.0              # Hz lower bound
BP_HIGH = 200.0            # Hz upper bound

# --- Batch bandpass experiment mode ---
# If True, the single filter settings above are ignored and every band in the list below is processed
BATCH_MODE = True
BATCH_BANDS = [
    # === Non-overlapping sub-bands (answers "which band is most useful") ===
    (3, 10, "3-10 Hz (very low)"),
    (10, 50, "10-50 Hz (low)"),
    (50, 100, "50-100 Hz (mid)"),
    (100, 200, "100-200 Hz (mid-high)"),

    # === Composite bands (answers "does merging sub-bands help") ===
    (10, 200, "10-200 Hz (best so far)"),
    (10, 100, "10-100 Hz (hammer main)"),
]

# --- Waterfall plot zoom window ---
ZOOM_T_START = 3         # seconds
ZOOM_T_END   = 4.3        # seconds
ZOOM_R_START = 300.0         # meters
ZOOM_R_END   = 740         # meters

# --- Color range ---
COLORMAP = 'jet'
VRANGE_MODE = 'fixed'      # 'fixed' = Pierre paper's blue-background palette, 'auto' = data-adaptive (recommended for batch mode)
VMIN_FIXED = -np.pi        # Range used in Pierre's paper
VMAX_FIXED = 2 * np.pi     # Zero lands at 1/3 -> jet colormap renders a blue background

def load_extract_data(DATA_PATH):
    # ============================================================
    #  *** Processing logic below - generally no need to modify ***
    # ============================================================

    print("numpy version:", np.__version__)
    print(f"Data file: {DATA_PATH}")

    # %%
    # ============================================================
    # Step 1: Load data & extract parameters
    # ============================================================

    data = sio.loadmat(DATA_PATH)
    p = data['p']

    print("Variable list:")
    for key in data.keys():
        if not key.startswith('__'):
            val = data[key]
            print(f"  {key}: type={type(val).__name__}, shape={val.shape if hasattr(val, 'shape') else 'N/A'}")

    HiTab = p['HiTab'][0, 0]
    print(f"\nHiTab shape: {HiTab.shape}, dtype: {HiTab.dtype}")

    tx = p['tx'][0, 0]
    rx = p['rx'][0, 0]

    Tcode = tx['Tcode'][0, 0].item()
    fSamp = rx['fSamp'][0, 0].item()
    nbReflectors = rx['nbReflectors'][0, 0].item()
    nbDetectedCodes = rx['nbDetectedCodes'][0, 0].item()
    nbOvsReflectors = rx['nbOvsReflectors'][0, 0].item()
    f_cutoff = rx['f_cutoff'][0, 0].item()

    fibre = p['fibre'][0, 0]
    c_fiber = fibre['cFiber'][0, 0].item()

    spatial_res = 0.5 * (c_fiber / fSamp)
    Fe_DAS = 1 / Tcode

    print("=" * 50)
    print("        DAS System Key Parameters")
    print("=" * 50)
    print(f"  Tcode (pulse interval):   {Tcode*1000:.4f} ms")
    print(f"  Pulse repetition (Fe_DAS): {Fe_DAS:.0f} Hz")
    print(f"  fSamp (optical sampling): {fSamp/1e6:.0f} MHz")
    print(f"  Spatial resolution:       {spatial_res:.4f} m")
    print(f"  nbReflectors:             {nbReflectors}")
    print(f"  nbDetectedCodes:          {nbDetectedCodes}")
    print(f"  Nyquist frequency:        {Fe_DAS/2:.0f} Hz")

    # %%
    # ============================================================
    # Step 2: HiTab -> dual-polarization matrix
    # ============================================================

    Hx = HiTab[0, :].reshape(nbDetectedCodes, nbOvsReflectors).T
    Hy = HiTab[1, :].reshape(nbDetectedCodes, nbOvsReflectors).T
    print(f"\nHx shape: {Hx.shape} (space x time)")

    # %%
    # ============================================================
    # Step 3: Scattered light intensity
    # ============================================================

    intensity = np.mean(np.abs(Hx)**2 + np.abs(Hy)**2, axis=1)

    # plt.figure(figsize=(12, 4))
    # plt.plot(np.arange(len(intensity)) * spatial_res, intensity)
    # plt.xlabel('Fiber position (m)')
    # plt.ylabel('Mean backscatter intensity')
    # plt.title('Backscatter intensity per reflector')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # %%
    # ============================================================
    # Step 4: Channel selection
    # ============================================================

    nbBlocks = nbReflectors // LOW_RESOL_FACTOR
    usable = nbBlocks * LOW_RESOL_FACTOR

    intensity_first = np.abs(Hx[:, 0])**2 + np.abs(Hy[:, 0])**2
    intensity_blocks = intensity_first[:usable].reshape(nbBlocks, LOW_RESOL_FACTOR)
    max_in_block = np.argmax(intensity_blocks, axis=1)
    selected_indices = max_in_block + np.arange(nbBlocks) * LOW_RESOL_FACTOR

    print(f"\nChannel selection: {nbReflectors} reflectors -> {len(selected_indices)} channels")
    print(f"  Downsampling factor: {LOW_RESOL_FACTOR}")
    print(f"  Effective spacing:   {LOW_RESOL_FACTOR * spatial_res:.2f} m")

    # %%
    # ============================================================
    # Step 5: Differential phase computation (5a + 5b, no filtering)
    # ============================================================

    Hx_selected = Hx[selected_indices, :]
    Hy_selected = Hy[selected_indices, :]

    # 5a: Polarization-diversity conjugate product
    conj_product = (Hx_selected[:-1, :] * np.conj(Hx_selected[1:, :]) +
                    Hy_selected[:-1, :] * np.conj(Hy_selected[1:, :]))
    diffPhi = np.angle(conj_product)
    diffPhi = np.vstack([np.zeros((1, diffPhi.shape[1])), diffPhi])

    # 5b: Phase unwrapping (along time axis)
    diffPhi = np.unwrap(diffPhi, axis=1)

    print(f"\nDifferential phase matrix: {diffPhi.shape} (channels x time)")
    print(f"  Range after unwrap: [{diffPhi.min():.2f}, {diffPhi.max():.2f}] rad")

    # Save the unfiltered version (all subsequent filters branch from here)
    diffPhi_unwrapped = diffPhi.copy()

    # Build coordinate axes (once; shared by all filters)
    t_axis = np.arange(nbDetectedCodes) * Tcode
    r_axis = selected_indices * spatial_res

    print(f"  Time range:     {t_axis[0]:.4f} - {t_axis[-1]:.4f} s")
    print(f"  Distance range: {r_axis[0]:.2f} - {r_axis[-1]:.2f} m")

    return diffPhi_unwrapped, t_axis, r_axis, Fe_DAS, Tcode, spatial_res

# %%
# ============================================================
# Step 6: Filter + waterfall plot (core processing functions)
# ============================================================

def apply_filter(data, Fe, ftype, order, hp_cutoff=None, bp_low=None, bp_high=None):
    """
    Filter the differential phase matrix.
    Returns the filtered matrix and a description string.
    """
    nyq = Fe / 2

    # Data duration (used to check the lowest usable frequency)
    data_duration = data.shape[1] / Fe
    f_min_valid = 2.0 / data_duration  # Need at least 2 complete cycles

    if ftype == 'highpass':
        if hp_cutoff / nyq >= 1.0:
            raise ValueError(f"Highpass cutoff {hp_cutoff} Hz exceeds Nyquist {nyq} Hz")
        Wn = hp_cutoff / nyq
        sos = scipy.signal.butter(order, Wn, btype='high', output='sos')
        desc = f"Highpass {hp_cutoff} Hz"
    elif ftype == 'bandpass':
        f_hi = min(bp_high, nyq * 0.95)
        f_lo = max(bp_low, f_min_valid)  # Lower bound cannot be below what the data supports
        if f_lo >= f_hi:
            raise ValueError(f"Invalid band: lower {f_lo:.2f} Hz >= upper {f_hi:.0f} Hz "
                             f"(data duration {data_duration:.1f}s, min usable freq {f_min_valid:.2f} Hz)")
        if f_lo != bp_low:
            print(f"  [warn] Lower bound adjusted from {bp_low} Hz to {f_lo:.2f} Hz (data only {data_duration:.1f}s)")
        Wn = [f_lo / nyq, f_hi / nyq]
        sos = scipy.signal.butter(order, Wn, btype='band', output='sos')
        desc = f"Bandpass {f_lo:.1f}-{f_hi:.0f} Hz"
    else:
        raise ValueError(f"Unknown filter type: {ftype}")

    filtered = scipy.signal.sosfiltfilt(sos, data, axis=1)
    return filtered, desc


def plot_waterfall(diffPhi_filt, t_ax, r_ax, title_str, save_path=None,
                   zoom_t_start=ZOOM_T_START, zoom_t_end=ZOOM_T_END,
                   zoom_r_start=ZOOM_R_START, zoom_r_end=ZOOM_R_END):
    """
    Waterfall plot: left = full view, right = zoom on the specified window.
    """
    # Color range
    if VRANGE_MODE == 'auto':
        p99 = np.percentile(np.abs(diffPhi_filt), 99)
        if p99 < 1e-10:
            p99 = 1.0  # Prevent vmin = vmax = 0 on all-zero data
            print("  [warn] Band signal is nearly zero; using default color range")
        # Asymmetric range [-p99, +2*p99]: zero sits at 1/3 -> jet blue background (matches Pierre's paper)
        vmin = -p99
        vmax = 2 * p99
    else:
        vmin = VMIN_FIXED
        vmax = VMAX_FIXED

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: full view
    im1 = axes[0].imshow(
        diffPhi_filt, aspect='auto', origin='lower',
        extent=[t_ax[0], t_ax[-1], r_ax[0], r_ax[-1]],
        cmap=COLORMAP, vmin=vmin, vmax=vmax,
        interpolation='bilinear'
    )
    axes[0].set_xlabel('Time (s)', fontsize=13)
    axes[0].set_ylabel('Distance (m)', fontsize=13)
    axes[0].set_title('Full View', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='Phase (rad)')

    # Right: zoomed
    im2 = axes[1].imshow(
        diffPhi_filt, aspect='auto', origin='lower',
        extent=[t_ax[0], t_ax[-1], r_ax[0], r_ax[-1]],
        cmap=COLORMAP, vmin=vmin, vmax=vmax,
        interpolation='bilinear'
    )
    axes[1].set_xlim(zoom_t_start, zoom_t_end)
    axes[1].set_ylim(zoom_r_start, zoom_r_end)
    axes[1].set_xlabel('Time (s)', fontsize=13)
    axes[1].set_ylabel('Distance (m)', fontsize=13)
    axes[1].set_title('Zoomed View', fontsize=12)
    plt.colorbar(im2, ax=axes[1], label='Phase (rad)')

    fig.suptitle(title_str, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space at the top for suptitle

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved: {save_path}")

    # plt.show()


# %%
# ============================================================
# Step 7: Run filtering + plotting
# ============================================================
def save_raw_data(diffPhi, t_ax, r_ax, Fe, Tcode, spat_res, source_path):
 
    if not OUT_DIR_RAW:
        print("  save of raw ignored !!!.")
        return

    os.makedirs(OUT_DIR_RAW, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    
    # ==========================================
    # 1.save in npz
    # ==========================================
    npz_path = os.path.join(OUT_DIR_RAW, f"{base_name}_RAW.npz")
    np.savez(npz_path, 
             diffPhi_unwrapped=diffPhi, 
             t_axis=t_ax, 
             r_axis=r_ax, 
             Fe_DAS=Fe, 
             Tcode=Tcode, 
             spatial_res=spat_res)
    print(f"\n  raw data saved : {npz_path}")
    
    # ==========================================
    # 2. Save of the raw plot
    # ==========================================
    plot_path = os.path.join(OUT_DIR_RAW, f"waterfall_RAW_{base_name}.png")
    title = f"{os.path.basename(source_path)}\nraw_data(before_filtering))"
    plot_waterfall(diffPhi, t_ax, r_ax, title, save_path=plot_path)

def preprocessing():
    diffPhi_unwrapped, t_axis, r_axis, Fe_DAS, Tcode, spatial_res = load_extract_data(DATA_PATH)
    print(Tcode, diffPhi_unwrapped.shape)
    # sauvegarder en npz le waterfall brut donnée par load_extract_data(..) et sauvegarder le plot du waterfall avec plot_waterfall(..)
    save_raw_data(diffPhi_unwrapped, t_axis, r_axis, Fe_DAS, Tcode, spatial_res, DATA_PATH)
    if BATCH_MODE:
        # ========== Batch mode: process every band in turn ==========
        print(f"\n{'='*60}")
        print(f"  Batch mode: {len(BATCH_BANDS)} bands total")
        print(f"{'='*60}")

        for i, (f_lo, f_hi, band_label) in enumerate(BATCH_BANDS):
            print(f"\n[{i+1}/{len(BATCH_BANDS)}] {band_label}")

            try:
                diffPhi_bp, desc = apply_filter(
                    diffPhi_unwrapped, Fe_DAS,
                    ftype='bandpass', order=FILTER_ORDER,
                    bp_low=f_lo, bp_high=f_hi
                )
            except ValueError as e:
                print(f"  [skip] {e}")
                continue

            print(f"  Range after filter: [{diffPhi_bp.min():.4f}, {diffPhi_bp.max():.4f}]")

            # Safe filename
            safe_name = (band_label.replace(" ", "_").replace("-", "-")
                         .replace("(", "").replace(")", "").replace(".", "p"))
            save_path = os.path.join(OUT_DIR, f"waterfall_{safe_name}.png") if OUT_DIR else None

            title = f"{os.path.basename(DATA_PATH)}\n{band_label} (Butterworth order {FILTER_ORDER})"
            plot_waterfall(diffPhi_bp, t_axis, r_axis, title, save_path)

    else:
        # ========== Single mode: use the single filter configured at the top ==========
        print(f"\n{'='*60}")
        print(f"  Single mode: {FILTER_TYPE}")
        print(f"{'='*60}")

        if FILTER_TYPE == 'highpass':
            diffPhi_filtered, desc = apply_filter(
                diffPhi_unwrapped, Fe_DAS,
                ftype='highpass', order=FILTER_ORDER,
                hp_cutoff=HP_CUTOFF
            )
        else:
            diffPhi_filtered, desc = apply_filter(
                diffPhi_unwrapped, Fe_DAS,
                ftype='bandpass', order=FILTER_ORDER,
                bp_low=BP_LOW, bp_high=BP_HIGH
            )

        print(f"  {desc}")
        print(f"  Range after filter: [{diffPhi_filtered.min():.4f}, {diffPhi_filtered.max():.4f}]")

        save_path = None
        if OUT_DIR:
            safe_desc = desc.replace(' ', '_').replace('-', '-').replace('.', 'p')
            save_path = os.path.join(OUT_DIR, f"waterfall_{safe_desc}.png")

        title = f"{os.path.basename(DATA_PATH)}\n{desc} (Butterworth order {FILTER_ORDER})"
        plot_waterfall(diffPhi_filtered, t_axis, r_axis, title, save_path)

    print("\nDone.")


    # %%
