import numpy as np
import matplotlib.pyplot as plt

# Function to read S-parameter data from a file with the specified encoding
def read_sparameter_data(filename):
    data = []
    with open(filename, 'r', encoding='latin-1') as file:
        for line in file:
            if not line.startswith('!') and not line.startswith('#'):
                values = line.split()
                if len(values) == 9:  # Ensure the line has the correct number of columns
                    freq, s11_db, s11_phase, s21_db, s21_phase, s12_db, s12_phase, s22_db, s22_phase = map(float, values)
                    data.append((freq, s11_db, s11_phase, s21_db, s21_phase, s12_db, s12_phase, s22_db, s22_phase))
    return np.array(data)

# Read the S-parameter data for both filters
filter1_data = read_sparameter_data('VHF-2000+___Plus25degC.S2P')
filter2_data = read_sparameter_data('VLF-5000+___Plus25degC.S2P')

# Extract frequency and S21 insertion loss data
freq1, s21_db_filter1 = filter1_data[:, 0], filter1_data[:, 3]
freq2, s21_db_filter2 = filter2_data[:, 0], filter2_data[:, 3]

# Find common frequencies
common_freqs = np.intersect1d(freq1, freq2)

# Extract data at common frequencies
indices1 = np.isin(freq1, common_freqs)
indices2 = np.isin(freq2, common_freqs)

common_s21_db_filter1 = s21_db_filter1[indices1]
common_s21_db_filter2 = s21_db_filter2[indices2]

# Number of each filter to stack
N1 = 1  # Number of filter #1 to stack
N2 = 1  # Number of filter #2 to stack

# Combine the insertion losses (in dB) for stacking N1 of filter #1 and N2 of filter #2
combined_insertion_loss = N1 * common_s21_db_filter1 + N2 * common_s21_db_filter2

# Find the -3 dB points
threshold = -3
dB_below_threshold = combined_insertion_loss - threshold
crossing_points = np.where(np.diff(np.sign(dB_below_threshold)))[0]


# Plot the combined insertion loss
plt.figure(figsize=(10, 6))
plt.plot(common_freqs, combined_insertion_loss, label=f'Combined Insertion Loss ({N1} x Filter #1 + {N2} x Filter #2)')
plt.title('Combined Insertion Loss of Cascaded Filters')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Insertion Loss (dB)')
plt.legend()
plt.grid(True)

# Annotate the -3 dB points
for idx in crossing_points:
    freq_label = common_freqs[idx]
    loss_label = combined_insertion_loss[idx]
    plt.scatter(freq_label, loss_label, color='red', s=10, edgecolors='black', zorder=5)
    plt.annotate(f'{freq_label:.1f} MHz', xy=(freq_label, loss_label), xytext=(freq_label + 10, loss_label + 1),
                 fontsize=9, ha='center')
    
plt.show()

# Extract phase data
phase_S21_filter1 = filter1_data[:, 4]
phase_S21_filter2 = filter2_data[:, 4]

phase_S12_filter1 = filter1_data[:, 6]
phase_S12_filter2 = filter2_data[:, 6]

# Extract data at common frequencies for phase
common_phase_S21_filter1 = phase_S21_filter1[indices1]
common_phase_S21_filter2 = phase_S21_filter2[indices2]

common_phase_S12_filter1 = phase_S12_filter1[indices1]
common_phase_S12_filter2 = phase_S12_filter2[indices2]

# Unwrap phase data
phase_S21_filter1_unwrapped = np.unwrap(np.deg2rad(common_phase_S21_filter1))
phase_S21_filter2_unwrapped = np.unwrap(np.deg2rad(common_phase_S21_filter2))

phase_S12_filter1_unwrapped = np.unwrap(np.deg2rad(common_phase_S12_filter1))
phase_S12_filter2_unwrapped = np.unwrap(np.deg2rad(common_phase_S12_filter2))

# Sum the phase responses
combined_phase_S21_unwrapped = N1 * phase_S21_filter1_unwrapped + N2 * phase_S21_filter2_unwrapped
combined_phase_S12_unwrapped = N1 * phase_S12_filter1_unwrapped + N2 * phase_S12_filter2_unwrapped

# Convert back to degrees
combined_phase_S21_deg = np.rad2deg(combined_phase_S21_unwrapped)
combined_phase_S12_deg = np.rad2deg(combined_phase_S12_unwrapped)

# Compute group delay
delta_freqs = np.diff(common_freqs) * 1e6  # Convert MHz to Hz for angular frequency
group_delay_S21 = -np.diff(combined_phase_S21_deg) / delta_freqs  # ns
group_delay_S12 = -np.diff(combined_phase_S12_deg) / delta_freqs  # ns

# Plot the results
plt.figure(figsize=(10, 12))

plt.subplot(3, 1, 1)
plt.plot(common_freqs, combined_phase_S21_deg, label='Combined Phase S21')
plt.title('Combined Phase Response (S21)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Phase (degrees)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(common_freqs[:-1], group_delay_S21, label='Group Delay S21')
plt.title('Group Delay (S21)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Group Delay (ns)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(common_freqs, combined_phase_S12_deg, label='Combined Phase S12')
plt.title('Combined Phase Response (S12)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Phase (degrees)')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(common_freqs[:-1], group_delay_S12, label='Group Delay S12')
plt.title('Group Delay (S12)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Group Delay (ns)')
plt.legend()

plt.tight_layout()
plt.show()
