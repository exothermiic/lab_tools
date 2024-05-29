import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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
freq1, s21_db_filter1, s12_db_filter1 = filter1_data[:, 0], filter1_data[:, 3], filter1_data[:, 5]
freq2, s21_db_filter2, s12_db_filter2 = filter2_data[:, 0], filter2_data[:, 3], filter2_data[:, 5]

# Find common frequencies
common_freqs = np.intersect1d(freq1, freq2)

# Extract data at common frequencies
indices1 = np.isin(freq1, common_freqs)
indices2 = np.isin(freq2, common_freqs)

common_s21_db_filter1 = s21_db_filter1[indices1]
common_s21_db_filter2 = s21_db_filter2[indices2]


# Combine the insertion losses (in dB) for cascading 2 of filter #1 and 2 of filter #2
combined_insertion_loss = 2 * common_s21_db_filter1 + 2 * common_s21_db_filter2

# Plot the combined insertion loss
plt.figure(figsize=(10, 6))
plt.plot(common_freqs, combined_insertion_loss, label='Combined Insertion Loss (2 x Filter #1 + 2 x Filter #2)')
plt.title('Combined Insertion Loss of Cascaded Filters')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Insertion Loss (dB)')
plt.legend()
plt.grid(True)
plt.show()

# Extract phase data
phase_S21_filter1 = filter1_data[:, 4]
phase_S21_filter2 = filter2_data[:, 4]


# Extract data at common frequencies for phase
common_phase_S21_filter1 = phase_S21_filter1[indices1]
common_phase_S21_filter2 = phase_S21_filter2[indices2]


# Unwrap phase data
phase_S21_filter1_unwrapped = np.unwrap(np.deg2rad(common_phase_S21_filter1))
phase_S21_filter2_unwrapped = np.unwrap(np.deg2rad(common_phase_S21_filter2))

# Sum the phase responses
combined_phase_S21_unwrapped = phase_S21_filter1_unwrapped + phase_S21_filter2_unwrapped

# Convert back to degrees
combined_phase_S21_deg = np.rad2deg(combined_phase_S21_unwrapped)

# Smooth the combined phase response (if necessary)
smoothed_combined_phase_S21 = savgol_filter(combined_phase_S21_deg, window_length=11, polyorder=3)

# Compute group delay
group_delay_S21 = np.diff(smoothed_combined_phase_S21) / np.diff(common_freqs)

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

plt.tight_layout()
plt.show()
