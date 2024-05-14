import numpy as np

def detect_spikes(data, threshold):
    """Detect spikes in the data."""
    spikes = np.where(data > threshold)[0]
    return spikes

def detect_troughs(data, threshold):
    """Detect troughs in the data."""
    troughs = np.where(data < threshold)[0]
    return troughs

def detect_spikes_followed_by_troughs(data, spike_threshold, trough_threshold):
    """Detect spikes followed by troughs."""
    spikes = detect_spikes(data, spike_threshold)
    troughs = detect_troughs(data, trough_threshold)
    patterns = []
    for spike in spikes:
        for trough in troughs:
            if trough > spike:
                patterns.append((spike, trough))
                break
    return patterns

def detect_flat_lines(data, threshold):
    """Detect flat lines in the data."""
    flat_lines = []
    i = 0
    while i < len(data):
        start = i
        while i < len(data) and abs(data[i] - data[start]) < threshold:
            i += 1
        if i - start > 1:
            flat_lines.append((start, i-1))
        i += 1
    return flat_lines

# Example data
data = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 1, 0])

# Detect spikes
spikes = detect_spikes(data, 2)
print("Spikes:", spikes)

# Detect troughs
troughs = detect_troughs(data, 1)
print("Troughs:", troughs)

# Detect spikes followed by troughs
patterns = detect_spikes_followed_by_troughs(data, 2, 1)
print("Spikes followed by troughs:", patterns)

# Detect flat lines
flat_lines = detect_flat_lines(data, 0.5)
print("Flat lines:", flat_lines)