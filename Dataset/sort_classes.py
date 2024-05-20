
# Read classes from unique_ab_pairs.txt and sort them alphabetically

import numpy as np

def read_ab_pairs(filename):
    ab_pairs = []
    with open(filename, 'r') as file:
        for line in file:
            clean_line = line.strip().replace('(', '').replace(')', '')
            a, b = map(float, clean_line.split(','))
            ab_pairs.append([a, b])
    return ab_pairs



ab_pairs = read_ab_pairs('Dataset/unique_ab_pairs.txt')

ab_values_sorted = sorted(ab_pairs, key=lambda x: np.sqrt(x[0]**2 + x[1]**2))

# Save the sorted classes to a file
with open('Dataset/ab_classes.txt', 'w') as file:
    for ab in ab_values_sorted:
        file.write(f'({ab[0]}, {ab[1]})\n')

print(ab_values_sorted)