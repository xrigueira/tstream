import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

iteration = 0

# Read weights data
weights = np.load('results/all_sa_encoder_weights.npy', allow_pickle=True, fix_imports=True)

# Subset the last row of the weights
weights = weights[iteration][0][-1]

# Split the data
days, weeks, months, years = weights[-30:], weights[-42:-30], weights[-50:-42], weights[-53:-50]

# Repeat the elements
weeks_repeated, months_repeated, years_repeated = np.repeat(weeks, 8), np.repeat(months, 30), np.repeat(years, 365)

# Concatenate all the arrays
result = np.concatenate((years_repeated, months_repeated, weeks_repeated, days))

# Load the src
src = np.load(f'results/src_p_{iteration}.npy', allow_pickle=True, fix_imports=True)

# Load the tgt_p
tgt_p = np.load(f'results/tgt_p_{iteration}.npy', allow_pickle=True, fix_imports=True)

# Load the tgt_y_hat
tgt_y_hat = np.load(f'results/tgt_y_hat_{iteration}.npy', allow_pickle=True, fix_imports=True)

# Sample data for bars
data_bar = weights
categories = range(-1461, 1)

# Sample data for lines (assuming you have data for all three lines)
data_line1 = src
data_line2 = tgt_p

# Create the plot
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# Plot bars on primary axes
bars = ax1.bar(categories, data_bar, color='coral')

# Invert the y-axis for bars
ax1.invert_yaxis() 

# Plot lines on secondary axes
ax2.plot(categories, data_line1, color='blue', marker='o', label='Line 1')  # Add label for clarity
ax2.plot(categories, data_line2, color='green', marker='s', label='Line 2')  # Adjust marker and label
# ax2.plot(categories, data_line3, color='purple', marker='^', label='Line 3')  # Adjust marker and label

# # Set labels and title
# ax1.set_xlabel('Categories')
# ax1.set_ylabel('Bar Values', color='coral')
# ax2.set_ylabel('Line Values', color='blue')
# plt.title('Inverted Bar Chart with Two Y-Axes and Multiple Lines')

# # Additional customization (optional)
# ax2.tick_params('y', colors='blue')  # Set color for right y-axis ticks
# lines1, labels1 = ax2.get_legend_handles_labels()  # Get lines and labels for legend
# ax1.legend(lines1, labels1, loc='upper left')  # Add legend to primary axes

# # Show the plot
# plt.tight_layout()
# plt.show()

