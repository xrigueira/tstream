import matplotlib.pyplot as plt

# Sample data for bars
data_bar = [5, 8, 2, 10]
categories = ["A", "B", "C", "D"]

# Sample data for lines (assuming you have data for all three lines)
data_line1 = [3, 6, 1, 7]
data_line2 = [2, 4, 5, 6]
data_line3 = [1, 3, 2, 5]

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
ax2.plot(categories, data_line3, color='purple', marker='^', label='Line 3')  # Adjust marker and label

# Set labels and title
ax1.set_xlabel('Categories')
ax1.set_ylabel('Bar Values', color='coral')
ax2.set_ylabel('Line Values', color='blue')
plt.title('Inverted Bar Chart with Two Y-Axes and Multiple Lines')

# Additional customization (optional)
ax2.tick_params('y', colors='blue')  # Set color for right y-axis ticks
lines1, labels1 = ax2.get_legend_handles_labels()  # Get lines and labels for legend
ax1.legend(lines1, labels1, loc='upper left')  # Add legend to primary axes

# Show the plot
plt.tight_layout()
plt.show()

