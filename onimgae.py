import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Read the CSV file into a DataFrame
csv_file = '4.csv'
df = pd.read_csv(csv_file, header=None)


height=512
width=640
size=1

# Reshape the DataFrame into a 2D NumPy array
data = df.values.reshape((height, width))  # Replace height and width with your image dimensions

# Function to calculate the average of a 3x3 or 4x4 grid
def calculate_average(data, i, j, size=3):
    submatrix = data[i:i+size, j:j+size]
    return np.mean(submatrix)

# Create a new array with averaged pixel values
avg_data = np.zeros_like(data)

for i in range(0, height, size):
    for j in range(0, width, size):
        avg_data[i:i+size, j:j+size] = calculate_average(data, i, j, size)
        if (avg_data[i:i+size, j:j+size]>320):
            avg_data[i:i+size, j:j+size]=avg_data[i:i+size, j:j+size]
        else:
            avg_data[i:i+size, j:j+size]=avg_data[i:i+size, j:j+size]-30

# Create a figure and axes
fig, ax = plt.subplots()

# Load the image
# image_path = 'DJI_0403_T.jpg'  # Replace with the path to your image
# img = Image.open(image_path)

# Plot the image with transparency
# ax.imshow(img, extent=[0, width, 0, height], origin='upper', norm=Normalize(vmin=0, vmax=255), alpha=0.5)

# Plot the averaged data
ax.imshow(avg_data,extent=[0, width, 0, height], origin='upper', cmap='hot', alpha=1)

# Show the plot
plt.show()