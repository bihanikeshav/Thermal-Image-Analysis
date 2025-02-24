import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Folder containing CSV files
folder_path = 'E:\Downloads\LT_651-20240430T055423Z-001\LT_651'  # Replace with the path to your folder
# Set the hotspot threshold
hotspot_threshold = 33  # Adjust the threshold as needed

# Iterate through CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        print("reading")
        # Read the CSV file into a DataFrame
        csv_path = os.path.join(folder_path, filename)
        df = pd.read_csv(csv_path, header=None)
        # Reshape the DataFrame into a 2D NumPy array
        data = df.values

        # Check if there are hotspots in the image
        max=np.max(data)
        if max >= 0:
            # Create a figure and axes
            fig, ax = plt.subplots()

            # Plot the image using the "jet" colormap
            im = ax.imshow(data, cmap='jet', extent=[0, data.shape[1], 0, data.shape[0]], origin='upper')

            # Add a colorbar for each image
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Temperature')

            # Show the plot for each image
            plt.title(f'Image with Hotspots: {filename}, Max temp: {round(max, 2)}°C')
            plt.savefig('Res_'+filename.split('.')[0]+'.jpg')
            # plt.show()  
            plt.close()