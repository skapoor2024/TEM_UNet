import cv2
import numpy as np
import sys
import os

# Check if directory name and file name are provided as command line arguments
if len(sys.argv) != 3:
    print("Please provide the directory name and file name as command line arguments.")
    print("Usage: python script.py <directory_name> <file_name>")
    sys.exit(1)

# Get directory name and file name from command line arguments
directory_name = sys.argv[1]
file_name = sys.argv[2]

# Validate the directory
if not os.path.isdir(directory_name):
    print(f"The directory '{directory_name}' does not exist.")
    sys.exit(1)

# Set the image file path
img_path = os.path.join(directory_name, file_name)

# Initialize variables
red_points = []

# Mouse callback function
def mark_red_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:  # Left mouse button double-clicked
        red_points.append((x, y))  # Store coordinates in the list
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)  # Draw a red circle at the clicked point
        cv2.imshow('Image', img)  # Display the modified image

    elif event == cv2.EVENT_RBUTTONDBLCLK:  # Right mouse button double-clicked
        points_to_remove = []
        for point in red_points:
            px, py = point
            if abs(px - x) <= 5 and abs(py - y) <= 5:  # Check if the clicked point is close to any marked point
                points_to_remove.append(point)

        for point in points_to_remove:
            red_points.remove(point)  # Remove the points from the list
            px, py = point
            cv2.circle(img, (px, py), 3, (0, 0, 0), -1)  # Draw a black circle to remove the points
        cv2.imshow('Image', img)  # Display the modified image

# Read the input image
img = cv2.imread(img_path)

# Create a window and bind the mouse callback function
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mark_red_points)

while True:
    cv2.imshow('Image', img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()

# Convert the list of coordinates to a NumPy array
red_points_arr = np.array(red_points)

# Save the array as .npy file in the given directory with the given file name
output_file_path = os.path.join(directory_name, file_name[:-4] + '.npy')
np.save(output_file_path, red_points_arr)