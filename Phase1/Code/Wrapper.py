#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import os
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def gaussian_kernel(size, sigma):
    axis = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def rotate_sobel(angle):
    """Rotates the Sobel filter to the specified angle."""
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    rotated = rotate(sobel_x, angle, reshape=False, order=1, mode='constant', cval=0)
    return rotated

def img_pad(image, pad_height, pad_width):
     image_height, image_width = image.shape
     padded_image = np.zeros((image_height + 2 * pad_height, image_width + 2 * pad_width), dtype=image.dtype)
     padded_image[pad_height:pad_height+image_height,pad_width:pad_width+image_width] = image
     return padded_image

def convolution(image, kernel):
	kernel_height, kernel_width = kernel.shape
	image_height, image_width = image.shape
	output = np.zeros((image_height, image_width), dtype=image.dtype)
	padding_height, padding_width = kernel_height // 2, kernel_width // 2
	
	padded_image = img_pad(image, padding_height, padding_width)
	# padded_image = np.pad(image, padding_height, mode='constant', constant_values=0)
	
	for i in range(image_height):
		for j in range(image_width):     
			output[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)  
	return output

def DoG_Bank(sigma_values, angles):
    gaussian_kernel_size = 13
    gaussian_filter_bank = []

    for sigma in sigma_values:
        gaussian = gaussian_kernel(gaussian_kernel_size, sigma)

        for angle in angles:
            rotated_sobel = rotate_sobel(angle)
            convolved_sobel = convolution(gaussian, rotated_sobel)
            normalized = cv2.normalize(convolved_sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            resized = cv2.resize(normalized, (48, 48), interpolation=cv2.INTER_LINEAR)
            gaussian_filter_bank.append(resized)
    
    return gaussian_filter_bank

def create_filter_grid(
    filters, rows, cols, filter_size, output_path="Filters/", filename=None,
    padding=5, normalize=True, border_color=255
):
    
    """Creates a grid image for the various filters passed."""
    
    os.makedirs(output_path, exist_ok=True)

    # Calculate grid dimensions
    grid_height = rows * filter_size + (rows - 1) * padding
    grid_width = cols * filter_size + (cols - 1) * padding
    grid = np.full((grid_height, grid_width), border_color, dtype=np.uint8)

    for i, filter_image in enumerate(filters):
        if i >= rows * cols:  # Stop if more filters than grid cells
            break

        # Normalize the filter to [0, 255] if requested
        if normalize:
            filter_image = (filter_image - filter_image.min()) / (filter_image.max() - filter_image.min())
            filter_image = (filter_image * 255).astype(np.uint8)

        # Resize filter to the specified size
        if filter_image.shape[0] != filter_size:
            filter_image = cv2.resize(filter_image, (filter_size, filter_size), interpolation=cv2.INTER_CUBIC)

        # Calculate position in the grid
        row, col = divmod(i, cols)
        y_start = row * (filter_size + padding)
        x_start = col * (filter_size + padding)

        # Place the filter in the grid
        grid[y_start:y_start + filter_size, x_start:x_start + filter_size] = filter_image

    # Determine the full save path
    save_path = os.path.join(output_path, filename) if filename else output_path

    # Save the grid as an image
    cv2.imwrite(save_path, grid)

    return grid

def first_derivative_gaussian(size, sigma, orientation):
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    x, y = np.meshgrid(ax, ax)
    theta = np.deg2rad(orientation)
    
    # Apply elongation factor (σy = 3σx)
    sigma_y = 3 * sigma
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    
    # Use different sigma for x and y directions
    gauss = np.exp(-(x_rot**2 / (2 * sigma**2) + y_rot**2 / (2 * sigma_y**2)))
    derivative = -x_rot * gauss / (sigma**2)
    
    # Normalize
    return derivative / np.abs(derivative).sum()

def second_derivative_gaussian(size, sigma, orientation):
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    x, y = np.meshgrid(ax, ax)
    theta = np.deg2rad(orientation)
    
    # Apply elongation factor (σy = 3σx)
    sigma_y = 3 * sigma
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    
    # Use different sigma for x and y directions
    gauss = np.exp(-(x_rot**2 / (2 * sigma**2) + y_rot**2 / (2 * sigma_y**2)))
    derivative = (x_rot**2 / (sigma**4) - 1 / (sigma**2)) * gauss
    
    # Normalize
    return derivative / np.abs(derivative).sum()

def laplacian_of_gaussian(size, sigma):

    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    x, y = np.meshgrid(ax, ax)
    return ((x**2 + y**2 - 2 * sigma**2) / sigma**4) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

def gaussian_filter(size, sigma):

    return cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T

def normalize_filter(filter_image):

    filter_norm = (filter_image - filter_image.min()) / (filter_image.max() - filter_image.min())
    return (filter_norm * 255).astype(np.uint8)

def hd_normalize_filter(filter_image):
    return (filter_image * 255).astype(np.uint8)

def LM_bank():
		# Define parameters for LMS (LM Small)
	scales = [1, np.sqrt(2), 2, 2*np.sqrt(2)]  # Basic scales for LMS
	orientations = np.linspace(0, 180, 6, endpoint=False)  # 6 orientations
	size = 49  # Size of filters
	lms_filters_bank = []

	# Generate first and second derivative filters
	for sigma in scales[:3]:  # First three scales only
		for orientation in orientations:
			# First derivative
			lms_filters_bank.append(first_derivative_gaussian(size, sigma, orientation))
			
		for orientation in orientations:
			# Second derivative
			lms_filters_bank.append(second_derivative_gaussian(size, sigma, orientation))


		# Generate Laplacian of Gaussian filters
	for sigma in [1, np.sqrt(2), 2, 2 * np.sqrt(2)]:
		lms_filters_bank.append(laplacian_of_gaussian(size, sigma))      # At σ

	for sigma in [1, np.sqrt(2), 2, 2 * np.sqrt(2)]:
		lms_filters_bank.append(laplacian_of_gaussian(size, 3 * sigma))

	# Generate Gaussian smoothing filters
	for sigma in [1, np.sqrt(2), 2, 2 * np.sqrt(2)]:
		lms_filters_bank.append(gaussian_filter(size, sigma))

	# Verify we have 48 filters
	assert len(lms_filters_bank) == 48, f"Generated {len(lms_filters_bank)} filters instead of 48"

	scales = [np.sqrt(2), 2, 2*np.sqrt(2),4]  # Basic scales for LMS
	orientations = np.linspace(0, 180, 6, endpoint=False)  # 6 orientations
	size = 49  # Size of filters
	lml_filters_bank = []

	for sigma in scales[:3]:  # First three scales only
		for orientation in orientations:
			# First derivative
			lml_filters_bank.append(first_derivative_gaussian(size, sigma, orientation))
			
		for orientation in orientations:
			# Second derivative
			lml_filters_bank.append(second_derivative_gaussian(size, sigma, orientation))

	# print(f"Derivative:{len(lml_filters_bank)}")


	# Generate Laplacian of Gaussian filters
	for sigma in scales:
		lml_filters_bank.append(laplacian_of_gaussian(size, sigma))      # At σ

	# print(f"σ Laplcian Gaussian:{len(lml_filters_bank)}")

	for sigma in scales:
		lml_filters_bank.append(laplacian_of_gaussian(size, 3 * sigma))

	# print(f"3*σ Laplcian Gaussian:{len(lml_filters_bank)}")

	# Generate Gaussian smoothing filters
	for sigma in scales:
		lml_filters_bank.append(gaussian_filter(size, sigma))


		# Verify we have 48 filters
	assert len(lml_filters_bank) == 48, f"Generated {len(lml_filters_bank)} filters instead of 48"
	
	return lms_filters_bank, lms_filters_bank

def gabor_filter(size, sigma, orientation, omega):
    """Generates a Gabor filter."""
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    x, y = np.meshgrid(ax, ax)
    theta = np.deg2rad(orientation)

    # Rotate coordinates
    c, s = np.cos(theta), np.sin(theta)
    x_rotated = x * c + y * s
    y_rotated = -x * s + y * c

    # Gaussian and sinusoidal components
    gaussian = np.exp(-(x_rotated**2 / (2 * sigma**2) + y_rotated**2 / (2 * sigma**2)))
    sine = np.sin(omega * x_rotated)

    # Combine to create Gabor filter
    gabor = gaussian * sine
    return gabor

def Gabor_bank():
	orientation_range = np.linspace(0, 180, 8, endpoint=False)  # 8 orientations evenly spaced
	sigma_range = [6, 7, 9, 11, 13]  # More compact Gaussian spreads
	omega_range = [0.3, 0.4, 0.5, 0.6, 0.7]  # Sinusoidal frequencies
	kernel_size = 37  # Keep filters compact

	gabor_filters_bank = []

	# Generate Gabor filters
	for i, sigma in enumerate(sigma_range):
		for orientation in orientation_range:
			gabor_filters_bank.append(gabor_filter(kernel_size, sigma, orientation, omega_range[i]))


	return gabor_filters_bank

def half_disk_mask(radius):
    """Generates a half-disk mask."""
    half_disk = np.zeros((radius * 2, radius * 2), dtype=np.uint8)  # Correct initialization
    for i in range(radius * 2):
        for j in range(radius * 2):
            if (i - radius)**2 + (j - radius)**2 < radius**2 and i <= radius:
                half_disk[i, j] = 1
    return half_disk

def create_binary_image_vectorized(img, num_bins):

    binary_img = np.array([(img == bin_value).astype(np.uint8) for bin_value in range(1, num_bins)])
    
    return binary_img

def gradient(property_map, num_bins, mask_left, mask_right):
    """
    Calculate the gradient map using chi-square between left and right masks.
    """
    print("Calculating gradient map")
    # Create binary images for all bins
    binary_images = create_binary_image_vectorized(property_map, num_bins)
    # binary_images = np.array([(property_map == bin_value).astype(np.uint8) for bin_value in range(1, num_bins)])

    
    # Initialize gradient map for 24 orientations
    gradient_map = np.zeros((property_map.shape[0], property_map.shape[1], 24))
    
    for m in range(24):
        chi_squared = np.zeros(property_map.shape)
        
        # Apply masks and compute chi-square for all bins
        for binary_image in binary_images:
            # conv_left = cv2.filter2D(binary_image, -1, mask_left[m])
            # conv_right = cv2.filter2D(binary_image, -1, mask_right[m])
            conv_left = cv2.filter2D(binary_image.astype(np.float32), -1, mask_left[m].astype(np.float32))
            conv_right = cv2.filter2D(binary_image.astype(np.float32), -1, mask_right[m].astype(np.float32))
            denominator = conv_left + conv_right + 1e-6  # Avoid division by zero
            chi_squared += ((conv_left - conv_right) ** 2) / denominator
        
        gradient_map[:, :, m] = chi_squared
    
    return gradient_map

def plot(img,output_folder,filename=None, cmap=None):

    os.makedirs(output_folder, exist_ok=True)

    # Display the image
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.show()

    # Save the image if a filename is provided
    if filename:
        save_path = os.path.join(output_folder, filename)
        plt.imsave(save_path, img, cmap=cmap)


def main():

    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """
        
    gaussian_orientations = range(0, 360, 15)
    gaussian_scales = [1.0, 1.5]

    DoG = DoG_Bank(gaussian_scales, gaussian_orientations)

    create_filter_grid(DoG, rows= len(gaussian_scales), cols=len(gaussian_orientations), filter_size=48, filename="DoG.png")
    print("Gaussian Filter Created")

    """
    Generate Leung-Malik Filter Bank: (LM)
    Display all the filters in this filter bank and save image as LM.png,
    use command "cv2.imwrite(...)"
    """

    LMS, LML = LM_bank()

    create_filter_grid(LMS, rows=4, cols=12, filter_size=64, filename="LMS.png")
    create_filter_grid(LML, rows=4, cols=12, filter_size=64, filename="LML.png")

    print("LM Filters Created")

    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """

    Gabor = Gabor_bank()
    create_filter_grid(Gabor, rows=5, cols=8, filter_size=64, filename="Gabor.png")
    print("Gabor Filter Created")

    # Creating filter's bank
    filters_banks = []
    for i in range(len(DoG)):
        if len(DoG[i].shape) == 2:
            filters_banks.append(DoG[i])

    for i in range(len(LMS)):
        if len(LMS[i].shape) == 2:
            filters_banks.append(LMS[i])

    for i in range(len(LML)):
        if len(LML[i].shape) == 2:
            filters_banks.append(LML[i])

    for i in range(len(Gabor)):
        if len(Gabor[i].shape) == 2:
            filters_banks.append(Gabor[i])

    print("Filters Banks Combined")

    """
    Generate Half-disk masks
    Display all the Half-disk masks and save image as HDMasks.png,
    use command "cv2.imwrite(...)"
    """

    hd_orients = np.arange(0,360,360/8)
    radius_sizes = np.array([5,10,15])

    left_mask = []
    right_mask = []
    masks = []

    for i in range(len(radius_sizes)):
        hd_mask = half_disk_mask(radius_sizes[i])
        for k in range(len(hd_orients)):
            mask_1 = rotate(hd_mask, hd_orients[k])
            left_mask.append(mask_1)
            masks.append(mask_1)
            mask_2 = rotate(mask_1, 180)
            right_mask.append(mask_2)
            masks.append(mask_2)

    masks = [hd_normalize_filter(mask) for mask in masks]

    create_filter_grid(filters=masks, rows= 6, cols= 8, filter_size=64, filename="HDMasks.png")
    print("Half Discs Mask Created")

    folder_path = "/home/sarthak_m/ComputerVision/HW0_Alohomora/YourDirectoryID_hw0/Phase1/BSDS500/Images"

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        original_img = cv2.imread(file_path)
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        # Initialize data correctly
        data = np.zeros((gray_img.size, len(filters_banks)))

        """
        Generate Texton Map
        Filter image using oriented gaussian filter bank
        """
        print(f"----------------Started process for:{filename}----------------")
        # Loop through the filters and compute responses
        for i in range(len(filters_banks)):
            temp = cv2.filter2D(gray_img, -1, filters_banks[i])  # Apply the filter
            temp = temp.reshape((1, gray_img.size))
            data[:, i] = temp
            # data[:, i] = temp.flatten()  # Assign flattened filter response to the i-th column

        print("Finished convoluting with filter banks")

        """
        Generate texture ID's using K-means clustering
        Display texton map and save image as TextonMap_ImageName.png,
        use command "cv2.imwrite('...)"
        """

        print("Texton Map")
        texton_kmeans = KMeans(n_clusters= 64, n_init=4, random_state=42).fit(data)
        texton_map = texton_kmeans.labels_.reshape(gray_img.shape)

        plot(texton_map, output_folder="Texton_Maps",filename=f"texton_map_{filename}", cmap="viridis")


        """
        Generate Texton Gradient (Tg)
        Perform Chi-square calculation on Texton Map
        Display Tg and save image as Tg_ImageName.png,
        use command "cv2.imwrite(...)"
        """

        print("Texton Gradient")

        Tg = gradient(texton_map, 64, left_mask, right_mask)
        print("Texton Gradient Computation Complete")

        tgm = np.mean(Tg, axis=2)

        plot(tgm, output_folder="Texton_Gradient_Maps", filename=f"texton_gradient_{filename}", cmap="viridis")


        """
        Generate Brightness Map
        Perform brightness binning 
        """

        print("Brightness Map")

        flattened_gray_img = gray_img.flatten().reshape(-1, 1)
        brightness_kmeans = KMeans(n_clusters=16, n_init=4, random_state=42).fit(flattened_gray_img)
        brightness_map = brightness_kmeans.labels_.reshape(gray_img.shape)

        plot(brightness_map, output_folder="Brightness_Maps", filename=f"brightness_map_{filename}", cmap="viridis")


        """
        Generate Brightness Gradient (Bg)
        Perform Chi-square calculation on Brightness Map
        Display Bg and save image as Bg_ImageName.png,
        use command "cv2.imwrite(...)"
        """

        print("Brightness Gradient")

        Bg = gradient(brightness_map, 16, left_mask, right_mask)
        print("Brightness Gradient Computation Complete")

        bgm = np.mean(Bg, axis=2)

        plot(bgm, output_folder="Brightness_Gradient_Maps", filename=f"brightness_gradient_{filename}", cmap="viridis")


        """
        Generate Color Map
        Perform color binning or clustering
        """

        print("Color Map")

        # Flatten the image into a 2D array
        img_flattened = original_img.reshape(-1, 3)
        color_kmeans = KMeans(n_clusters=16, n_init=4, random_state=42).fit(img_flattened)
        color_map = color_kmeans.labels_.reshape(original_img.shape[:2])

        plot(color_map, output_folder="Color_Maps", filename=f"color_map_{filename}", cmap="viridis")


        """
        Generate Color Gradient (Cg)
        Perform Chi-square calculation on Color Map
        Display Cg and save image as Cg_ImageName.png,
        use command "cv2.imwrite(...)"
        """

        print("Color Gradient")
        Cg = gradient(color_map, 16, left_mask, right_mask)
        print("Color gradient computation complete")

        cgm = np.mean(Cg, axis=2)

        plot(cgm, output_folder="Color_Gradient_Maps", filename=f"color_gradient_{filename}", cmap="viridis")


        """
        Read Sobel Baseline
        use command "cv2.imread(...)"
        """

        baseline_path = "/home/sarthak_m/ComputerVision/HW0_Alohomora/YourDirectoryID_hw0/Phase1/BSDS500"
        filename_no_ext, _ = os.path.splitext(filename)
        sobel_baseline = baseline_path + "/SobelBaseline/" + filename_no_ext + ".png"
        print(sobel_baseline)


        """
        Read Canny Baseline
        use command "cv2.imread(...)"
        """

        canny_baseline = baseline_path + "/CannyBaseline/" + filename_no_ext + ".png"
        print(canny_baseline)

        """
        Combine responses to get pb-lite output
        Display PbLite and save image as PbLite_ImageName.png
        use command "cv2.imwrite(...)"
        """

        print("PB Lite")
        sobel_img = cv2.imread(sobel_baseline)
        canny_img = cv2.imread(canny_baseline)
        sobel_gray = cv2.cvtColor(sobel_img, cv2.COLOR_BGR2GRAY)
        canny_gray = cv2.cvtColor(canny_img, cv2.COLOR_BGR2GRAY)

        w = 0.5
        avg = (tgm + bgm + cgm)/3
        cs = w*canny_gray + (1-w)*sobel_gray
        pb = np.multiply(avg, cs)

        plot(pb, output_folder="Pb_Lite_Output", filename=f"Pb_Lite_{filename}", cmap="gray")
    
if __name__ == '__main__':
    main()
 


