"""
The program will make an n x n collage of images.
If the number of images is not a sqrt of n, it will replicate the images and fill the collage.
"""

import cv2
import os
import numpy as np

def compute_grid_size(number_of_images):
    """
    return the grid size of the collage based on the number of images
    :param number_of_images: int number of total images
    :return: int grid
    """
    grid = 1
    # compute the grid size
    while True:
        if np.sqrt(number_of_images) <= grid:
            break
        grid += 1

    return grid

def is_image_file(file_path):
    """
    check if a path an image file
    :param file_path: string path to the image
    :return: bool True if an image else False
    """
    # Get the file extension from the file path
    _, file_extension = os.path.splitext(file_path)

    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    # Check if the file extension is in the list of image extensions
    return file_extension.lower() in image_extensions

def collage(input_path, output_path):
    """
    read square images in a folder and collage them on a fixed size canvas
    :param path: string - path to a folder
    :return: None
    """
    # define the canvas
    SIZE = (900, 900, 3)
    canvas = np.zeros(SIZE, dtype=np.uint8)

    # accessing the folder
    files = os.listdir(input_path)
    files.remove(('.DS_Store'))
    number_of_images = len(files)

    # compute the number of grid cells
    grid = compute_grid_size(number_of_images)

    # resize and paste the images onto the canvas
    cell_size = SIZE[0] // grid

    image_idx = 0

    for row in range(grid):
        row_idx = row * cell_size
        for col in range(grid):
            col_idx = col * cell_size

            # read images
            path = os.path.join(input_path, files[image_idx % number_of_images])
            if is_image_file(path):
                image = cv2.imread(path)

                # resize the image
                image = cv2.resize(image, (cell_size, cell_size))

                # paste the image on canvas
                canvas[row_idx: row_idx + cell_size, col_idx: col_idx + cell_size, :] = image

            image_idx += 1

    # save the image
    cv2.imwrite(output_path + '/collage.png', canvas)

    # display the image
    cv2.imshow('collage', canvas)

    # Wait for a key event and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    images_path = "../media /collection" # path to images
    output_path = "../media /collage" # path to save the output
    collage(images_path, output_path)

