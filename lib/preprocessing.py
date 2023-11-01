import cv2
import numpy as np


class Preprocessing:
    def __init__(self, input_path):
        self.input_path = input_path
        self.image = cv2.imread(input_path)

    def save_image(self, output_path):
        cv2.imwrite(output_path, self.image)

    def to_gray(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.image

    def to_binary(self, threshold=128):
        gray_image = self.to_gray()
        _, binary_image = cv2.threshold(
            gray_image, threshold, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_image
    
    def to_otsu(self):
        _, self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self.image

    def invert_binary_image(self):
        self.image = cv2.bitwise_not(self.image)
        return self.image

    def erode_binary_image(self, kernel_size=3, iteration=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.image = cv2.erode(self.image, kernel, iterations=iteration)
        return self.image

    def dilate_binary_image(self, kernel_size=3, iteration=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.image = cv2.dilate(self.image, kernel, iterations=iteration)
        return self.image

    def logical_and_binary_images(self, binary_image1, binary_image2):
        self.image = cv2.bitwise_and(binary_image1, binary_image2)
        return self.image

    def find_largest_contour_coordinates(self):
        # Temukan kontur dalam citra biner
        contours, _ = cv2.findContours(
            self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        # Temukan kontur terbesar berdasarkan luas
        largest_contour = max(contours, key=cv2.contourArea)
        # Dapatkan koordinat kotak pembatas
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h

    def get_sub_image(self, imgae_input, x, y, w, h):
        self.image = imgae_input[y:y+h, x:x+w]
        return self.image

    def resize(self, width, height):
        self.image = cv2.resize(self.image, (width, height))
        return self.image

    def convert_edge_to_black_with_thickness(self, thickness):
        rows, cols = self.image.shape

        for y in range(rows):
            for x in range(cols):
                if (
                    x < thickness or x >= cols - thickness or y < thickness or y >= rows - thickness
                ):
                    self.image[y, x] = 0

        return self.image

    def get_higher_text_line_segmentation(self):
        if not isinstance(self.image, np.ndarray) or self.image.size == 0 or self.image.ndim != 2:
            raise ValueError(
                'Invalid array format. Expected a 2D binary array.')

        lines = self.image
        is_text_line = False
        current_line = []

        max_area = 0

        for row in self.image:
            is_text_line = 1 in row

            if is_text_line:
                current_line.append(list(row))
            elif current_line:
                if len(current_line) * len(current_line[0]) > max_area:
                    max_area = len(current_line) * len(current_line[0])
                    lines = current_line
                current_line = []

        if current_line:
            if len(current_line) * len(current_line[0]) > max_area:
                max_area = len(current_line) * len(current_line[0])
                lines = current_line

        self.image = np.array(lines, dtype=np.uint8)

        return self.image

    def char_column_segmentation(self, char_width, char_height):
        if not isinstance(self.image, np.ndarray) or self.image.size == 0 or self.image.ndim != 2:
            raise ValueError(
                'Invalid array format. Expected a 2D binary array.')

        columns = []
        current_column = []

        for col in range(self.image.shape[1]):
            if 1 in self.image[:, col]:
                current_column.append(self.image[:, col])
            elif current_column:
                current_column = np.array(current_column).T  # Rotasi CCW
                # current_column = np.fliplr(current_column)  # Flip horizontal

                # Resize kolom karakter ke ukuran yang diinginkan
                current_column = cv2.resize(
                    current_column, (char_width, char_height))

                columns.append(current_column)
                current_column = []

        if current_column:
            current_column = np.array(current_column).T  # Rotasi CCW
            # current_column = np.fliplr(current_column)  # Flip horizontal

            # Resize kolom karakter terakhir ke ukuran yang diinginkan
            current_column = cv2.resize(
                current_column, (char_width, char_height))

            columns.append(current_column)

        return columns
    
    def normalizeImage(self, image):
        return (image * 255).astype(np.uint8)
    
    def water_meter_segmentation(self, threshold, char_width, char_height):
        original_image = self.to_binary(threshold=threshold)
        inverted_image = self.invert_binary_image()
        data = self.erode_binary_image(kernel_size=3, iteration=3)
        data = self.dilate_binary_image(kernel_size=3, iteration=4)
        data = self.erode_binary_image(kernel_size=3, iteration=7)
        data = self.dilate_binary_image(kernel_size=3, iteration=4)
        data = self.erode_binary_image(kernel_size=3, iteration=1)
        data = self.logical_and_binary_images(data, inverted_image)
        data = self.dilate_binary_image(kernel_size=3, iteration=8)
        data = self.erode_binary_image(kernel_size=3, iteration=1)
        data = self.to_otsu()

        x, y, w, h = self.find_largest_contour_coordinates()
        self.get_sub_image(original_image, x, y, w, h)
        self.resize(395, 75)
        self.convert_edge_to_black_with_thickness(8)
        self.get_higher_text_line_segmentation()
        chars = self.char_column_segmentation(char_width, char_height)

        return chars


