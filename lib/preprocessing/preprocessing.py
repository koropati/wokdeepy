import cv2
import numpy as np
from .intersection_circle import find_intersection_circle


class Preprocessing:
    def __init__(self, input_path):
        self.input_path = input_path
        self.image = cv2.imread(input_path)
        self.ori_image = self.image
        self.img_height, self.img_width = self.image.shape[:2]

    def save_image(self, output_path):
        cv2.imwrite(output_path, self.image)

    def get_original_image(self):
        return self.ori_image

    def set_image(self, image_imput):
        self.image = image_imput

    def to_gray(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.image

    def to_binary(self, threshold=128):
        gray_image = self.to_gray()
        _, binary_image = cv2.threshold(
            gray_image, threshold, 1, cv2.THRESH_BINARY)
        return binary_image

    def to_binary_otsu(self, threshold=128):
        gray_image = self.to_gray()
        _, binary_image = cv2.threshold(
            gray_image, threshold, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_image

    def to_otsu(self):
        _, self.image = cv2.threshold(
            self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self.image

    def sharpening(self, imageInput, kernel=np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])):
        sharpened_image = cv2.filter2D(imageInput, -1, kernel)
        return sharpened_image

    def add_weight(self, imageInput, brightness=10, contrast=2.3):
        result = cv2.addWeighted(imageInput, contrast, np.zeros(
            imageInput.shape, imageInput.dtype), 0, brightness)
        return result

    def enhance_colour(self, imageInput, hue=0.7, saturnation=1.5, value=0.5):
        image = cv2.cvtColor(imageInput, cv2.COLOR_BGR2HSV)
        image[:, :, 0] = image[:, :, 0] * hue
        image[:, :, 1] = image[:, :, 1] * saturnation
        image[:, :, 2] = image[:, :, 2] * value

        result = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return result

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

    def blur_image(self, image_input, kSize=(7, 7)):
        self.image = cv2.blur(image_input, kSize)
        return self.image

    def median_blur(self, imageInput, kSize=(7, 7)):
        result = cv2.medianBlur(imageInput, kSize)
        return result

    def gaussian_blur(self, imageInput, kSize=(7, 7), sigmaX=0):
        result = cv2.GaussianBlur(imageInput, ksize=kSize, sigmaX=sigmaX)
        return result

    def hough_circles(self, image_input, dp, minDist, param1, param2, minRadius, maxRadius, maskingType=0):
        detected_circles = None
        cropped_image = None
        image_with_all_circle = self.ori_image

        while detected_circles is None or len(detected_circles[0]) < 1:
            detected_circles = cv2.HoughCircles(image_input,
                                                cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1,
                                                param2=param2, minRadius=minRadius, maxRadius=maxRadius)
            print("param1: ", param1, "param2: ", param2, "minRadius: ", minRadius, "maxRadius: ", maxRadius)
            print("Detected: ", detected_circles)
            if detected_circles is None or len(detected_circles[0]) < 1:
                # param1 += 5
                # param2 += 5
                minRadius += 1
                maxRadius += 1

        masking = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        masking_background = np.ones(
            (self.img_height, self.img_width, 3), dtype=np.uint8)
        if detected_circles is not None:
            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))

            for i in range(len(detected_circles[0])):
                x, y, rad = detected_circles[0][i][0], detected_circles[0][i][1], detected_circles[0][i][2]
                cv2.circle(image_with_all_circle, (x, y), rad, (0, 255, 0), 2)

            # resultCoordinate = self.find_intersecting_circle(detected_circles[0])
            resultCoordinate = find_intersection_circle(detected_circles[0])


            a, b, r = resultCoordinate[0], resultCoordinate[1], resultCoordinate[2]

            cv2.circle(self.image, (a, b), r, (0, 255, 0), 2)
            cv2.circle(masking, (a, b), r, 1, thickness=-1)
            cv2.circle(masking_background, (a, b), r, 0, thickness=-1)

            # Hitung koordinat titik awal (x1, y1) dan titik akhir (x2, y2) untuk crop
            x1, y1 = max(0, a - r), max(0, b - r)
            x2, y2 = min(
                self.ori_image.shape[1], a + r), min(self.ori_image.shape[0], b + r)
            mask_image = self.masking_area(self.ori_image, masking)
            if maskingType <= 0:
                cropped_image = mask_image[y1:y2, x1:x2]
            else:
                mask_image = cv2.add(mask_image, masking_background)
                cropped_image = mask_image[y1:y2, x1:x2]

        if maskingType <= 0:
            return self.image, image_with_all_circle, masking, cropped_image
        else:
            return self.image, image_with_all_circle, masking_background, cropped_image
    
    def crop_by_circle_coordinate(self, imageInput, coordinate):
        a, b, r = coordinate[0], coordinate[1], coordinate[2]
        x1, y1 = max(0, a - r), max(0, b - r)
        x2, y2 = min(imageInput.shape[1], a + r), min(imageInput.shape[0], b + r)
        cropped_image = imageInput[y1:y2, x1:x2]
        return cropped_image

    
    def get_hough_circles(self, image_input, dp, minDist, param1, param2, minRadius, maxRadius):
        detected_circles = None
        results = None

        while detected_circles is None or len(detected_circles[0]) < 1:
            detected_circles = cv2.HoughCircles(image_input,
                                                cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1,
                                                param2=param2, minRadius=minRadius, maxRadius=maxRadius)
            if detected_circles is None or len(detected_circles[0]) < 1:
                minRadius += 1
                maxRadius += 1

        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))

            results = detected_circles[0]
        return results


    def masking_area(self, original_image, masking_image):
        masked_image = cv2.bitwise_and(
            original_image, original_image, mask=masking_image)
        return masked_image

    def find_intersecting_circle(self, coordinates):
        max_intersection_count = 0
        result_circle = None

        if len(coordinates) == 1:
            x1, y1, r1 = coordinates[0]
            result_circle = (x1, y1, r1)
            result_circle
        else:
            for i in range(len(coordinates)):
                intersection_count = 0
                x1, y1, r1 = coordinates[i]

                for j in range(len(coordinates)):
                    if i != j:
                        x2, y2, r2 = coordinates[j]

                        distance = int(((np.float64(x2) - np.float64(x1)) **
                                       2 + (np.float64(y2) - np.float64(y1)) ** 2) ** 0.5)

                        if distance < r1 + r2:
                            intersection_count += 1

            if intersection_count > max_intersection_count:
                max_intersection_count = intersection_count
                result_circle = (x1, y1, r1)

        return result_circle

    def normalizeImage(self, image):
        normalized_image = image.astype(np.float32)
        min_val = np.min(normalized_image)
        max_val = np.max(normalized_image)
        normalized_image = (normalized_image - min_val) / (max_val - min_val)
        return (normalized_image * 255).astype(np.uint8)

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
