import argparse
import cv2
from lib import Preprocessing


def main():
    parser = argparse.ArgumentParser(
        description="Contoh aplikasi dengan argumen baris perintah")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        '-i', '--input', required=True, help="Path input")
    train_parser.add_argument(
        '-o', '--output', required=True, help="Path output")

    # Buat subparser untuk perintah "test"
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument('-i', '--input', required=True, help="Path input")
    test_parser.add_argument(
        '-o', '--output', required=True, help="Path output")

    playground_parser = subparsers.add_parser("playground")
    playground_parser.add_argument(
        '-i', '--input', required=True, help="Path input")
    playground_parser.add_argument(
        '-o', '--output', required=True, help="Path output")

    args = parser.parse_args()

    if args.command == "train":
        # train_model(args.input, args.output)
        print("hehhe")
    elif args.command == "test":
        print("hohohoh")
        # test_model(args.input, args.output)
    elif args.command == "playground":
        prep = Preprocessing(args.input)
        original_image = prep.to_binary(threshold=128)
        inverted_image = prep.invert_binary_image()
        cv2.imwrite("img\\test_out\\debug.jpg", inverted_image)
        data = prep.erode_binary_image(kernel_size=3, iteration=3)
        cv2.imwrite("img\\test_out\\debug2.jpg", data)
        data = prep.dilate_binary_image(kernel_size=3, iteration=4)
        cv2.imwrite("img\\test_out\\debug3.jpg", data)
        data = prep.erode_binary_image(kernel_size=3, iteration=7)
        cv2.imwrite("img\\test_out\\debug4.jpg", data)
        data = prep.dilate_binary_image(kernel_size=3, iteration=4)
        cv2.imwrite("img\\test_out\\debug5.jpg", data)
        data = prep.erode_binary_image(kernel_size=3, iteration=1)
        cv2.imwrite("img\\test_out\\debug6.jpg", data)
        data = prep.logical_and_binary_images(data, inverted_image)
        cv2.imwrite("img\\test_out\\debug7.jpg", data)
        data = prep.dilate_binary_image(kernel_size=3, iteration=8)
        cv2.imwrite("img\\test_out\\debug8.jpg", data)
        data = prep.erode_binary_image(kernel_size=3, iteration=1)
        cv2.imwrite("img\\test_out\\debug9.jpg", data)
        data = prep.to_otsu()
        cv2.imwrite("img\\test_out\\debug10.jpg", data)
        x, y, w, h = prep.find_largest_contour_coordinates()
        prep.get_sub_image(original_image, x, y, w, h)
        prep.resize(395, 75)
        prep.convert_edge_to_black_with_thickness(8)
        prep.get_higher_text_line_segmentation()
        chars = prep.char_column_segmentation(20, 20)

        for i, image in enumerate(chars):
            filename = args.output + f'image_segment_{i}.jpg'
            cv2.imwrite(filename, prep.normalizeImage(image))


if __name__ == "__main__":
    main()
