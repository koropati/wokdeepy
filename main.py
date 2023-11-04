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
        # result = prep.get_original_image()
        # result = prep.enhance_colour(result, hue=1.1, saturnation=1.5, value=0.9)
        # result = prep.gaussian_blur(result,kSize=(7,7), sigmaX=2.5)
        # result = prep.sharpening(result)
        result = prep.to_binary(128)
        data, image_with_all_circle, masking, result = prep.hough_circles(result, 2, 2, 1.0, 26, 190, 200, maskingType=1)
        # result = prep.enhance_colour(result)
        # result = prep.sharpening(result)
        # prep.set_image(result)
        # result = prep.to_binary(128)

        cv2.imwrite(args.output, prep.normalizeImage(image_with_all_circle))


        
        # chars = prep.water_meter_segmentation(128, 20,20)
        # for i, image in enumerate(chars):
        #     filename = args.output + f'image_segment_{i}.jpg'
        #     cv2.imwrite(filename, prep.normalizeImage(image))


if __name__ == "__main__":
    main()
