import numpy as np
import cv2
import argparse

from numpy.ma.core import where


def simple_convert(img):
    img_std = np.std(img, axis=(0, 1))
    img_mean = np.mean(img, axis=(0, 1))

    col_bits = 8 * img_std / img_std.sum()
    col_bits_int = (col_bits * 2 + 1) // 2

    print("R: {} bits, G: {} bits, B: {} bits".format(int(col_bits_int[2]), int(col_bits_int[1]), int(col_bits_int[0])))

    col_interval = 256 / (2 ** col_bits_int)
    dot_img = img / col_interval
    dot_img = (dot_img * 2 + 1) // 2
    dot_img = dot_img * col_interval - 1
    dot_img[dot_img < 0] = 0

    return dot_img

def statistical_convert(img):
    pass


def Dotcam():
    capture = cv2.VideoCapture(-1)
    while(True):
        ret, frame = capture.read()
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


def main(args):
    if args.camera == 1:
        Dotcam()
    else:
        img = cv2.imread(args.fig)
        dot = simple_convert(img)

        cv2.imwrite(args.save, dot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert RGB each 8bit piccture to Dot picture")
    parser.add_argument("--fig", default="Demos/Lenna_(test_image).png")
    parser.add_argument("--save", default="Demos/outputs.png")
    parser.add_argument("--camera", default=0, type=int)
    args = parser.parse_args()
    main(args)
