import numpy as np
import cv2
import argparse


def statistical_convert(img):
    img_std = np.std(img, axis=-1)
    img_mean = np.mean(img, axis=-1)

    b_bits = 8 * (img_std[0] / img_std.sum())
    g_bits = 8 * (img_std[0] / img_std.sum())

    return img


def main(args):
    img = cv2.imread(args.fig)
    dot = statistical_convert(img)

    cv2.imwrite(args.save, dot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert RGB each 8bit piccture to Dot picture")
    parser.add_argument("--fig", default="")
    parser.add_argument("--save", default="outputs.png")
    args = parser.parse_args()
    main(args)
