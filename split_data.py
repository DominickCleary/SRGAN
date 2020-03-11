import argparse
import os.path
import shutil
import random

parser = argparse.ArgumentParser(description='Split testing and training data')
parser.add_argument('--images_dir', type=str,
                    help='location of images')
parser.add_argument('--output_dir', type=str,
                    help='location of images')


if __name__ == '__main__':
    opt = parser.parse_args()
    IMAGES_DIR = opt.images_dir
    OUTPUT_DIR = opt.output_dir

files = [file for file in os.listdir(
    IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, file))]

for x in range(425):
    selection = random.randint(0, len(files)-1)
    file = files.pop(selection)
    shutil.move(os.path.join(IMAGES_DIR, file), OUTPUT_DIR)
