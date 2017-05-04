import idx2numpy
import numpy as np
from scipy import misc
import glob
import os


labels_to_ints = {
    'airplane': 0,
    'alarm_clock': 1,
    'ant': 2,
    'ape': 3,
    'apple': 4,
    'armor': 5,
    'axe': 6,
    'banana': 7,
    'bat': 8,
    'bear': 9,
    'bee': 10,
    'beetle': 11,
    'bell': 12,
    'bench': 13,
    'bicycle': 14,
    'blimp': 15,
    'bread': 16,
    'butterfly': 17,
    'cabin': 18,
    'camel': 19,
    'candle': 20,
    'cannon': 21,
    'car_(sedan)': 22,
    'castle': 23,
    'cat': 24,
    'chair': 25,
    'chicken': 26,
    'church': 27,
    'couch': 28,
    'cow': 29,
    'crab': 30,
    'crocodilian': 31,
    'cup': 32,
    'deer': 33,
    'dog': 34,
    'dolphin': 35,
    'door': 36,
    'duck': 37,
    'elephant': 38,
    'eyeglasses': 39,
    'fan': 40,
    'fish': 41,
    'flower': 42,
    'frog': 43,
    'geyser': 44,
    'giraffe': 45,
    'guitar': 46,
    'hamburger': 47,
    'hammer': 48,
    'harp': 49,
    'hat': 50,
    'hedgehog': 51,
    'helicopter': 52,
    'hermit_crab': 53,
    'horse': 54,
    'hot-air_balloon': 55,
    'hotdog': 56,
    'hourglass': 57,
    'jack-o-lantern': 58,
    'jellyfish': 59,
    'kangaroo': 60,
    'knife': 61,
    'lion': 62,
    'lizard': 63,
    'lobster': 64,
    'motorcycle': 65,
    'mouse': 66,
    'mushroom': 67,
    'owl': 68,
    'parrot': 69,
    'pear': 70,
    'penguin': 71,
    'piano': 72,
    'pickup_truck': 73,
    'pig': 74,
    'pineapple': 75,
    'pistol': 76,
    'pizza': 77,
    'pretzel': 78,
    'rabbit': 79,
    'raccoon': 80,
    'racket': 81,
    'ray': 82,
    'rhinoceros': 83,
    'rifle': 84,
    'rocket': 85,
    'sailboat': 86,
    'saw': 87,
    'saxophone': 88,
    'scissors': 89,
    'scorpion': 90,
    'sea_turtle': 91,
    'seagull': 92,
    'seal': 93,
    'shark': 94,
    'sheep': 95,
    'shoe': 96,
    'skyscraper': 97,
    'snail': 98,
    'snake': 99,
    'songbird': 100,
    'spider': 101,
    'spoon': 102,
    'squirrel': 103,
    'starfish': 104,
    'strawberry': 105,
    'swan': 106,
    'sword': 107,
    'table': 108,
    'tank': 109,
    'teapot': 110,
    'teddy_bear': 111,
    'tiger': 112,
    'tree': 113,
    'trumpet': 114,
    'turtle': 115,
    'umbrella': 116,
    'violin': 117,
    'volcano': 118,
    'wading_bird': 119,
    'wheelchair': 120,
    'windmill': 121,
    'window': 122,
    'wine_bottle': 123,
    'zebra': 124
}

png = []
labels = []

# concatenate all of the image data you want to process into binary
# replace this path with dir of whatever images you want to convert to an array

for subdir, dirs, files in os.walk("../sketchy/images/sketch"):
    for file in files:
        #print(subdir.split(os.path.sep)[-1])
        #print(os.path.join(subdir, file))
        png.append(misc.imread(os.path.join(subdir, file))) # concatenate image data
        labels.append(labels_to_ints[subdir.split(os.path.sep)[-1]]) # concatenate label based on dir
'''
for image_path in glob.glob("../sketchy/images/sketch/airplane/*.png"):
#for image_path in glob.glob("sketches/*.png"):
    png.append(misc.imread(image_path))
'''

# convert to numpy ndarray
im = np.asarray(png)
labs = np.asarray(labels)

print('images and labels read into numpy array')

# another way is to just use the destination file name
#idx2numpy.convert_to_file('test_airplane.idx', im)

# convert numpy array to idx binary format
# make sure to use 'wb' and not 'w' because python 3 will throw a fit (b for binary)
i_write = open('train_images.idx', 'wb')
idx2numpy.convert_to_file(i_write, im)
i_write.close()

print('image idx binary written')

l_write = open('train_labels.idx', 'wb')
idx2numpy.convert_to_file(l_write, labs)
l_write.close()

print('label idx binary written')

# following code for reading idx files into readable text
'''
#f_read = open('train-images-idx3-ubyte', 'rb')
f_read = open('test_idx.idx', 'rb')
ndarr = idx2numpy.convert_from_file(f_read)

s = idx2numpy.convert_to_string(ndarr)
f_write = open('train_images_text.txt', 'w')

f_write.write(str(s))
f_write.close()
print('finished writing images')

f_read = open('train-labels.idx1-ubyte', 'rb')
ndarr = idx2numpy.convert_from_file(f_read)

s = idx2numpy.convert_to_string(ndarr)
f_write2 = open('train_labels_text.txt', 'w')
f_write2.write(str(s))
f_write2.close()
print('finished writing labels')
'''
