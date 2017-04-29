from PIL import Image
import os, sys

path = 'sketch/'
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((64,64), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG', quality=100)

def invert():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            inverted_image = PIL.ImageOps.invert(im)
            inverted_image.save(f + '.png', 'PNG', quality=100)

#resize()

#invert()
