import idx2numpy
import numpy as np
from scipy import misc
import glob

png = []

# concatenate all of the image data you want to process into binary
# replace this path with dir of whatever images you want to convert to an array
for image_path in glob.glob("../sketchy/images/sketch/airplane/*.png"):
    png.append(misc.imread(image_path))

# convert to numpy ndarray
im = np.asarray(png)

# another way is to just use the destination file name
#idx2numpy.convert_to_file('test_airplane.idx', im)

# convert numpy array to idx binary format
# make sure to use 'wb' and not 'w' because python 3 will throw a fit (b for binary)
f_write = open('test_idx.idx', 'wb')
idx2numpy.convert_to_file(f_write, im)

