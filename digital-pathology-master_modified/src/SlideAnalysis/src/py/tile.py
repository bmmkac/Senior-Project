import os
import skimage.io
import cv2
import numpy
from PIL import Image

class Tile(object):
    def __init__(self, level, address, tile_size, overlap, outdir, format):
        self._level = level
        self._address = address
        self._tile_size = tile_size
        self._overlap = overlap
        self._format = format
        
        self._coordinates = ( (tile_size-overlap) * address[0], (tile_size - overlap) * address[1] )
        self._filename = os.path.join(outdir, '{}_{}.{}'.format(
            self._coordinates[0], self._coordinates[1], self._format))

    def get_filename(self):
        return self._filename

    def get_coordinates(self):
        return self._coordinates

    def get_image_as_skimage(self):
        img = skimage.io.imread(self._filename)[:, :, :3]
        return img
    
    def get_image_as_nparray(self):
        img = cv2.imread(self._filename)
        return img

    def get_image_as_PILImage(self):
        img = Image.open(self._filename)
        return img

    def cleanup(self):
        os.remove(self._filename)