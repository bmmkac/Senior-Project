import os
from PIL import Image
from openslide import OpenSlide, open_slide

def save_img(slide, col, row, delta, dir, size=None, lvl=0, JPEG=True):
    width, height = slide.level_dimensions[lvl]
    if row < 0 or row + delta > height:
        return None
    if col < 0 or col + delta > width:
        return None
    
    try:
        filename = '%d_%d.%s' % ( col, row, 'jpg' if JPEG else 'png')
        filename = os.path.join(dir, filename)

        img = slide.read_region((col,row), lvl, (delta, delta))
        
        if(size): # resize
            img.thumbnail(size, Image.ANTIALIAS)
        if(JPEG): # convert for JPG
            img = img.convert('RGB')

        img.save(filename)
        return filename
    except:
        return None


def decompose_file(filename, delta, begin=[0,0], n=[None,None], out_dir='./', JPEG=True, size=None):
    slide = open_slide(filename)
    end = list(slide.dimensions)
    
    # update end coordinates if number of subsamples was specified
    if(n[0]):
        end[0] = begin[0] + n[0]*delta
    if(n[1]):
        end[1] = begin[1] + n[1]*delta
    
    file = os.path.splitext(os.path.basename(filename))[0]
    dir = os.path.join(out_dir, file, str(delta))
    if(not os.path.isdir(dir)):
        os.makedirs(dir)
    
    # save image subsamples over the region defined by (begin[0], begin[1]) to (end[0], end[1])
    for c in range(begin[0], end[0], delta):
        for r in range(begin[1], end[1], delta):
            imgfile = save_img(
                slide=slide,
                col=c,
                row=r, 
                delta=delta, 
                dir=dir,
                JPEG=JPEG,
                size=size
                )
            if(imgfile is None): # error saving sub-image
                print('Failed: c=%d, r=%d' % (c, r))
                slide = open_slide(filename) # re-open in case of exception
    
    slide.close()


class Adj():
    '''
    Enum used for referencing adjacent tiles
    '''
    TOP_LEFT = 0
    TOP = 1
    TOP_RIGHT = 2
    LEFT = 3
    CENTER = 4
    RIGHT = 5
    BOTTOM_LEFT = 6
    BOTTOM = 7
    BOTTOM_RIGHT = 8

class GoogleMapsTile():
    '''
    Class for working with a GoogleMapsTile.
    Supports loading a tile and cropping cell images from it.
    Optionally, you can load all of the adjacent tiles around it to get a single, larger image.
    '''
    TILE_SIZE = 256

    def __init__(self, filename, loadAdjTiles=False, immediateLoad=True):
        '''
        Initialize a GoogleMapsTile object.
        @param filename - path to the tile image
        @param loadAdjTiles - if True, loads all of the adjacent tiles to form a single image.
        @param immediateLoad - if True, loads the image immediately. Otherwise, you must call load() to load the image. 
        '''
        self.filename = filename
        self.img = None
        self.loadAdjTiles = loadAdjTiles

        # get the row and column numbers of the tile
        self.tileRow = int(os.path.basename(os.path.dirname(self.filename)))
        self.tileCol = int(os.path.splitext(os.path.basename(self.filename))[0])

        # get the root zoom directory for the tile
        self.zoomDir = os.path.dirname(os.path.dirname(self.filename))

        # get the slide name for the tile
        self.slide_name = os.path.basename(os.path.dirname(self.zoomDir))

        if(immediateLoad): # load image
            self.load()
    
    def __eq__(self, other):
        return hasattr(other, 'filename') and self.filename == other.filename

    def __hash__(self):
        return hash(self.filename)

    def getExtension(self):
        '''
        Get the file extension for the tile image (typically '.jpg')
        '''
        return os.path.splitext(self.filename)[1]

    def load(self):
        '''
        Load the tile image, optionally with all adjacent tiles
        '''
        if(self.loadAdjTiles):
            self.loadTileWithAdjacent()
        else:
            self.img = Image.open(self.filename)

    def unload(self):
        '''
        Release image
        '''
        self.img = None

    def joinTile(self, pos, filename):
        '''
        Join a tile into part of a larger image. For use by loadTileWithAdjacent().
        @param pos - represents the position to load the subimage into
        @param filename - subimage to load
        '''
        dx = pos % 3
        dy = pos // 3

        im = Image.open(filename)
        self.img.paste(im, (dx*self.TILE_SIZE, dy*self.TILE_SIZE))
        im.close()

    def loadTileWithAdjacent(self):
        '''
        Loads tile with all adjacents tiles to form a single, larger image
        '''
        ext = self.getExtension()

        self.img = Image.new('RGB', (3*self.TILE_SIZE, 3*self.TILE_SIZE))
        
        self.joinTile(Adj.TOP_LEFT,     os.path.join(self.zoomDir, '%d/%d%s' % (self.tileRow-1, self.tileCol-1, ext)))
        self.joinTile(Adj.TOP,          os.path.join(self.zoomDir, '%d/%d%s' % (self.tileRow-1, self.tileCol,   ext)))
        self.joinTile(Adj.TOP_RIGHT,    os.path.join(self.zoomDir, '%d/%d%s' % (self.tileRow-1, self.tileCol+1, ext)))
        self.joinTile(Adj.LEFT,         os.path.join(self.zoomDir, '%d/%d%s' % (self.tileRow,   self.tileCol-1, ext)))
        self.joinTile(Adj.CENTER,       self.filename)
        self.joinTile(Adj.RIGHT,        os.path.join(self.zoomDir, '%d/%d%s' % (self.tileRow,   self.tileCol+1, ext)))
        self.joinTile(Adj.BOTTOM_LEFT,  os.path.join(self.zoomDir, '%d/%d%s' % (self.tileRow+1, self.tileCol-1, ext)))
        self.joinTile(Adj.BOTTOM,       os.path.join(self.zoomDir, '%d/%d%s' % (self.tileRow+1, self.tileCol,   ext)))
        self.joinTile(Adj.BOTTOM_RIGHT, os.path.join(self.zoomDir, '%d/%d%s' % (self.tileRow+1, self.tileCol+1, ext)))

    def cropCell(self, center, size):
        '''
        Return a cropped portion of the tile given the center and size.
        '''
        x, y = center
        if(self.loadAdjTiles):
            x += self.TILE_SIZE
            y += self.TILE_SIZE

        crop_area = ((x - size // 2), (y - size // 2), (x + size // 2),
            (y + size // 2))

        return self.img.crop(crop_area)
