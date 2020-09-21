import os
import re
import argparse
import requests
from math import floor
from DigiPathUtils import GoogleMapsTile
from DigiPathDb import DigiPathDbWrapper

SERVER_URL = 'http://129.25.13.204'
WEBAPP_ADDIMAGE_ENDPOINT = 'https://senior-design-pathology.herokuapp.com/addImage'

CROP_SIZE = 50
LOAD_ADJ_TILES = True   # pre-load adjacent tiles to avoid black border from clipping near edges

def parseArgs():
    ''' 
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputFile',  '-f',   default='/home/qpproj/centroid_positions.txt', help='Path to file containing centroid values')
    parser.add_argument('--outputDir',  '-d',   default='/home/images/cells', help='Directory to output cell images to')
    args = parser.parse_args()
    return args

def strToInt(val):
    '''
    Takes a string representation of a number and returns an int. Floats are truncated.
    '''
    return int(floor(float(val)))

def readCellLocations(filename):
    ''' 
    Read x,y locations from file for each tile image.
    Return a map from GoogleMapsTile to list of cell locations
    '''
    cell_locations = {}
    with open(filename) as f:  
        tile = None
        for line in f:
            if line[0] == "/":
                # indicates that information for a new file will be read in
                fname = line.strip()
                tile = GoogleMapsTile(fname, loadAdjTiles=LOAD_ADJ_TILES, immediateLoad=False)
                cell_locations[tile] = set() # use a set to avoid duplicates
            else:
                x, y = re.split(r'\t+', line.strip())
                loc = (strToInt(x), strToInt(y))
                cell_locations[tile].add(loc)
    return cell_locations

def cropCells(cellLocations, cell_id, dir_name):
    '''
    For each tile, loop through the list of cell locations and crop out each cell.
    Save each cell as an individual file in dir_name.
    Return a list of new cell files that were created, along with their ID, tile path, slide, and pixel coordinates.
    '''
    newCells = []
    for tile, locs in cellLocations.items():
        tile.load() # load tile
        for loc in locs:
            ext = tile.getExtension()
            
            base_dir = os.path.join(dir_name, tile.slide_name, str(tile.tileRow), str(tile.tileCol))
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
                
            new_filename = os.path.join(base_dir, str(cell_id) + ext)
            cropped_image = tile.cropCell(loc, CROP_SIZE)
            cropped_image.save(new_filename)

            cell = {
                'cell_id': cell_id,
                'path': new_filename,
                'slide_name': tile.slide_name,
                'tile_path': tile.filename,
                'pixel_row': loc[1],
                'pixel_col': loc[0]
            }
            newCells.append(cell)
            cell_id += 1
        tile.unload() # unload tile to release memory
    return newCells


def main():
    args = parseArgs()
    
    input_file_name = args.inputFile
    dir_name = args.outputDir

    # create output directory if needed
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with DigiPathDbWrapper() as db:
        # get starting cell id
        startingCellId = db.get_next_cell_ID()

        # read cell locations and crop each one into a file
        cellLocations = readCellLocations(input_file_name)
        newCells = cropCells(cellLocations, startingCellId, dir_name)

        numCells = len(newCells)
        print("Successfully created {} images in directory '{}'".format(numCells, dir_name))

        # publish new cells
        for cell in newCells:
            db.new_cell(cell['path'], cell['slide_name'], cell['tile_path'], cell['pixel_row'], cell['pixel_col'])

if __name__ == "__main__":
    main()