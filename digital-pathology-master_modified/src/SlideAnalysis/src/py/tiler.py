###############################################################################
# Creator: Joe Urbano
# Date: 3/27/2019
# Project: Senior Design, Digital Pathology
# File: tiler.py
# Purpose: The tiler module breaks down a whole slide image into tiles
#          and processes each tile individually. The tiler runs multiple
#          processes to process the tiles concurrently.
# Associated Files: Associated files located at gitrepo/test/SlideAnalysis/
#                   testtiler.py - the unit tests for tiler.py
###############################################################################

import os
import sys
import shutil
import PIL
import argparse
import json
from multiprocessing import Process, JoinableQueue
from openslide import OpenSlide, open_slide, OpenSlideError
from openslide.deepzoom import DeepZoomGenerator

from tile import Tile

class TileWorker(Process):
    '''A child process that generates and writes tiles.'''

    def __init__(self, tiler_queue, processor_queue, slidepath, tile_size, overlap, limit_bounds, format):
        '''
        TileWorker Constructor - initialize a TileWorker process
        Params: 
            tiler_queue: (JoinableQueue), the queue of tiles to be read from the slide
            processor_queue: (JoinableQueue), the queue of tiles ready to be processed
            slidepath: (str), path to the slide file
            tile_size: (int), the size of each tile in pixels
            overlap: (int), the amount of overlap between tiles in pixels
            limit_bounds: (bool), limits the slide boundaries
            format: (str), image format ('jpeg' or 'png')
        '''
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._tiler_queue = tiler_queue
        self._processor_queue = processor_queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._format = format
        self._slide = None

    def run(self):
        '''
        Start process execution
        '''
        self._slide = open_slide(self._slidepath)
        dz = self._get_dz()
        
        while True:
            try:
                data = self._tiler_queue.get()
                if data is None: # no more tiles to process, break out of loop
                    self._tiler_queue.task_done()
                    break
                
                level, address, outdir = data

                tile = Tile(
                    level=level,
                    address=address,
                    tile_size=self._tile_size,
                    overlap=self._overlap,
                    outdir=outdir,
                    format=self._format
                )

                if not os.path.exists(tile.get_filename()):
                    dz_tile = dz.get_tile(level, address)
                    dz_tile.save(tile.get_filename())

                if self._processor_queue:
                    self._processor_queue.put(tile)

                self._tiler_queue.task_done()
            except OpenSlideError as e:
                # on OpenSlideErrors, the slide object is corrupted
                # and needs to be reinitialized
                self._slide = open_slide(self._slidepath)
                dz = self._get_dz()
                print(e)
                self._tiler_queue.task_done()

    def _get_dz(self):
        '''
        Create and return an openslide.DeepZoomGenerator object
        '''
        return DeepZoomGenerator(
            self._slide, 
            self._tile_size, 
            self._overlap,
            limit_bounds=self._limit_bounds
        )


class DeepZoomImageTiler(object):
    '''Handles generation of tiles and metadata for a single image.'''

    def __init__(self, dz, basename, format, tile_size, queue):
        '''
        DeepZoomImageTiler Constructor
        Params: 
            dz: (DeepZoomGenerator), deep zoom generator object for the slide
            basename: (string), the basename of the output directory
            format: (str), image format ('jpeg' or 'png')
            tile_size: (int), the size of each tile in pixels
            queue: (JoinableQueue), the queue of tiles to be processed
        '''
        self._dz = dz
        self._basename = basename
        self._format = format
        self._tile_size = tile_size
        self._queue = queue
        self._processed = 0

    def run(self):
        '''
        Run the tiler to queue up tiles to be processed
        '''
        self._write_tiles()

    def _write_tiles(self):
        '''
        Queue up each tile to be processed
        '''
        level = self._dz.level_count - 1
        tiledir = os.path.join(self._basename, str(self._tile_size))
        if not os.path.exists(tiledir):
            os.makedirs(tiledir)
        cols, rows = self._dz.level_tiles[level]
        for row in range(rows):
            for col in range(cols):
                self._queue.put((level, (col, row), tiledir))
                self._tile_done()

    def _tile_done(self):
        '''
        Postprocessing to run after each tile is queued up
        '''
        self._processed += 1
        count = self._processed

        level = self._dz.level_count - 1
        num_tiles = self._dz.level_tiles[level]
        total = num_tiles[0] * num_tiles[1]
        if count % 100 == 0 or count == total:
            print("Tiling slide: wrote %d/%d tiles" % (count, total),
                    end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)


class Tiler(object):
    '''Handles generation of tiles and metadata for all images in a slide.'''

    def __init__(self, slidepath, basename, format, tile_size, overlap,
                limit_bounds, workers, tile_processor_queue=None):
        '''
        Tiler Constructor
        Params: 
            slidepath: (str), path to the slide file
            basename: (string), the basename of the output directory
            format: (str), image format ('jpeg' or 'png')
            tile_size: (int), the size of each tile in pixels
            overlap: (int), the amount of overlap between tiles in pixels
            limit_bounds: (bool), limits the slide boundaries
            workers: (int), number of TileWorker processes to run
            tile_processor_queue: (JoinableQueue) queue of tiles ready to
                be processed
        '''
        self._slide = open_slide(slidepath)
        self._basename = os.path.join(basename, os.path.splitext(os.path.basename(slidepath))[0])
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._worker_procs = [None] * workers

        # initialize TileWorker processes
        for _i in range(workers):
            self._worker_procs[_i] = TileWorker(
                tiler_queue=self._queue, 
                processor_queue=tile_processor_queue,
                slidepath=slidepath, 
                tile_size=tile_size, 
                overlap=overlap, 
                limit_bounds=limit_bounds,
                format=format
            )
            self._worker_procs[_i].start()

    def run(self):
        '''
        Run the tiler
        '''
        self._run_image()
        self._shutdown()

    def _run_image(self):
        '''
        Tile a single image from the slide using the DeepZoomImageTiler
        '''
        dz = DeepZoomGenerator(
            self._slide, 
            self._tile_size, 
            self._overlap,
            limit_bounds=self._limit_bounds
        )
        
        # Print out total number of tiles
        level = dz.level_count - 1
        num_tiles = dz.level_tiles[level]
        numTiles = num_tiles[0] * num_tiles[1]

        msg = {'numTiles': numTiles}
        print(json.dumps(msg))

        tiler = DeepZoomImageTiler(
            dz=dz, 
            basename=self._basename, 
            format=self._format, 
            tile_size=self._tile_size,
            queue=self._queue
        )
        tiler.run()

    def _shutdown(self):
        '''
        Shutdown the worker processes after the image has been completely tiled
        '''
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()

    def cleanup(self):
        '''
        Remove the directory where the tile images are saved 
        '''
        if os.path.isdir(self._basename):
            shutil.rmtree(self._basename)

    def terminate(self):
        for _i in range(self._workers):
            if self._worker_procs[_i] is not None:
                self._worker_procs[_i].terminate()

        self.cleanup()

if __name__ == '__main__':
    '''
    Main method allows you to test the tiler functionality from the commandline.
    For help, see: python tiler.py -h 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('slidepath', 
        help='Path to .tif file')
    parser.add_argument('--ignore-bounds',      dest='limit_bounds', 
        action='store_false',                   default=True,
        help='display entire scan area')
    parser.add_argument('--overlap', '-e',      dest='overlap', 
        metavar='PIXELS',       type=int,       default=0, 
        help='overlap of adjacent tiles [0]')
    parser.add_argument('--format', '-f',       dest='format',
        metavar='{jpeg|png}',   type=str,       default='jpeg', 
        help='image format for tiles [jpeg]')
    parser.add_argument('--jobs', '-j',         dest='workers', 
        metavar='COUNT',        type=int,       default=4, 
        help='number of worker processes to start [4]')
    parser.add_argument('--output', '-o',       dest='basename', 
        metavar='NAME',         type=str,       default='__tmp__',
        help='base name of output directory')
    parser.add_argument('--size',    '-s',      dest='tile_size', 
        metavar='PIXELS',       type=int,       default=256, 
        help='tile size [256]')

    args = parser.parse_args()

    Tiler(
        args.slidepath, 
        args.basename, 
        args.format, 
        args.tile_size, 
        args.overlap, 
        args.limit_bounds, 
        args.workers
    ).run()

    