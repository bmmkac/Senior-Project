###############################################################################
# Creator: Joe Urbano
# Date: 4/2/2019
# Project: Senior Design, Digital Pathology
# File: ProcessSlide.py
###############################################################################

import os
import sys
import argparse
import shutil

import json
import multiprocessing
import signal

from tile_processor import TileProcessorPool
from tiler import Tiler
from calculator import find_hpfs
from visualizer import visualize
from openslide import OpenSlide
import time

TMP_DIR             = '__tmp__'
FORMAT              = 'jpeg'
TILE_SIZE           = 5000
OVERLAP             = 50
TILER_WORKERS       = multiprocessing.cpu_count()
PROCESSOR_WORKERS   = multiprocessing.cpu_count()

tile_processor_pool = None
tiler = None

def handle_interrupt(signum, frame):
    signal.signal(signal.SIGINT, original_sigint)

    try:
        if tile_processor_pool is not None:
            tile_processor_pool.terminate()
        if tiler is not None:
            tiler.terminate()
        shutil.rmtree(TMP_DIR)
    except:
        pass

    print(json.dumps({'terminating': args.slidepath}))

    sys.exit(1)


if __name__ == '__main__':
    # store the original SIGINT handler
    try:
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, handle_interrupt)

        multiprocessing.set_start_method('spawn')

        parser = argparse.ArgumentParser()
        parser.add_argument('slidepath', 
            help='Path to .tif file')

        args = parser.parse_args()

        print(json.dumps({'initializing': args.slidepath}))

        start = time.time()

        tile_processor_pool = TileProcessorPool(
            workers=PROCESSOR_WORKERS
        )
        tile_processor_pool.start()

        tiler = Tiler(
            slidepath=args.slidepath, 
            basename=TMP_DIR, 
            format=FORMAT, 
            tile_size=TILE_SIZE, 
            overlap=OVERLAP, 
            limit_bounds=True, 
            workers=TILER_WORKERS,
            tile_processor_queue=tile_processor_pool.get_queue()
        )
        tiler.run()

        results = tile_processor_pool.gather_results()

        print(json.dumps({'tilingComplete': args.slidepath}))

        total_points_found, hpf_centers, hpf_points = find_hpfs(results)
        hpfs = list(zip(hpf_centers, hpf_points))

        basename = os.path.basename(args.slidepath)
        with OpenSlide(args.slidepath) as slide:
            hpf_data = visualize(slide, hpf_centers, hpf_points, dir='.', basename=basename)
        elapsed = time.time() - start

        tiler.cleanup()

        shutil.rmtree(TMP_DIR)

        print(json.dumps(
            {
                'processingComplete': args.slidepath,
                'elapsedTime': time.strftime("%H:%M:%S", time.gmtime(elapsed)),
                'eosinophilCount': total_points_found,
                'hpf': hpf_data
            }
        ))
    except:
        pass
