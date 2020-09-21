import os
import sys
import itertools
import numpy
from multiprocessing import Process, Manager, JoinableQueue, Pipe
import threading
import json
import time

from tile import Tile
from predictor import *
from segmenter import SegmenterHTK, SegmenterCV, test_white_threshold

from math import floor

CELL_SIZE = 50

class TileProcessor(object):
    def __init__(self, segmenter, predictor, pipe_out=None):
        self._segmenter = segmenter
        self._predictor = predictor
        self._pipe_out = pipe_out

    def communicate(self, msg):
        if self._pipe_out:
            self._pipe_out.send_bytes(msg.encode('utf-8'))
        else:
            print(msg)

    def process_tile(self, tile):
        img = tile.get_image_as_nparray()

        if test_white_threshold(img):
            return []
        else:
            # segment tile image to get cell coordinates
            cpid = os.getpid()
            print("===========Start Segmenter for tile=====================")
            start = time.time()
            try:
                coordinates = self._segmenter.segment(img)
            except:
                coordinates = []
            stop = time.time()
            timer1 = time.strftime("%H:%M:%S", time.gmtime(stop-start));
            print("============Segmenter Time= "+str(timer1)+"========================")
            # round coordinates to nearest pixel
            coordinates = list(
                map(
                    lambda coord:
                        [floor(coord[0] + .5), floor(coord[1] + .5)]
                    , coordinates
                )
            )

            # crop cell images from list of coordinates
            cell_image_list = list(
                map(
                    lambda coord:
                        img[coord[1]-CELL_SIZE//2 : coord[1]+CELL_SIZE//2,
                            coord[0]-CELL_SIZE//2 : coord[0]+CELL_SIZE//2]
                    , coordinates
                )
            )
            print("===========Start Classifier for tile=====================")
            start = time.time()
            ecell_coordinates = self._predictor.filter_positive_classifications(cell_image_list, coordinates)
            stop = time.time()
            timer2 = time.strftime("%H:%M:%S", time.gmtime(stop-start));
            print("============Classifier Time= "+str(timer2)+"========================")
            data={}
            data[cpid] = []
            data[cpid].append({
                'Segmenter_time':timer1 ,
                'Classifier_time':timer2
            })
            with open('timer.txt', 'a') as outfile:
                json.dump(data, outfile)

            # map tile coordinates to world coordinates
            x,y = tile.get_coordinates()
            return list(
                map(
                    lambda coord:
                        [coord[0] + x, coord[1] + y]
                    , ecell_coordinates
                )
            )


class TileProcessorWorker(Process):
    '''A child process that processes tiles.'''

    def __init__(self, queue, results, pipe_out):

        Process.__init__(self, name='TileProcessorWorker')
        self.daemon = True
        self._queue = queue
        self._results = results
        self._pipe_out = pipe_out

        self._segmenter = SegmenterHTK()
        self._predictor = Predict()

        self._tile_processor = TileProcessor(self._segmenter, self._predictor, self._pipe_out)

    def communicate(self, msg):
        if self._pipe_out:
            self._pipe_out.send_bytes(msg.encode('utf-8'))
        else:
            print(msg)

    def run(self):
        '''
        Start process execution
        '''
        coordinates = []
        while True:
            data = self._queue.get()
            if data is None: # no more tiles to process, break out of loop
                self._queue.task_done()
                break

            tile = data

            self.communicate(json.dumps({'processingTile': tile.get_filename()}))
            ecells = self._tile_processor.process_tile(tile)
            self.communicate(json.dumps({'completedTile': tile.get_filename()}))

            coordinates.extend(ecells)

            tile.cleanup() # delete tile after processing it

            self._queue.task_done()

        self._results[os.getpid()] = coordinates
        self._pipe_out.close()


class TileProcessorPool(object):
    class ChildOutputWorker(threading.Thread):
        def __init__(self, threadID, conn):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.conn = conn

        def run(self):
            while True:
                try:
                    block = self.conn.recv_bytes()
                    print(block.decode('utf-8'))
                    sys.stdout.flush()
                except EOFError:
                    self.conn.close()
                    break

    def __init__(self, workers):
        self._all_open_child_conns = [None] * workers
        self._worker_procs = [None] * workers
        self._child_output_threads = [None] * workers
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._manager = Manager()
        self._results = self._manager.dict()

    def get_queue(self):
        return self._queue

    def start(self):
        ''' start up the process pool '''
        # initialize TileProcessorWorker processes
        for _i in range(self._workers):
            child_conn, parent_conn = Pipe(duplex=False)
            self._all_open_child_conns[_i] = child_conn
            self._worker_procs[_i] = TileProcessorWorker(
                queue=self._queue,
                results=self._results,
                pipe_out=parent_conn
            )
            self._worker_procs[_i].start()
            parent_conn.close() # close this instance of the pipe

            self._child_output_threads[_i] = TileProcessorPool.ChildOutputWorker(_i, child_conn)
            self._child_output_threads[_i].start()

    def _shutdown(self):
        '''
        Shutdown the worker processes after the tiles have all been processed
        '''
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()

        for i in range(self._workers):
            self._child_output_threads[i].join()

    def gather_results(self):
        '''
        Returns a list of eosinophil coordinates as a list of lists:
        [[1,2], [3,4]...]
        '''
        self._shutdown()
        return list(itertools.chain.from_iterable(self._results.values()))

    def terminate(self):
        for _i in range(self._workers):
            if self._worker_procs[_i] is not None:
                self._worker_procs[_i].terminate()
