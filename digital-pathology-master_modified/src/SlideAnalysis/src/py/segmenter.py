import numpy as np
import cv2
import scipy as sp
import skimage.color
import skimage.io
import skimage.measure
import skimage.morphology
from PIL import Image

import os
import threading
import itertools
import warnings

from math import floor, ceil

from htk.area_open import area_open
from htk.cdog import cdog
from htk.color_deconvolution import color_deconvolution
from htk.detect_nuclei_kofahi import detect_nuclei_kofahi
from htk.lab_mean_std import lab_mean_std
from htk.max_clustering import max_clustering
from htk.reinhard import reinhard

# img : numpy array
def test_white_threshold(img):
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    avg_color = np.average(avg_color)

    if avg_color > 210:
        return True
    else:
        return False

# Color normalization
def normalize(img):
    # Load reference image for normalization
    ref_image_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'htk', 'L1.png')  # L1.png
    im_reference = skimage.io.imread(ref_image_file)[:, :, :3]

    # get mean and stddev of reference image in lab space
    mean_ref, std_ref = lab_mean_std(im_reference)

    # perform reinhard color normalization
    im_nmzd = reinhard(img, mean_ref, std_ref)
        
    return im_nmzd

# Color deconvolution
def deconvolution(img):
    # create stain to color map
    stainColorMap = {
        'hematoxylin': [0.65, 0.70, 0.29],
        'eosin': [0.07, 0.99, 0.11],
        'dab': [0.27, 0.57, 0.78],
        'null': [0.0, 0.0, 0.0]
    }

    # specify stains of input image
    stain_1 = 'hematoxylin'  # nuclei stain
    stain_2 = 'eosin'  # cytoplasm stain
    stain_3 = 'null'  # set to null of input contains only two stains

    # create stain matrix
    W = np.array([stainColorMap[stain_1],
                    stainColorMap[stain_2],
                    stainColorMap[stain_3]]).T

    # perform standard color deconvolution
    im_stains = color_deconvolution(img, W).Stains

    return im_stains


class SegmenterHTK(object):

    # Segment Nuclei
    def segment_nuclei(self,im_stains):

        # get nuclei/hematoxylin channel
        im_nuclei_stain = im_stains[:, :, 0]

        # segment foreground
        foreground_threshold = 60

        im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
            im_nuclei_stain < foreground_threshold)

        # run adaptive multi-scale LoG filter
        min_radius = 10
        max_radius = 15
        min_nucleus_area = 80
        local_max_search_radius = 10

        im_nuclei_seg_mask = detect_nuclei_kofahi(im_nuclei_stain, im_fgnd_mask, min_radius,max_radius, min_nucleus_area, local_max_search_radius)
        # compute nuclei properties
        nuclei = skimage.measure.regionprops(im_nuclei_seg_mask)

        return nuclei

    def get_centers(self,nuclei):
        centers = []

        for i in range(len(nuclei)):
            c = [nuclei[i].centroid[1], nuclei[i].centroid[0], 0]
            centers.append([c[0], c[1]])

        return centers

    def segment(self,img):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im_nmzd = normalize(img)
            img_stains = deconvolution(im_nmzd)
            nuclei = self.segment_nuclei(img_stains)
            centers = self.get_centers(nuclei)
            return centers



def get_circular_structuring_element(radius):
    strel = np.zeros((radius*2+1,radius*2+1),np.uint8)
    cv2.circle(strel,(radius,radius), radius, 1, -1, cv2.LINE_8, 0)
    return strel

def morphological_reconstruction(ipMarker, ipMask):
    marker = skimage.img_as_float(ipMarker)
    mask = skimage.img_as_float(ipMask)

    result = skimage.morphology.reconstruction(marker, mask, method='dilation')
    return skimage.img_as_ubyte(result)

def watershedCV(img):
    opening_radius = 15 #px
    min_area = 100 # px^2
    max_area = 2000
    
    threshold = 80
    
    mat = img.copy()

    # normalize, deconvolve stains
    mat = normalize(mat)
    mat = deconvolution(mat)
    hematoxylin = mat[:, :, 0] # get hematoxylin channel

    mat = hematoxylin.copy() # copy hematoxylin and save it for later use

    # increase image contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    mat = clahe.apply(mat)

    # preprocessing
    mat = cv2.medianBlur(mat,1)
    mat = cv2.GaussianBlur(mat,(5,5), 0.75)
    matBackground = cv2.morphologyEx(mat, cv2.MORPH_CLOSE, get_circular_structuring_element(1))
    mat = morphological_reconstruction(mat, matBackground)
    
    # Apply opening by reconstruction & subtraction to reduce background
    matBackground = cv2.morphologyEx(mat, cv2.MORPH_OPEN, get_circular_structuring_element(opening_radius))
    matBackground = morphological_reconstruction(matBackground, mat)
    mat = mat - matBackground
    
    # threshold background to pull out nuclei
    ret,thresh = cv2.threshold(matBackground,50,255,cv2.THRESH_BINARY_INV)
    
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    # prepare data for watershed segmentation
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.02*dist_transform.max(),255,0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    # prepare markers for watershed segmentation
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    
    # watershed
    markers = cv2.watershed(img, markers)
    
    # extract contour info after segmentation
    matTemp = np.zeros(markers.shape, np.int32)
    matTemp = cv2.min(matTemp, markers)
    matTemp = np.uint8(matTemp)
    
    contours, hierarchy = cv2.findContours(matTemp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # filter contours by size and threshold
    filtered_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area and area <= max_area:
            mask = np.zeros(matTemp.shape,np.uint8)
            cv2.drawContours(mask,[c],0,255,-1)
            mean_val = cv2.mean(hematoxylin, mask=mask)[0]
            if mean_val < threshold:
                filtered_contours.append(c)
    
    return filtered_contours


class SegmenterCV_Thread(threading.Thread):
    def __init__(self, threadID, img, coordinates, results):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.img = img.copy()
        self.coordinates = coordinates
        self.results = results

    def run(self):
        centers = self.process()
        self.results[self.threadID] = centers

    def process(self):
        if test_white_threshold(self.img):
            return []

        contours = watershedCV(self.img)

        centers = []
        for c in contours:
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centers.append([self.coordinates[0] + cx, self.coordinates[1] + cy])

        return centers

class SegmenterCV(object):
    def segment(self, img):
        num_threads = 9
        threads = [None] * num_threads
        results = [None] * num_threads
        overlap = 50

        w, h = img.shape[0], img.shape[1]
        dx = (w + 2 * overlap) // 3
        dy = (h + 2 * overlap) // 3

        for i in range(3):
            for j in range(3):
                t_id = 3*j + i

                x1 = i*dx - i*overlap
                x2 = w if i == 2 else x1 + dx
                y1 = j*dy - j*overlap
                y2 = h if j == 2 else y1 + dy

                sub_img = img[y1:y2, x1:x2]

                threads[t_id] = SegmenterCV_Thread(t_id, sub_img, (x1, y1), results)
                threads[t_id].start()
        
        for i in range(len(threads)):
            threads[i].join()
        
        return list(itertools.chain.from_iterable(results))

