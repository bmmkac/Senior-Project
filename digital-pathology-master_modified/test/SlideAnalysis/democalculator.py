###############################################################################
# Creator: Rob Ross
# Date: 3/18/2019
# Project: Senior Design, Digital Pathology
# File: democalculator.py
# Purpose: Contains speed and accuracy demonstrations of the calculator module. 
# Associated Files: /gitrepo/src/SlideAnalysis/src/calculator.py 
#                        - the module which democalculator.py demonstrates.
#                   ./testcalculator.py - unit tests for calculator.py
#                   ./coveragecalculator.py - calcs unit test code coverage.
###############################################################################

#first two import lines allow us to import calculator
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.append(os.path.abspath('../../src/SlideAnalysis/src'))
import unittest
import calculator
import random
import time
import math
import matplotlib.pyplot as plt

class TestCalculator(unittest.TestCase):

    def test_lots_of_points(self):
        '''
        Measures the time required to find points in a range 
        with the kdtree and iterating through a list.  Outputs the times
        so that a user can observe the time difference.  It also prints 
        out the list of points found with the kdtree and with the list 
        so that the user can visually verify accuracy.
        '''
        n = 2000
        max_value = 100000
        min_value = 0
        min_xy = (50000,0)
        max_xy = (60000,math.inf)
        #generate n points randomly (0..100000, 0..100000)
        point_list = []
        i = 0
        excluded = set()
        while i < n:
            point = calculator.Point(random.randint(min_value, max_value), 
                                    random.randint(min_value, max_value))
            if point not in excluded:
                point_list.append(point)
                excluded.add(point)
                i = i + 1
        #time placing the points into a KDTree
        start = time.time()
        my_tree = calculator.KDTree()
        slide_range = calculator.Range((0, 0), (math.inf, math.inf), 
                                        (0,0), (math.inf,math.inf))
        my_tree.build_kdtree(point_list, 0, slide_range)
        build_time = time.time() - start
        print('time to build', n, ' node kdtree: ', build_time)

        #time searching the tree for a 10,000 by 10,000 box
        start = time.time()
        found_points = my_tree.search_kdtree(calculator.Range(min_xy, max_xy, 
                                                            min_xy, max_xy))
        search_time = time.time() - start
        print('time to search', n, ' node kdtree: ', search_time)
        print('found', len(found_points), 'points in ', search_time)
        found_points_sorted = sorted(found_points, key=lambda point: point.cn[0])
        for i in range(len(found_points)):
            print(str(found_points_sorted[i]), end=" ")
        print()

        #time searching the tree for the same box
        start = time.time()
        found_points_2 = []
        for point in point_list:
            if min_xy <= point.cn[0] and \
                    point.cn[0] <= max_xy and \
                    min_xy <= point.cn[1] and \
                    point.cn[1] <= max_xy:
                found_points_2.append(point)
        search_time = time.time() - start
        print('time to search', n, ' nodes in a list: ', search_time)
        print('found', len(found_points_2), 'points in ', search_time)
        found_points_sorted_2 = sorted(found_points_2, key=lambda point: point.cn[0])
        for i in range(len(found_points_sorted_2)):
            print(str(found_points_sorted_2[i]), end=" ")
        print()

    def test_find_hpfs(self):
        '''
        1) generates 1000 random points between (0,0) and (100000,100000).
        2) generates 75 specific points, clustered around 5 random areas
        3) runs find_hpfs on the set of 1075 points
        4) saves a visual representation of the results in ./results 
           for visual verification by tester
        '''
        n = 1000
        max_value = 100000
        min_value = 0
        #generate n points randomly (0..100000, 0..100000)
        point_list = []
        i = 0
        excluded = set()
        while i < n:
            point = (random.randint(min_value, max_value), 
                        random.randint(min_value, max_value))
            if point not in excluded:
                point_list.append(point)
                excluded.add(point)
                i = i + 1
        # Include random clusters of 15 points to force findHPFs 
        # to choose HPFs centered near where we expect them.  
        # In final output image, clusters are shown in green.
        foci = []
        focus_points = []
        for i in range(5):
            foci.append((random.randint(min_value, max_value),
                (random.randint(min_value, max_value))))
            for j in range(15):
                focus_points.append((foci[i][0]+100*j, foci[i][1]+100*j))
        point_list.extend(focus_points)
        start = time.time()
        total_points_found, hpf_center, hpf_points = \
                                            calculator.find_hpfs(point_list)
        total_time = time.time() - start

        #Output results for visual inspection by tester
        print("totalpointsfound =", total_points_found)

        #Add points in HPF circles:
        xs_in_hpfs = []
        ys_in_hpfs = []
        for point in hpf_points:
            xs_in_hpfs.append(point[0])
            ys_in_hpfs.append(point[1])
        plt.plot(xs_in_hpfs, ys_in_hpfs, 'ro', markersize=2)

        #Add points out of HPF circles:
        xs_out_hpfs = []
        ys_out_hpfs = []
        for point in point_list:
            if point not in hpf_points:
                xs_out_hpfs.append(point[0])
                ys_out_hpfs.append(point[1])
        plt.plot(xs_out_hpfs, ys_out_hpfs, 'o', markersize=2)

        #Add foci points:
        foci_xs = []
        foci_ys = []
        for point in foci:
            foci_xs.append(point[0])
            foci_ys.append(point[1])
        plt.plot(foci_xs, foci_ys, 'go', markersize=6)

        #Add HPF circles to the figure:
        circle_1 = plt.Circle(hpf_center[0], 5500, color='r')
        circle_2 = plt.Circle(hpf_center[1], 5500, color='r')
        circle_3 = plt.Circle(hpf_center[2], 5500, color='r')
        circle_4 = plt.Circle(hpf_center[3], 5500, color='r')
        circle_5 = plt.Circle(hpf_center[4], 5500, color='r')

        #Set up figure
        plt.ylabel('results of findHPFs')
        fig = plt.gcf()
        ax = plt.gca()
        ax.add_artist(circle_1)
        ax.add_artist(circle_2)
        ax.add_artist(circle_3)
        ax.add_artist(circle_4)
        ax.add_artist(circle_5)
        ax.set_aspect('equal')
        circle_1.set_facecolor('none')
        circle_2.set_facecolor('none')
        circle_3.set_facecolor('none')
        circle_4.set_facecolor('none')
        circle_5.set_facecolor('none')
        fig.savefig('results.png')

#######UNCOMMENT TO SEE PRINT OUT OF ALL EXCLUDED POINTS
#        print("Points NOT in the HPFs:", end=" ")
#        outside_points = point_list.copy()
#        for point in hpf_points:
#            outside_points.remove(point)
#        for point in outside_points:
#            print(point, end=" ")
#        print()

        print("search completed in {} seconds".format(total_time))
        print("see ./results.png for a visualization of the solution")


if __name__ == '__main__':
    unittest.main()
