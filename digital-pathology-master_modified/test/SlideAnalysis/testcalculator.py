###############################################################################
# Creator: Rob Ross
# Date: 3/18/2019
# Project: Senior Design, Digital Pathology
# File: testcalculator.py
# Purpose: Contains unit tests for the calculator module. 
# Associated Files: gitrepo/src/SlideAnalysis/src/calculator.py 
#                       - the class which testCalculator.py tests
#                   democalculator.py - demos functionality of calculator.py
#                   coveragecalculator.py - calculates code coverage of
#                           calculator.py by testcalculator.py
###############################################################################

#first two import lines allow us to import calculator
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.append(os.path.abspath('../../src/SlideAnalysis/src'))
import unittest
import math
import calculator

class TestCalculator(unittest.TestCase):

###############################################################################
###############################Point Class#####################################
###############################################################################
    def test_point_init_good_values(self):
        '''
        Verifies that point initializer returns correct values when passed
        two non-negative integers for x and y (good input).
        '''
        test_point = calculator.Point(2,3)
        self.assertEqual(test_point.x, 2)
        self.assertEqual(test_point.y, 3)
        self.assertEqual(test_point.cn, ((2,3),(3,2)))

    def test_point_init_non_int_x(self):
        '''
        Verifies that point initializer raises TypeError when passed a string
        as an x value.
        '''
        # x value is not an integer
        self.assertRaises(TypeError, calculator.Point, 'testString', 3)

    def test_point_init_non_int_y(self):
        '''
        Verifies that point initializer raises TypeError when passed a string
        as a y value.
        '''
        # y value is not an integer
        self.assertRaises(TypeError, calculator.Point, 3, 'testString')

    def test_point_init_neg_int_x(self):
        '''
        Verifies that point initializer raises TypeError when passed a negative
        integer for x.
        '''
        # x value is a negative integer
        self.assertRaises(TypeError, calculator.Point, -7, 3)

    def test_point_init_neg_int_y(self):
        '''
        Verifies that point initializer raises TypeError when passed a negative
        integer for y.
        '''
        # y value is a negative integer
        self.assertRaises(TypeError, calculator.Point, 3, -7)

    def test_point_str(self):
        '''
        Verifies that point.__str__ prints "(xval,yval)" when passed a
        Point(xval,yval).
        '''
        test_point = calculator.Point(2,3)
        self.assertEqual(str(test_point), '(2,3)')

    def test_point_eq(self):
        '''
        Verifies that point.__eq__ returns true when comparing the same 
        point and to itself and false when comparing different points.
        '''
        test_point_1 = calculator.Point(2,3)
        test_point_2 = calculator.Point(2,3)
        test_point_3 = calculator.Point(2,4)
        test_point_4 = calculator.Point(4,3)
        
        self.assertEqual(test_point_1, test_point_2)
        self.assertNotEqual(test_point_1, test_point_3)
        self.assertNotEqual(test_point_1, test_point_4)

    def test_point_hash(self):
        '''
        Verifies that point.__hash__ results in Points being correctly
        added to a set.
        '''
        test_point_1 = calculator.Point(2,3)
        test_point_2 = calculator.Point(2,3)
        test_point_3 = calculator.Point(2,4)
        test_set = set()
        self.assertEqual(len(test_set), 0)
        test_set.add(test_point_1)
        self.assertEqual(len(test_set), 1)
        test_set.add(test_point_2)
        self.assertEqual(len(test_set), 1)
        test_set.add(test_point_3)
        self.assertEqual(len(test_set), 2)
        

###############################################################################
###############################Range Class#####################################
###############################################################################
    def test_range_init_good_input(self):
        '''
        Verifies that Range initializes correctly when given 4 values
        all of which are either non-negative integers or +infinity
        '''
        test_range = calculator.Range((0,1), (math.inf, math.inf), \
            (5, 7), (8, 9))
        self.assertEqual(test_range.x_min[0], 0)
    
    def test_range_init_bad_x_min(self):
        '''
        Verifies that range initializer raises TypeError when passed a bad
        value (tuple containing string, neg int, or float) or
        non-tuple for x_min.
        '''
        self.assertRaises(TypeError,calculator.Range, ('bad_string',1), \
            (math.inf, math.inf), (5, 7), (8, 9))
        self.assertRaises(TypeError, calculator.Range, (0,'bad_string'), \
            (math.inf, math.inf), (5, 7), (8, 9))    
        self.assertRaises(TypeError,calculator.Range, (3.5, 1), \
            (math.inf, math.inf), (5, 7), (8, 9))
        self.assertRaises(TypeError, calculator.Range, (0, 3.5), \
            (math.inf, math.inf), (5, 7), (8, 9))    
        self.assertRaises(TypeError,calculator.Range, (-100,1), \
            (math.inf, math.inf), (5, 7), (8, 9))
        self.assertRaises(TypeError, calculator.Range, (0,-80), \
            (math.inf, math.inf), (5, 7), (8, 9))    
        self.assertRaises(TypeError, calculator.Range, 'not_tuple', \
            (math.inf, math.inf), (5, 7), (8, 9))    

    def test_range_init_bad_x_max(self):
        '''
        Verifies that range initializer raises TypeError when passed a bad
        value (tuple containing string, neg int, or float) or
        non-tuple for x_max.
        '''
        self.assertRaises(TypeError,calculator.Range, (0,0), \
            ('bad_string',1), (5, 7), (8, 9))
        self.assertRaises(TypeError, calculator.Range, (0,0), \
        (0,'bad_string'), (5, 7), (8, 9))    
        self.assertRaises(TypeError,calculator.Range, (0,0), \
            (3.5, 1), (5, 7), (8, 9))
        self.assertRaises(TypeError, calculator.Range, (0,0), \
            (0, 3.5), (5, 7), (8, 9))    
        self.assertRaises(TypeError,calculator.Range, (0,0), \
            (-100,1), (5, 7), (8, 9))
        self.assertRaises(TypeError, calculator.Range, (0,0), \
            (0,-80), (5, 7), (8, 9))    
        self.assertRaises(TypeError, calculator.Range, (0,0), \
            'not_tuple', (5, 7), (8, 9))    

    def test_range_init_bad_y_min(self):
        '''
        Verifies that range initializer raises TypeError when passed a bad
        value (tuple containing string, neg int, or float) or
        non-tuple for y_min.
        '''
        self.assertRaises(TypeError,calculator.Range, (0,0), \
            (math.inf, math.inf), ('bad_string',1), (8, 9))
        self.assertRaises(TypeError, calculator.Range, (0,0), \
            (math.inf, math.inf), (0,'bad_string'), (8, 9))    
        self.assertRaises(TypeError,calculator.Range, (0,0), \
            (math.inf, math.inf), (3.5, 1), (8, 9))
        self.assertRaises(TypeError, calculator.Range, (0,0), \
            (math.inf, math.inf), (0, 3.5), (8, 9))    
        self.assertRaises(TypeError,calculator.Range, (0,0), \
            (math.inf, math.inf), (-100,1), (8, 9))
        self.assertRaises(TypeError, calculator.Range, (0,0), \
            (math.inf, math.inf), (0,-80), (8, 9))    
        self.assertRaises(TypeError, calculator.Range, (0,0), \
            (math.inf, math.inf), 'not_tuple', (8, 9))    

    def test_range_init_bad_y_max(self):
        '''
        Verifies that range initializer raises TypeError when passed a bad
        value (tuple containing string, neg int, or float) or
        non-tuple for y_max.
        '''
        self.assertRaises(TypeError,calculator.Range, (0,0), \
            (math.inf, math.inf), (5, 7), ('bad_string',1))
        self.assertRaises(TypeError, calculator.Range, (0,0), \
            (math.inf, math.inf), (5, 7), (0,'bad_string'))    
        self.assertRaises(TypeError,calculator.Range, (0,0), \
            (math.inf, math.inf), (5, 7), (3.5, 1))
        self.assertRaises(TypeError, calculator.Range, \
            (0,0), (math.inf, math.inf), (5, 7), (0, 3.5))    
        self.assertRaises(TypeError,calculator.Range, (0,0), \
            (math.inf, math.inf), (5, 7), (-100,1))
        self.assertRaises(TypeError, calculator.Range, \
            (0,0), (math.inf, math.inf), (5, 7), (0,-80))    
        self.assertRaises(TypeError, calculator.Range, (0,0), \
            (math.inf, math.inf), (5, 7), 'not_tuple')    

    def test_range_init_x_min_greater_than_x_max(self):
        '''
        Verifies that range initializer raises TypeError when x_min > x_max.
        '''
        self.assertRaises(TypeError,calculator.Range, (10,10), (8, 12), \
            (5, 7), (6,8))
        self.assertRaises(TypeError,calculator.Range, (10,10), (10, 9), \
            (5, 7), (6,8))
        # demonstrate that having x_min == x_max is acceptable
        self.assertEqual(calculator.Range((10,10), (10, 10), (5, 7), (6,8)), \
            calculator.Range((10,10), (10, 10), (5, 7), (6,8)))

    def test_range_init_y_min_greater_than_y_max(self):
        '''
        Verifies that range initializer raises TypeError when y_min > y_max.
        '''
        self.assertRaises(TypeError,calculator.Range, (0,10), (8, 12), \
            (10, 7), (6,8))
        self.assertRaises(TypeError,calculator.Range, (0,10), (10, 9), \
            (6, 9), (6,8))
        #demonstrate that having y_min == y_max is acceptable.
        self.assertEqual(calculator.Range((0,10), (10, 10), (5, 7), (5,7)), \
            calculator.Range((0,10), (10, 10), (5, 7), (5,7)))

    def test_range_string(self):
        '''
        Verifies that Range.__str__ returns the string value of Range
        '''
        self.assertEqual(str(calculator.Range((0,0), (math.inf, math.inf), \
            (5,6), (10,12))), "x=(0,0)..(inf,inf), y=(5,6)..(10,12)")

    def test_range_equality(self):
        '''
        Verifies that Range.__eq__ returns True if the compared Ranges
        are equal and false otherwise.
        '''
        self.assertTrue(calculator.Range((0,0), (math.inf, math.inf), \
            (5,6), (7,8)) == calculator.Range((0,0), (math.inf, math.inf), \
            (5,6), (7,8)))
        self.assertFalse(calculator.Range((1,0), (math.inf, math.inf), \
            (5,6), (7,8)) == calculator.Range((0,0), (math.inf, math.inf), \
            (5,6), (7,8)))
        self.assertFalse(calculator.Range((0,0), (5, math.inf), \
            (5,6), (7,8)) == calculator.Range((0,0), (math.inf, math.inf), \
            (5,6), (7,8)))
        self.assertFalse(calculator.Range((0,0), (math.inf, math.inf), \
            (6,6), (7,8)) == calculator.Range((0,0), (math.inf, math.inf), \
            (5,6), (7,8)))
        self.assertFalse(calculator.Range((0,0), (math.inf, math.inf), \
            (5,6), (7,8)) == calculator.Range((0,0), (math.inf, math.inf), \
            (5,6), (7,10)))
        
    def test_range_intersects(self):
        '''
        Verifies that Range.intersects returns True when 2 ranges intersect
        and False otherwise.
        '''
        tst_r = calculator.Range((5,0), (10, math.inf), (15,0), (20, math.inf))
        r_1 = calculator.Range((4,0), (9, math.inf), (15,0), (20, math.inf))
        r_2 = calculator.Range((6,0), (11, math.inf), (15,0), (20, math.inf))
        r_3 = calculator.Range((4,0), (11, math.inf), (15,0), (20, math.inf))
        r_4 = calculator.Range((6,0), (10, math.inf), (15,0), (20, math.inf))
        r_5 = calculator.Range((5,0), (10, math.inf), (14,0), (19, math.inf))
        r_6 = calculator.Range((5,0), (10, math.inf), (16,0), (21, math.inf))
        r_7 = calculator.Range((5,0), (10, math.inf), (14,0), (21, math.inf))
        r_8 = calculator.Range((5,0), (10, math.inf), (16,0), (19, math.inf))
        r_9 = calculator.Range((20,0), (25, math.inf), (30,0), (35, math.inf))
        self.assertTrue(tst_r.intersects(r_1))
        self.assertTrue(tst_r.intersects(r_2))
        self.assertTrue(tst_r.intersects(r_3))
        self.assertTrue(tst_r.intersects(r_4))
        self.assertTrue(tst_r.intersects(r_5))
        self.assertTrue(tst_r.intersects(r_6))
        self.assertTrue(tst_r.intersects(r_7))
        self.assertTrue(tst_r.intersects(r_8))
        self.assertTrue(tst_r.intersects(r_1))
        self.assertTrue(tst_r.intersects(r_2))
        self.assertTrue(tst_r.intersects(r_3))
        self.assertFalse(tst_r.intersects(r_9))

###############################################################################
##############################KDTree Class#####################################
###############################################################################
#TODO: Write unit tests for this class.

    def test_kdtree_report_subtree(self):
        '''
        Verifies that report_subtree correctly reports the subtree from the
        root node of the kdtree.
        '''
        #create a list of points
        point_list = [calculator.Point(1,1), 
                      calculator.Point(2,2), 
                      calculator.Point(101, 101), 
                      calculator.Point(102,102), 
                      calculator.Point(201,201)
                     ]

        #create a kdtree
        test_kdtree = calculator.KDTree()
        test_kdtree.build_kdtree(points=point_list, depth=0, \
            reg=calculator.Range((0,0), (math.inf, math.inf), (0,0), (math.inf, math.inf)))
        
        #report the subtree of the kdtree
        test_point_list = test_kdtree.report_subtree()
        
        #sort the two lists and verify that they are the same
        point_list.sort(key=lambda point:point.cn[0])
        test_point_list.sort(key=lambda point:point.cn[0])
        self.assertEqual(len(point_list), len(test_point_list))
        for i in range(len(point_list)):
            self.assertEqual(test_point_list[i], point_list[i])

    def test_kdtree_print_node_empty(self):
        '''
        Verifies that kdtree.print_node functions correctly when passed 
        an empty node
        '''
        test_kdtree = calculator.KDTree()
        test_kdtree.print_node(depth=0)

###############################################################################
############################Calculator Module##################################
###############################################################################

    def test_create_point_list_1(self):
        '''
        tests whether create_point_list accurately accepts a 
        list of two tuples and creates a list of two Point
        '''
        rawpl = [(1,2), (3,4)]
        testpl = calculator.create_point_list(rawpl)
        self.assertEqual(testpl[0].x, 1)
        self.assertEqual(testpl[0].y, 2)
        self.assertEqual(testpl[1].x, 3)
        self.assertEqual(testpl[1].y, 4)
        self.assertEqual(len(testpl), 2)

    def test_create_point_list_2(self):
        '''
        Verifies that createPointList includes repeated points only once
        '''
        rawpl = [(1,2), (1,2)]
        testpl = calculator.create_point_list(rawpl)
        self.assertEqual(testpl[0].x, 1)
        self.assertEqual(testpl[0].y, 2)
        self.assertEqual(len(testpl), 1)
        
    def test_create_point_list_3(self):
        '''
        Verifies that createPointList excludes 
        points which contain non-negative integers
        '''
        rawpl = [(-1,5), (1,2), (1,-5), (2.5,7), (7,2.5), ('squirrel', 5)]
        testpl = calculator.create_point_list(rawpl)
        self.assertEqual(testpl[0].x, 1)
        self.assertEqual(testpl[0].y, 2)
        self.assertEqual(len(testpl), 1)

    def test_find_squares_and_circles(self):
        '''
        prints out the set of upper left corners of squares and centers of circles
        corresponding to the point (20000, 40000) for visual user verification
        '''
        pt = calculator.Point(20000, 40000)
        squares,circles = calculator.find_squares_and_circles(pt)
        self.assertEqual(squares[0], (14500,40000))
        self.assertEqual(squares[1],(10611,38389))
        self.assertEqual(squares[2],(9000,34500))
        self.assertEqual(squares[3],(10611,30611))
        self.assertEqual(squares[4],(14500,29000))
        self.assertEqual(squares[5],(18389,30611))
        self.assertEqual(squares[6],(20000,34500))
        self.assertEqual(squares[7],(18389,38389))
        self.assertEqual(circles[0],(20000,45500))
        self.assertEqual(circles[1],(16111,43889))
        self.assertEqual(circles[2],(14500,40000))
        self.assertEqual(circles[3],(16111,36111))
        self.assertEqual(circles[4],(20000,34500))
        self.assertEqual(circles[5],(23889,36111))
        self.assertEqual(circles[6],(25500,40000))
        self.assertEqual(circles[7],(23889,43889))


    def test_filter_by_circle(self):
        '''
        tests filter_by_circle in the calculator module
        '''
        center = (11000, 11000)
        points = [calculator.Point(11000,11500), 
                    calculator.Point(16501,11000), 
                    calculator.Point(10000,16501), 
                    calculator.Point(11000,5499), 
                    calculator.Point(20000,20000), 
                    calculator.Point(16500,11001)]
        points = calculator.filter_by_circle(points, center)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].x, 11000)
        self.assertEqual(points[0].y, 11500)

    def test_print_point_list(self):
        '''
        Verifies that print_point_list prints the point list as expected.
        '''
        point_list = [calculator.Point(2,3), calculator.Point(3, 5)]
        calculator.print_point_list(point_list)

    def test_find_hpfs(self):
        '''
        Verifies that find_hpfs finds the expected high power fields.
        Creates 6 clusters of points, with specific centers and
        varying cardinalities (2-7 points in each cluster)
        Then runs find_hpfs on the list of points to verify that the
        correct centers and number of points are returned.
        The points are carefully chosen to ensure that a particular center
        will be found for each hpf.
        '''
        #create 6 clusters of points, with 2, 3, 4, 5, 6, and 7 points in them
        point_list = []
        #cluster_1 has 2 points centered at (11000, 11000).  
        #It has the least points, and so should not be found.
        cluster_1 = [(11000,16500),
                     (11000,5500)]

        #cluster_2 has 3 points centered at (11000,44000)
        cluster_2 = [(11000,44000), 
                     (11000,38500), 
                     (11000,49500)]
        #cluster_3 has 4 points centered at (11000,77000)
        cluster_3 = [(11000,77000), 
                     (11000,82500), 
                     (11000,71500), 
                     (5500,77000)]
        #cluster_4 has 5 points centered at (55000,11000)
        cluster_4 = [(55000,11000), 
                     (55000,16500), 
                     (55000,5500), 
                     (49500,11000), 
                     (60500,11000)]
        #cluster_5 has 6 points centered at (55000,44000)
        cluster_5 = [(55000,44000), 
                     (55000,38500), 
                     (55000,49500), 
                     (49500,44000), 
                     (60500,44000), 
                     (56000,46000)]
        #cluster_6 has 7 points centered at (55000,77000)
        cluster_6 = [(55000,77000), 
                     (55000, 71500), 
                     (55000,82500), 
                     (49500,77000), 
                     (60500,77000), 
                     (56000, 78000), 
                     (54000,76000)]

        #add all clusters to point_list
        point_list.extend(cluster_1)
        point_list.extend(cluster_2)
        point_list.extend(cluster_3)
        point_list.extend(cluster_4)
        point_list.extend(cluster_5)
        point_list.extend(cluster_6)

        #find points, centers and list of found points
        total_points_found, hpf_centers, hpf_points = \
            calculator.find_hpfs(point_list)
        
        self.assertEqual(total_points_found, 3+4+5+6+7)
        #clusters were added to hpf_centers in order of decreasing cardinality
        self.assertEqual(hpf_centers[0], (55000,77000))#center of cluster_6
        self.assertEqual(hpf_centers[1], (55000,44000))#center of cluster_5
        self.assertEqual(hpf_centers[2], (55000,11000))#center of cluster_4
        self.assertEqual(hpf_centers[3], (11000,77000))#center of cluster_3
        self.assertEqual(hpf_centers[4], (11000,44000))#center of cluster_2

if __name__ =="__main__":
    unittest.main()
