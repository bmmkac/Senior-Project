###############################################################################
# Creator: Rob Ross
# Date: 3/5/2019
# Project: Senior Design, Digital Pathology
# File: calculator.py
# Purpose: The calculator module reads in a list of tuples of int 
#          (corresponding to points in the 2-d plane) and finds the five 
#          non-contiguous circles of radius = 5500 pixels that encompass the 
#          most points.  The library relies on a datastructure called a kdtree.
#          More can be learned in the book: Computational Geometry: Algorithms
#          and Applications 2nd ed by M de Berg, M van Kreveld et al.  p 99 ff.
# Associated Files: Associated files located at gitrepo/test/SlideAnalysis/
#                   testcalculator.py - the unit tests for calculator.py
#                   democalculator.py - demos functionality of calculator.py
#                   coveragecalculator.py - calculates code coverage of
#                         calculator.py by testcalculator.py 
###############################################################################


import math

class Point:
    '''
    The Point class stores x and y coordinates in a point.  
    The coordinates must be non-negative integers (because all coordinates 
    correspond to pixel values in a slide image and all coordinates are 
    non-neg ints).  If a negative integer (or a non-integer value) is passed
    as the x or y coordinate, the initializer will store the point (-1, -1)
    Additionally, the class has a cn attribute.  This stores the x, y 
    coordinates as a tuple of tuples: ((x,y),(y,x)).  This attribute is 
    used to build and search the kdtree.
    '''

    def __init__(self, x, y):
        '''
        Params: 
            x: (non-negative integer), the x value of the point.
            y: (non-negative integer), the y value of the point.
        Return: (Point), the point (x,y)
        Note: If x or y is not a non-negative integer, an exception is raised.
        '''
        if isinstance(x,int) and x >= 0:
            self.x = x
        else:
            raise TypeError('x value of the point must be a non-negative'
            ' int or math.inf.  x value was: {}'.format(x))
        if isinstance(y,int) and y >= 0:
            self.y = y
        else:
            raise TypeError('y value of the point must be a non-negative'
            ' int or math.inf.  y value was: {}'.format(y))
        # self.cn is the composite number space rendering of x and y.
        # Provides a helpful ordering for building and searching a KDTree.
        self.cn = ((self.x,self.y),(self.y,self.x))

    def __str__(self):
        ''' Coverts the point (x,y) to the string "(x,y)".'''
        return '(' + str(self.x) + ',' + str(self.y) + ')'

    def __eq__(self, other):
        '''
        Allows two different points to be compared to one another.
        '''
        if self.x == other.x and self.y == other.y:
            return True
        return False

    def __hash__(self):
        '''
        Allows points to be placed into a set.
        '''
        return hash((self.x, self.y))

class Range:
    '''
    Range class stores a 2-d range of values ((x_min..x_max),(y_min..y_max)).  
    Each Range value (x_min, x_max, y_min, y_max) is not an int.  Rather, it is
    a tuple of int or +inf.
    The tuples corresponding to x_min and x_max are (x,y).  
    The tuples corresponding to y_min and y_max are (y,x).

    A point is considered in Range iff the following four conditions hold:
        1) point.x > x_min[0] or (point.x == x_min[0] and point.y >= x_min[1])
        2) point.x < x_max[0] or (point.x == x_max[0] and point.y <= x_max[1])
        3) point.y > y_min[0] or (point.y == y_min[0] and point.x >= y_min[1])
        4) point.y < y_max[0] or (point.y == y_max[0] and point.x <= y_max[1])

    The Range class is designed to serve the KDTree class.  A KDTree is a 
    generalization of a binary search tree to multiple dimensions.
    A binary search tree sorts elements recursively along a single axis.
    A KDTree sorts according to the median value along each axis in the space,
    sequentially.  Thus, when in 2-D, the KDTree sorts by x values on level 0,
    by y values on level 1, by x values on level 2, etc.  Generally, level i 
    nodes sort by the x axis if i % 2 == 0, and by the y axis otherwise.

    Range solves the following problem.  How might a KDTree store multiple 
    unique points with the same x value?  eg. p1=(5,1), p2=(5,2), p3=(5,3) 
    In order for the KDTree to be balanced (for maximum speed), we want 
    to be able to divide the set of points more finely than simply
    at one x value.  If we have n points that are (5, *), we want to be able
    to send the first k points that are (5, *) to the left child, so that the
    y value of each of those k points is less than the y value of each of the 
    n-k (5, *) points which we send to the right child.  The structure of the
    range class (using tuples (x,y) and (y,x)) enables finely-grained division.

    If a value other than a tuple of non-neg int or +inf is passed to the 
    range class, an error is thrown.  
    '''


    def __init__(self, x_min, x_max, y_min, y_max):
        '''
        Create a Range.
        Params: 
            x_min: a tuple of int (xval,yval), the minimum x value.
            x_max: a tuple of int (xval,yval), the maximum x value.
            y_min: a tuple of int (xval,yval), the minimum y value.
            y_max: a tuple of int (xval,yval), the maximum y value.
            xval and yval must be a non-negative integer or +inf
        Returns: (Range) the range (x_min..x_max, y_min..y_max)
        '''
        #set x_min
        if (((isinstance(x_min[0], float) and math.isinf(x_min[0])) or \
                    (isinstance(x_min[0],int))) and x_min[0] >= 0) and \
                (((isinstance(x_min[1], float) and math.isinf(x_min[1])) or \
                    (isinstance(x_min[1],int))) and x_min[1] >=0):
            self.x_min = x_min
        else:
            raise TypeError('range values must be tuples '
                            'of non-neg int.  x_min was {}'.format(x_min))
        
        #set x_max
        if (((isinstance(x_max[0], float) and math.isinf(x_max[0])) or \
                    (isinstance(x_max[0],int))) and x_max[0] >= 0) and \
                (((isinstance(x_max[1], float) and math.isinf(x_max[1])) or \
                    (isinstance(x_max[1],int))) and x_max[1] >=0): 
            self.x_max = x_max
        else:
            raise TypeError('range values must be tuples '
                            'of non-neg int.  x_max was {}'.format(x_max))
        
        #set y_min
        if (((isinstance(y_min[0], float) and math.isinf(y_min[0])) or \
                    (isinstance(y_min[0],int))) and y_min[0] >= 0) and \
                (((isinstance(y_min[1], float) and math.isinf(y_min[1])) or \
                    (isinstance(y_min[1],int))) and y_min[1] >=0): 
            self.y_min = y_min
        else:
            raise TypeError('range values must be tuples '
                            'of non-neg int.  y_min was {}'.format(y_min))
        
        #set y_max
        if (((isinstance(y_max[0], float) and math.isinf(y_max[0]) ) or \
                    (isinstance(y_max[0],int))) and y_max[0] >= 0) and \
                (((isinstance(y_max[1], float) and math.isinf(y_max[1])) or \
                    (isinstance(y_max[1],int))) and y_max[1] >=0): 
            self.y_max = y_max
        else:
            raise TypeError('range values must be tuples '
                            'of non-neg int.  y_max was {}'.format(y_max))
        
        # x_min must be less than or equal to x_max
        if self.x_min > self.x_max:
            raise TypeError('range.x_min must be less than or '
            'equal to range.x_max.  Actual values: '
            'x_min={}, x_max={}'.format(self.x_min, self.x_max))

        # y_min must be less than or equal to y_max
        if self.y_min > self.y_max:
            raise TypeError('range.y_min must be less than or '
            'equal to range.y_max.  Actual values: '
            'y_min={}, y_max={}'.format(self.y_min, self.y_max))


    def __str__(self):
        '''
        Returns: (string) the converted range self in the form
            "(x=x_min..x_max,y=y_min..y_max)".
        '''
        return 'x=(' + str(self.x_min[0]) + ',' + str(self.x_min[1]) + ')..(' \
            + str(self.x_max[0]) + ',' + str(self.x_max[1]) + '), y=(' \
            + str(self.y_min[0]) + ',' + str(self.y_min[1]) + ')..(' \
            + str(self.y_max[0]) + ',' + str(self.y_max[1]) + ')'

    def __eq__(self, other):
        '''
        Returns True if each tuple in self (x_min, x_max, y_min, y_max) equals
        the corresponding tuple in other.  Returns False otherwise. 
        '''
        #Test whether other is a Range
        if not isinstance(other, Range):
            raise TypeError('other must be a range.  '
                        'other was a {}'.format(type(other)))
        #test for equality.  Return true if equal.  Return false otherwise.
        if self.x_min == other.x_min and self.x_max == other.x_max\
            and self.y_min == other.y_min and self.y_max == other.y_max:
            return True 
        return False

    def intersects(self, search_range):
        '''
        Params: 
            self: (Range), a rectangle.
            search_range: (Range), a rectangle.
        Return: (boolean) If self and search_range have at least one point 
            in common, true. Otherwise, false.
        Notes:
            The intersection  between self and search_range can take 3 forms:
                1) self extends further to the right than search_range
                2) self extends further to the left than search_range
                3) search_range extends further to the left and right than 
                   self, and self extends higher or lower than search_range.
            Each of these forms are taken in turn in the function.
            x_min, x_max, y_min, and y_max are tuples of int.
            we compare tuples t1 and t2 as follows: 
                t1 > t2 iff t1[0] > t2[0] or (t1[0]==t2[0] and t1[1] > t2[1]) 
        '''

        # self extends further to the right than srange
        if self.x_min <= search_range.x_max and \
            search_range.x_max <= self.x_max:
            if self.y_min <= search_range.y_max and \
                search_range.y_max <= self.y_max:
                return True
            elif self.y_min <= search_range.y_min and \
                search_range.y_min <= self.y_max:
                return True
            elif search_range.y_min <= self.y_min and \
                self.y_max <= search_range.y_max:
                return True

        # self extends further to the left than srange
        if self.x_min <= search_range.x_min and \
            search_range.x_min <= self.x_max:
            if self.y_min <= search_range.y_max and \
                search_range.y_max <= self.y_max:
                return True
            elif self.y_min <= search_range.y_min and \
                search_range.y_min <= self.y_max:
                return True
            elif search_range.y_min <= self.y_min and \
                self.y_max <= search_range.y_max:
                return True

        # srange extends to the l and r of self, 
        # and self extends higher or lower than srange
        if search_range.x_min <= self.x_min and \
            self.x_max <= search_range.x_max:
            if self.y_min <= search_range.y_max and \
                search_range.y_max <= self.y_max:
                return True
            elif self.y_min <= search_range.y_min and \
                search_range.y_min <= self.y_max:
                return True

        return False


class KDTree:
    '''
    KDTree is a binary tree that structures 2 dimensional data allowing for 
    efficient search over a range.  E.G., given a list of points a KDTree
    would efficiently find all the points contained in the box (0,0), (5,0), 
    (5,5), (0,5).  For more information on KDTrees, see the book: 
    Computational Geometry: Algorithms and Applications 2nd ed by 
    M de Berg, M van Kreveld et al.  p 99 ff.
    '''

    def __init__(self, cut_value=None, cut_direction=None, left_child=None, \
        right_child=None, region=None, point=None):
        '''
        params: 
            cutvalue: (int,int)  On even levels of the tree, cutvalue=(x,y).
                On odd levels of the tree, cutvalue=(y,x).
                All points descending from the left child of even level nodes
                satisfy the following: descendent.x < cutvalue[0] or 
                descendent.x == cutvalue[0] and descendent.y <= cutvalue[1]     
                All points descending from the leftchild of odd-level nodes
                satisfy the following: descendent.y < cutvalue[0] or 
                descendent.y == cutvalue[0] and descendent.x <= cutvalue[1]
            cutdirection: either 'x' (cutting along the line x=cutvalue[0])
                or 'y' (cutting along the line y=cutvalue[0])
            leftchild: a KDTree, the left child
            rightchild: a KDTree, the right child
            region: a Range.  The smallest range that we can guarantee all 
                points in leaf nodes below self fall inside the range.
            point: a Point
        Returns: (KDTree) a node of a KDTree.
        Notes: 
            Leaf nodes have a self.point = (int,int).  
            All other leaf node attributes are none.
            Interior nodes have a self.point = none.  
            All other interior node attributes have a non-none value.
            If arguments are of incorrect type, __init__ raises a TypeError.
        '''
        if cut_value is None or isinstance(cut_value, int):
            self.cval = cut_value
        else:
            raise TypeError()

        if cut_direction is None or \
            cut_direction == "x" or cut_direction == "y":
            self.cdir = cut_direction
        else:
            raise TypeError()

        if left_child is None or isinstance(left_child, KDTree):
            self.lc = left_child
        else:
            raise TypeError()

        if right_child is None or isinstance(right_child, KDTree):
            self.rc = right_child
        else:
            raise TypeError()

        if region is None or isinstance(region, Range):
            self.reg = region
        else:
            raise TypeError()

        if point is None or isinstance(point, Point):
            self.pt = point
        else:
            raise TypeError()


    def print_node(self, depth):
        '''
        Params: 
            self: KDTree node to be printed
            depth: the distance from the root to the node.  
                Used to indent nodes of the same depth to the same level.
        Returns: (void)
        Notes:
            Prints a node: first depth*2 spaces, then node information.
            Useful for verification during testing.  
            Won't be used during deployment.
        '''
        node_string = ''
        for space in range(depth):
            node_string += '  '
        #if the node has a value in self.pt other than None, it is a leaf node.
        if self.pt: 
            node_string += 'Leaf Node:' + str(self.pt)
        else:#otherwise, it is an interior node.
            node_string += 'Interior Node: Cut Value: ' + str(self.cval) +  \
                ' Cut Direction: ' +  str(self.cdir) +  \
                ' Region: ' + str(self.reg)
        print(node_string)


    def print_kdtree(self, depth):
        '''
        Params: 
            self: KDTree to be printed
            depth: the distance from the root to self.  
                Used to indent nodes of same depth to same level.
        Returns: (void)
        Notes:
            Prints tree, with each level indented 2 spaces from the previous.  
            Useful for verification during testing.  
            Not used during deployment.
        '''
        self.print_node(depth)
        if self.lc:
            self.lc.print_kdtree(depth+1)
        if self.rc:
            self.rc.print_kdtree(depth+1)


    def build_kdtree(self, points, depth, reg):
        '''
        Params: points: Point[]
            self: KDTree, the current node
            depth: int, the number of edges traversed to get 
                from the root to self.  The root is depth 0.
            reg: Range, the region enclosing all points 
                in the subtree rooted at this node.
        Returns: (KDTree) the root of the KDTree
        Notes: 
            build_kdtree assumes all points are unique.  It doesn't assume
            x and y coordinates are unique. 
            Thus, its behavior is not guaranteed for p1=(5,6), p2=(5,6), but
            it is guaranteed for p1=(5,6), p2=(5,8), p3=(2,6)
        '''
        #if leaf, build leaf and return
        if len(points) == 1:
            self.pt = points[0]
            return

        #if interior node at even depth
        if depth % 2 == 0:
            #sort points by x value
            points.sort(key=lambda point:point.cn[0])
            #find median point (floor(n/2)) -1 for 0 based index and cut value
            median_index = math.floor((len(points)/2)-1)
            c_value = points[median_index].cn[0]
            #build left child subtree (points left of the dividing line)
            l_points = points[:median_index+1] # +1 for slice. med pt goes lt.
            l_region = Range(reg.x_min, c_value, reg.y_min, reg.y_max)
            l_child = KDTree()#lchild set in the next level of recursion
            l_child.build_kdtree(l_points, depth+1, l_region)
            #build right child substree (points right of the dividing line)
            r_points = points[median_index+1:] # +1 for slicing
            r_region = Range((c_value[0], c_value[1]+1), 
                            reg.x_max, 
                            reg.y_min, 
                            reg.y_max)
            r_child = KDTree()#rchild set in the next level of recursion
            r_child.build_kdtree(r_points, depth+1, r_region)
            #build current node and return
            self.cval = c_value
            self.cdir = 'x'
            self.lc = l_child
            self.rc = r_child
            self.reg = reg
            return

        #if interior node at odd depth
        else:
            #sort points by y value
            points.sort(key=lambda point:point.cn[1])
            #find median point (floor(n/2)) -1 for 0-based index) and cut value
            median_index = math.floor((len(points)/2)-1)
            c_value = points[median_index].cn[1]
            #build left child subtree (points below the dividing line)
            l_points = points[:median_index+1] # +1 for slice. med pt goes lt.
            l_region = Range(reg.x_min,reg.x_max, reg.y_min, c_value)
            l_child = KDTree()#lchild set in the next level of recursion
            l_child.build_kdtree(l_points, depth+1, l_region)
            #build right child substree (points above the dividing` line)
            r_points = points[median_index+1:] # +1 for slicing
            r_region = Range(reg.x_min, 
                            reg.x_max, 
                            (c_value[0], c_value[1]+1), 
                            reg.y_max)
            r_child = KDTree()#rchild set in the next level of recursion
            r_child.build_kdtree(r_points, depth+1, r_region)
            #build current node and return
            self.cval = c_value
            self.cdir = 'x'
            self.lc = l_child
            self.rc = r_child
            self.reg = reg
            return


    def search_kdtree(self, search_range):
        '''
        Params: 
            self: the root of a (subtree of a) KDTree
            search_range: Range, the range searched for.
        Returns: (Points[]) all points in a rectangular 
            region parallel to the x and y axes.
        Note: 
            Once we have the points, calculator uses filter_by_circle()
            to filter out points outside the corresponding circular HPF.
        '''
        #leaf
        if self.pt is not None:
            if search_range.x_min <= self.pt.cn[0] and \
                self.pt.cn[0] <= search_range.x_max and \
                search_range.y_min <= self.pt.cn[1] and \
                self.pt.cn[1] <= search_range.y_max:
                return [self.pt]
            else:
                return []
        
        #There may be points in left child, right child, or both. Explore each.  
        #Place all appropriate points in point_list and return point_list.
        point_list = []
        #left child is a leaf
        if self.lc.pt is not None:
            point_list = point_list + self.lc.search_kdtree(search_range)
        #left child is not a leaf and entirely in the search region
        elif search_range.x_min <= self.lc.reg.x_min and \
            self.lc.reg.x_max <= search_range.x_max and \
            search_range.y_min <= self.lc.reg.y_min and \
            self.lc.reg.y_max <= search_range.y_max:
            point_list = point_list + self.lc.report_subtree()
        #left child is not a leaf and partly in the search region
        elif self.lc.reg.intersects(search_range):
            point_list = point_list + self.lc.search_kdtree(search_range)

        #right child is a leaf
        if self.rc.pt is not None:
            point_list = point_list + self.rc.search_kdtree(search_range)
        #right child is not a leaf and entirely in the search region
        elif search_range.x_min <= self.rc.reg.x_min and \
            self.rc.reg.x_max <= search_range.x_max and \
            search_range.y_min <= self.rc.reg.y_min and \
            self.rc.reg.y_max <= search_range.y_max:
            point_list = point_list + self.rc.report_subtree()
        #right child is not a leaf and partly in the search region
        elif self.rc.reg.intersects(search_range):
            point_list = point_list + self.rc.search_kdtree(search_range)

        return point_list


    def report_subtree(self):
        '''
        Param: 
            self: (KDTree) the root of the subtree to be reported
        Returns: Point[] all points that are in the node's subtree.
        '''
        #leaf
        if self.pt is not None:
            return [self.pt]
        #points in both left child and right child
        elif self.lc is not None and self.rc is not None:
            point_list = self.lc.report_subtree() + self.rc.report_subtree()
            return point_list
        #points in only left child
        elif self.lc is not None:
            return self.lc.report_subtree()
        #points in only right child
        else:
            return self.rc.report_subtree()


###############################################################################
# The calculator module contains additional functions not belonging to a 
# separate class.  These functions use the Point, Range, and KDTree classes.
###############################################################################

def create_point_list(lst):
    '''
    Params: 
        lst: (int,int)[], a list of x,y coords [(x1,y1),(x2,y2), ..., (xn,yn)]
    Returns: Point[] a list of the x,y coords converted into Points
    Notes:
        If a tuple in the parameter contains values other than 
            non-negative integers, that tuple will be excluded from Point.
        Also, if a tuple is repeated in lst, that tuple will be represented 
            only once in the returned list.
    '''
    point_list = []
    point_set = set()
    for pt in lst:
        try:
            converted_pt = Point(pt[0],pt[1])
        except TypeError as err:
            print(err)
#            print(('TypeError in createpoint_list: a point object must be
#                'initialized with two ints (x,y). x was {}. y was {}')\
#                    .format(pt[0], pt[1]))
            continue
        if converted_pt not in point_set:
            point_list.append(converted_pt)
            point_set.add(converted_pt)
    return point_list


def print_point_list(point_list):
    '''
    Params: 
        point_list: (Point[]) the list of points to be printed.
    Returns: (void)
    Notes: 
        Prints each point on a separate line.  
        Useful for testing phase.  Won't use in deployment.
    '''
    for point in point_list:
        print(str(point))


def find_squares_and_circles(point):
    '''
    Params: 
        point: (Point) the point to which the squares and circles correspond.
    Returns: squares, circles.  Each is a list of tuples of int.
        squares: each element has the form: (xcorner, ycorner), where these
            are the x and y coordinates of the upper left corner
            of the square circumscribing a circle, with diameter 11,000, 
            that contains point on its perimeter
        circles: each ellement has the form (xcenter, ycenter), where these
            are the x and y coordinates of the center of the circle
            containing point on its perimeter.  On the circles, point is 
            located at 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, or 7pi/4.
    Notes: 
        A HPF is a 2.2 mm diameter circle, which converts to an 11k pixel 
            diameter circle.  We model that with:
                11k by 11k px square (finds first set of points)
                11k px in diameter circle (find final set of points)
    '''

    S = 11000     # diameter of the HPF
    A = 1611      # round(s * (sqrt(2) - 1)/(2 * sqrt(2)))
    B = 9389      # round(s * (sqrt(2) + 1)/(2 * sqrt(2)))
    C = 3889      # round(s / (2*sqrt(2)))
    x = point.x
    y = point.y
    #some of these squares or circles may contain area outside of the slide.
    #This is fine, and won't affect the final choice of maximum HPF circle.
    squares = [(x-int(S/2),y), (x-B,y-A), (x-S,y-int(S/2)), (x-B,y-B), 
             (x-int(S/2),y-S), (x-A,y-B), (x,y-int(S/2)), (x-A,y-A)]
    for i in range(len(squares)):
        if squares[i][0] < 0:
            squares[i] = (0, squares[i][1])
        if squares[i][1] < 0:
            squares[i] = (squares[i][0], 0)
    circles = [(x,y+int(S/2)), (x-C,y+C), (x-int(S/2),y), (x-C,y-C), 
             (x,y-int(S/2)), (x+C,y-C), (x+int(S/2),y), (x+C,y+C)]
    return (squares, circles)

def filter_by_circle(point_list, center):
    '''
    Params: 
        point_list: (Point[]) the points in the square that need to be filtered
        center: ((int,int)), the center of a radius 5500 circle
    Returns: (Point[]), the list of points that are inside the circle
    '''
    final_point_list = []
    RADIUS = 5500 # corresponds to a HPF with diameter = 2.2 mm = 11000 pixels
    for point in point_list:
        if math.ceil(math.sqrt((point.x - center[0])**2 + \
            (point.y - center[1])**2)) <= RADIUS:
            final_point_list.append(point)
    return final_point_list

def find_hpfs(points):
    '''
    Params: 
        points: ((int,int)[]) of arbitrary length.  Each entry in the list is 
            the slide's (x,y) coordinates of an eosinophil cell center. 
    returns: total_points_found, hpf_centers, hpf_points
            totalpoints_found: (int), the number of points in all 5 HPFs found
            hpf_centers: ((int,int)[]) of length 5.  Each tuple is the x,y 
                coordinates of a center of a circular HPF
            hpf_points: ((int,int)[]).  All points in any of the 5 HPFs.
    Notes: 
        HPF stands for high power field.  
        It is a circle, radius 2.2 mm = 5500 pixels
    '''
    #convert points to point_list, a Point[]
    #createpoint_list removes duplicates and excludes non-neg ints
    point_list = create_point_list(points)
   
    #set up some needed variables
    hpf_centers = [] # (int,int)[5] contains x, y coords of hpf circle centers.
    total_points_found = 0
    hpf_points = [] # Point[] all points in the 5 most populus high power fields
    NUMBHPFS = 5 # number of high power fields we seek

    for hpf in range(NUMBHPFS):
        if len(point_list) == 0:
            break

        max_points = 0 # the max number of points in a hpf in this iteration
        max_circle_center = (-1,-1) # center of the hpf with the most ecells
        max_point_list = [] # Point[] all points in the most populus HPF
        #place point_list in a KDTree
        point_tree = KDTree()
        point_tree.build_kdtree(
            points=point_list, 
            depth=0, 
            reg=Range((0,0), (math.inf,math.inf), (0,0), (math.inf,math.inf))
            )
        for point in point_list:
            #find the eight squares associated with the point.  
            squares, circles = find_squares_and_circles(point)
            #for each of the 8 squares and circles corresponding to one point
            for i in range(len(squares)):
                #find the set of points in the square then filter by the circle
                slide_range = Range((squares[i][0],0), 
                                (squares[i][0]+11000,math.inf), 
                                (squares[i][1],0), 
                                (squares[i][1]+11000,math.inf)
                                )
                curr_point_list = point_tree.search_kdtree(slide_range)
                curr_point_list = filter_by_circle(curr_point_list, circles[i])
                curr_n_points = len(curr_point_list)

                # if this is the most points so far, mark as current best.
                if curr_n_points > max_points:
                    max_points = curr_n_points
                    max_point_list = curr_point_list.copy()
                    max_circle_center = circles[i]
        # now we have the circle (hpf) with the most points in it.  
        # Record points and circle.  Remove points from point_list.  Continue.
        hpf_centers.append(max_circle_center)
        total_points_found = total_points_found + max_points
        hpf_points.extend(max_point_list)
        for point in max_point_list:
            point_list.remove(point)

    #convert hpfpoints to a list of tuples
    temp = []
    for point in hpf_points:
        temp.append((point.x, point.y))
    hpf_points = temp

    #return the number of points in the 5 HPFs (a single int), 
    #       the centers of the 5 HPFs, 
    #       and a one dimensional list of all points in the 5 HPFs
    return (total_points_found, hpf_centers, hpf_points)
