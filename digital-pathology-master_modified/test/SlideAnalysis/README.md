This folder contains three kinds of files: unit tests (test*.py), code 
coverage analysis (coverage*.py), and demonstrations (demo*.py).
For each file, you can run it from the commandline by typing
     python3 <filename>

# Unit Tests
Listed below are our unit tests, by module and by class.

## testcalculator.py
### Point Class

### Range Class

### KDTree Class

### Rest of Calculator Module not contained in a class

# Code Coverage Analysis
##coveragecalculator.py

#Demonstrations

## democalculator.py
Note: Demonstrations are unit tests which have print statements or other side
effects which might be useful to demonstrate the functionality of the module.
Thus, they are part of the class TestCalculator and are named test_*.

### test\_lots\_of\_points
Measures the time required to find points in a range with a KDTree of Point
and by iterating through a list of Point.  Prints out the times so the user can
verify that the KDTree is faster and prints out the two sets of points found
So that the user can visually verify accuracy.

### test\_find\_hpfs
1. generates 1000 random points between (0,0) and (100000,100000).
2. generates 75 specific points, clustered around 5 randomly chosen areas.
3. runs find\_hpfs on the set of 1075 points
4. saves a visual representation of the results in ./results for visual verification.
