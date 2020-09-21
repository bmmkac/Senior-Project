###############################################################################
# Creator: Joe Urbano
# Date: 3/27/2019
# Project: Senior Design, Digital Pathology
# File: testtiler.py
# Purpose: Contains unit tests for the tiler module. 
# Associated Files: gitrepo/src/SlideAnalysis/src/tiler.py 
#                       - the class which testtiler.py tests
###############################################################################

#first two import lines allow us to import tiler
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.append(os.path.abspath('../../src/SlideAnalysis/src'))
import unittest
import tiler

class TestTiler(unittest.TestCase):

###############################################################################
############################## Tiler Class ####################################
###############################################################################
    def test(self):
        '''
        TODO
        '''
        x = 4
        self.assertEqual(x, 4)

if __name__ =="__main__":
    unittest.main()