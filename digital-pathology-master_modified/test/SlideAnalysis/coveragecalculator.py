###############################################################################
# Creator: Rob Ross
# Date: 3/22/2019
# Project: Senior Design, Digital Pathology
# File: coveragecalculator.py
# Purpose: Outputs code coverage of calculator.py by testcalculator.py
#          Stores results in ./htmlcov
#          See line-by-line coverage at ./htmlcov/index.html
# Associated Files: /gitrepo/src/SlideAnalysis/src/calculator.py 
#                        - the module which democalculator.py demonstrates.
#                   ./testcalculator.py - unit tests for calculator.py
#                   ./democalculator.py - demos functionality of calculcator.py
###############################################################################
import coverage
import unittest

cov = coverage.Coverage()
cov.start()                                     # start code coverage analysis
tests = unittest.TestLoader().discover(".")     # search for all unit tests in current dir
unittest.TextTestRunner(verbosity=2).run(tests) # run all tests in current dir, showing output of each test case
cov.stop()
cov.html_report()                               # generate HTML report in ./htmlcov
