#/usr/bin/env python
"""
pythonize_model_test.py

Test that pythonize_model returns the correct types.

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""
import unittest
import numpy as np
from mlpack import pythonize_model 

class TestPythonize(unittest.TestCase):
  """
  This class defines tests for the to_matrix() and to_matrix_with_info() utility
  functions.
  """

  def testPythonizeModel(self):
    """
    Test that a simple logistic regression's parameters
    """
    x = np.random.rand(10,10)
    y = [1,0,1,0,1,0,1,0,0,0]

    from mlpack import logistic_regression
    output = logistic_regression(training=x,labels=y)

    out = pythonize_model.transform(output['output_model'])

    self.assertTrue(isinstance(out, dict))
    self.assertTrue(isinstance(out['parameters'], np.ndarray))


def test_suite():
    """
    Run all tests.
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(TestPythonize))
    return suite

if __name__ == '__main__':
    unittest.main()
