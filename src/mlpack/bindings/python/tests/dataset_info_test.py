#/usr/bin/env python
"""
test_dataset_info.py

Test that to_matrix() and to_matrix_with_info() return the correct types.

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""
import unittest
import pandas as pd
import numpy as np

from mlpack.matrix_utils import to_matrix
from mlpack.matrix_utils import to_matrix_with_info

class TestToMatrix(unittest.TestCase):
  """
  This class defines tests for the to_matrix() and to_matrix_with_info() utility
  functions.
  """

  def testPandasToMatrix(self):
    """
    Test that a simple pandas numeric matrix can be turned into a numpy ndarray.
    """
    d = pd.DataFrame(np.random.randn(100, 4), columns=list('abcd'))

    m, _ = to_matrix(d)

    self.assertTrue(isinstance(m, np.ndarray))
    self.assertEqual(m.shape[0], 100)
    self.assertEqual(m.shape[1], 4)
    self.assertEqual(m.dtype, np.dtype(np.double))
    colnames = list('abcd')
    for i in range(m.shape[1]):
      for j in range(m.shape[0]):
        self.assertEqual(m[j, i], d[colnames[i]][j])

  def testPandasIntToMatrix(self):
    """
    Test that a matrix holding ints is properly turned into a double matrix.
    """
    d = pd.DataFrame({'a': range(5)})

    m, _ = to_matrix(d)

    self.assertTrue(isinstance(m, np.ndarray))
    self.assertEqual(m.shape[0], 5)
    self.assertEqual(m.shape[1], 1)
    for i in range(5):
      self.assertEqual(m[i], i)

  def testPandasMixedToMatrix(self):
    """
    Test that a matrix with one int and one double feature are transformed
    correctly.
    """
    d = pd.DataFrame({'a': range(50)})
    d['b'] = np.random.randn(50, 1)
    self.assertTrue((d['a'].dtype == np.dtype('int32')) or
                    (d['a'].dtype == np.dtype('int64')))
    self.assertEqual(d['b'].dtype, np.dtype(np.double))

    m, _ = to_matrix(d)

    self.assertTrue(isinstance(m, np.ndarray))
    self.assertEqual(m.dtype, np.dtype(np.double))
    self.assertEqual(m.shape[0], 50)
    self.assertEqual(m.shape[1], 2)
    colNames = list('ab')
    for i in range(2):
      for j in range(50):
        self.assertEqual(d[colNames[i]][j], m[j, i])

  def testArraylikeToMatrix(self):
    """
    Test that if we pass some array, we get back the right thing.  This array
    will be filled with doubles only.
    """
    a = [[0.01, 0.02, 0.03],
         [0.04, 0.05, 0.06],
         [0.07, 0.08, 0.09],
         [0.10, 0.11, 0.12]]

    m, _ = to_matrix(a)

    self.assertTrue(isinstance(m, np.ndarray))
    self.assertEqual(m.dtype, np.dtype(np.double))
    self.assertEqual(m.shape[0], 4)
    self.assertEqual(m.shape[1], 3)

    for i in range(4):
      for j in range(3):
        self.assertEqual(a[i][j], m[i, j])

  def testMultitypeArraylikeToMatrix(self):
    """
    Test that if we pass an array with multiple types, we get back the right
    thing.  The numpy ndarray should be filled with doubles only.
    """
    a = [[0.01, 0.02, 3],
         [0.04, 0.05, 6],
         [0.07, 0.08, 9],
         [0.10, 0.11, 12]]

    m, _ = to_matrix(a)

    self.assertTrue(isinstance(m, np.ndarray))
    self.assertEqual(m.dtype, np.dtype(np.double))
    self.assertEqual(m.shape[0], 4)
    self.assertEqual(m.shape[1], 3)

    for i in range(4):
      for j in range(3):
        self.assertEqual(a[i][j], m[i, j])

  def testNumpyToMatrix(self):
    """
    Make sure we can convert a numpy matrix without copying anything.
    """
    m1 = np.random.randn(100, 5)
    m2, _ = to_matrix(m1)

    self.assertTrue(isinstance(m2, np.ndarray))
    self.assertEqual(m2.dtype, np.dtype(np.double))

    p1 = m1.__array_interface__
    p2 = m2.__array_interface__

    self.assertEqual(p1['data'], p2['data'])

  def testPandasToMatrixNoCategorical(self):
    """
    Make sure that if we pass a Pandas dataframe with no categorical features,
    we get back the matrix we expect.
    """

class TestToMatrixWithInfo(unittest.TestCase):
  """
  This class defines tests for the to_matrix() and to_matrix_with_info() utility
  functions.
  """

  def testPandasToMatrix(self):
    """
    Test that a simple pandas numeric matrix can be turned into a numpy ndarray.
    """
    d = pd.DataFrame(np.random.randn(100, 4), columns=list('abcd'))

    m, _, dims = to_matrix_with_info(d, np.double)

    self.assertTrue(isinstance(m, np.ndarray))
    self.assertEqual(m.shape[0], 100)
    self.assertEqual(m.shape[1], 4)
    self.assertEqual(m.dtype, np.dtype(np.double))
    colnames = list('abcd')
    for i in range(m.shape[1]):
      for j in range(m.shape[0]):
        self.assertEqual(m[j, i], d[colnames[i]][j])

    self.assertTrue(dims.shape[0], 4)
    self.assertEqual(dims[0], 0)
    self.assertEqual(dims[1], 0)
    self.assertEqual(dims[2], 0)
    self.assertEqual(dims[3], 0)

  def testPandasIntToMatrix(self):
    """
    Test that a matrix holding ints is properly turned into a double matrix.
    """
    d = pd.DataFrame({'a': range(5)})

    m, _, dims = to_matrix_with_info(d, np.double)

    self.assertTrue(isinstance(m, np.ndarray))
    self.assertEqual(m.shape[0], 5)
    self.assertEqual(m.shape[1], 1)
    for i in range(5):
      self.assertEqual(m[i], i)

    self.assertTrue(dims.shape[0], 1)
    self.assertEqual(dims[0], 0)

  def testPandasMixedToMatrix(self):
    """
    Test that a matrix with one int and one double feature are transformed
    correctly.
    """
    d = pd.DataFrame({'a': range(50)})
    d['b'] = np.random.randn(50, 1)
    self.assertTrue((d['a'].dtype == np.dtype('int32')) or
                    (d['a'].dtype == np.dtype('int64')))
    self.assertEqual(d['b'].dtype, np.dtype(np.double))

    m, _, dims = to_matrix_with_info(d, np.double)

    self.assertTrue(isinstance(m, np.ndarray))
    self.assertEqual(m.dtype, np.dtype(np.double))
    self.assertEqual(m.shape[0], 50)
    self.assertEqual(m.shape[1], 2)
    colNames = list('ab')
    for i in range(2):
      for j in range(50):
        self.assertEqual(d[colNames[i]][j], m[j, i])

    self.assertEqual(dims.shape[0], 2)
    self.assertEqual(dims[0], 0)
    self.assertEqual(dims[1], 0)

  def testArraylikeToMatrix(self):
    """
    Test that if we pass some array, we get back the right thing.  This array
    will be filled with doubles only.
    """
    a = [[0.01, 0.02, 0.03],
         [0.04, 0.05, 0.06],
         [0.07, 0.08, 0.09],
         [0.10, 0.11, 0.12]]

    m, _, dims = to_matrix_with_info(a, np.double)

    self.assertTrue(isinstance(m, np.ndarray))
    self.assertEqual(m.dtype, np.dtype(np.double))
    self.assertEqual(m.shape[0], 4)
    self.assertEqual(m.shape[1], 3)

    for i in range(4):
      for j in range(3):
        self.assertEqual(a[i][j], m[i, j])

    self.assertEqual(dims.shape[0], 3)
    self.assertEqual(dims[0], 0)
    self.assertEqual(dims[1], 0)
    self.assertEqual(dims[2], 0)

  def testMultitypeArraylikeToMatrix(self):
    """
    Test that if we pass an array with multiple types, we get back the right
    thing.  The numpy ndarray should be filled with doubles only.
    """
    a = [[0.01, 0.02, 3],
         [0.04, 0.05, 6],
         [0.07, 0.08, 9],
         [0.10, 0.11, 12]]

    m, _, dims = to_matrix_with_info(a, np.double)

    self.assertTrue(isinstance(m, np.ndarray))
    self.assertEqual(m.dtype, np.dtype(np.double))
    self.assertEqual(m.shape[0], 4)
    self.assertEqual(m.shape[1], 3)

    for i in range(4):
      for j in range(3):
        self.assertEqual(a[i][j], m[i, j])

    self.assertEqual(dims.shape[0], 3)
    self.assertEqual(dims[0], 0)
    self.assertEqual(dims[1], 0)
    self.assertEqual(dims[2], 0)

  def testNumpyToMatrix(self):
    """
    Make sure we can convert a numpy matrix without copying anything.
    """
    m1 = np.random.randn(100, 5)
    m2, _, dims = to_matrix_with_info(m1, np.double)

    self.assertTrue(isinstance(m2, np.ndarray))
    self.assertEqual(m2.dtype, np.dtype(np.double))

    p1 = m1.__array_interface__
    p2 = m2.__array_interface__

    self.assertEqual(p1['data'], p2['data'])

    self.assertEqual(dims.shape[0], 5)
    self.assertEqual(dims[0], 0)
    self.assertEqual(dims[1], 0)
    self.assertEqual(dims[2], 0)
    self.assertEqual(dims[3], 0)
    self.assertEqual(dims[4], 0)

  def testCategoricalOnly(self):
    """
    Make sure that we can convert a categorical-only Pandas matrix.
    """
    d = pd.DataFrame({"A": ["a", "b", "c", "a"] })
    d["A"] = d["A"].astype('category') # Convert to categorical.

    m, _, dims = to_matrix_with_info(d, np.double)

    self.assertTrue(isinstance(m, np.ndarray))
    self.assertEqual(m.dtype, np.dtype(np.double))

    self.assertEqual(dims.shape[0], 1)
    self.assertEqual(dims[0], 1)

    self.assertEqual(m.shape[0], 4)
    self.assertEqual(m.shape[1], 1)
    self.assertEqual(m[0], m[3])
    self.assertTrue(m[0] != m[1])
    self.assertTrue(m[1] != m[2])
    self.assertTrue(m[0] != m[2])

def test_suite():
    """
    Run all tests.
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(TestToMatrix))
    suite.addTest(loader.loadTestsFromTestCase(TestToMatrixWithInfo))
    return suite

if __name__ == '__main__':
    unittest.main()
