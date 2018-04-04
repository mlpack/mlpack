#!/usr/bin/env python
"""
test_python_binding.py

Test that passing types to Python bindings works successfully.

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""
import unittest
import pandas as pd
import numpy as np
import copy

from mlpack.test_python_binding import test_python_binding

class TestPythonBinding(unittest.TestCase):
  """
  This class tests the basic functionality of the Python bindings.
  """

  def testRunBindingCorrectly(self):
    """
    Test that when we run the binding correctly (with correct input parameters),
    we get the expected output.
    """
    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 flag1=True)

    self.assertEqual(output['string_out'], 'hello2')
    self.assertEqual(output['int_out'], 13)
    self.assertEqual(output['double_out'], 5.0)

  def testRunBindingNoFlag(self):
    """
    If we forget the mandatory flag, we should get wrong results.
    """
    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0)

    self.assertNotEqual(output['string_out'], 'hello2')
    self.assertNotEqual(output['int_out'], 13)
    self.assertNotEqual(output['double_out'], 5.0)

  def testRunBindingWrongString(self):
    """
    If we give the wrong string, we should get wrong results.
    """
    output = test_python_binding(string_in='goodbye',
                                 int_in=12,
                                 double_in=4.0,
                                 flag1=True)

    self.assertNotEqual(output['string_out'], 'hello2')

  def testRunBindingWrongInt(self):
    """
    If we give the wrong int, we should get wrong results.
    """
    output = test_python_binding(string_in='hello',
                                 int_in=15,
                                 double_in=4.0,
                                 flag1=True)

    self.assertNotEqual(output['int_out'], 13)

  def testRunBindingWrongDouble(self):
    """
    If we give the wrong double, we should get wrong results.
    """
    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=2.0,
                                 flag1=True)

    self.assertNotEqual(output['double_out'], 5.0)

  def testRunBadFlag(self):
    """
    If we give the second flag, this should fail.
    """
    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 flag1=True,
                                 flag2=True)

    self.assertNotEqual(output['string_out'], 'hello2')
    self.assertNotEqual(output['int_out'], 13)
    self.assertNotEqual(output['double_out'], 5.0)

  def testNumpyMatrix(self):
    """
    The matrix we pass in, we should get back with the third dimension doubled
    and the fifth forgotten.
    """
    x = np.random.rand(100, 5);
    z = copy.copy(x)

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 matrix_in=z)

    self.assertEqual(output['matrix_out'].shape[0], 100)
    self.assertEqual(output['matrix_out'].shape[1], 4)
    self.assertEqual(output['matrix_out'].dtype, np.double)
    for i in [0, 1, 3]:
      for j in range(100):
        self.assertEqual(x[j, i], output['matrix_out'][j, i])

    for j in range(100):
      self.assertEqual(2 * x[j, 2], output['matrix_out'][j, 2])

  def testNumpyMatrixForceCopy(self):
    """
    The matrix we pass in, we should get back with the third dimension doubled
    and the fifth forgotten.
    """
    x = np.random.rand(100, 5);

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 matrix_in=x,
                                 copy_all_inputs=True)

    self.assertEqual(output['matrix_out'].shape[0], 100)
    self.assertEqual(output['matrix_out'].shape[1], 4)
    self.assertEqual(output['matrix_out'].dtype, np.double)
    for i in [0, 1, 3]:
      for j in range(100):
        self.assertEqual(x[j, i], output['matrix_out'][j, i])

    for j in range(100):
      self.assertEqual(2 * x[j, 2], output['matrix_out'][j, 2])

  def testArraylikeMatrix(self):
    """
    Test that we can pass an arraylike matrix.
    """
    x = [[1, 2, 3, 4, 5],
         [6, 7, 8, 9, 10],
         [11, 12, 13, 14, 15]]

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 matrix_in=x)

    self.assertEqual(output['matrix_out'].shape[0], 3)
    self.assertEqual(output['matrix_out'].shape[1], 4)
    self.assertEqual(output['matrix_out'].dtype, np.double)
    self.assertEqual(output['matrix_out'][0, 0], 1)
    self.assertEqual(output['matrix_out'][0, 1], 2)
    self.assertEqual(output['matrix_out'][0, 2], 6)
    self.assertEqual(output['matrix_out'][0, 3], 4)
    self.assertEqual(output['matrix_out'][1, 0], 6)
    self.assertEqual(output['matrix_out'][1, 1], 7)
    self.assertEqual(output['matrix_out'][1, 2], 16)
    self.assertEqual(output['matrix_out'][1, 3], 9)
    self.assertEqual(output['matrix_out'][2, 0], 11)
    self.assertEqual(output['matrix_out'][2, 1], 12)
    self.assertEqual(output['matrix_out'][2, 2], 26)
    self.assertEqual(output['matrix_out'][2, 3], 14)

  def testArraylikeMatrixForceCopy(self):
    """
    Test that we can pass an arraylike matrix.
    """
    x = [[1, 2, 3, 4, 5],
         [6, 7, 8, 9, 10],
         [11, 12, 13, 14, 15]]

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 matrix_in=x,
                                 copy_all_inputs=True)

    self.assertEqual(output['matrix_out'].shape[0], 3)
    self.assertEqual(output['matrix_out'].shape[1], 4)
    self.assertEqual(len(x), 3)
    self.assertEqual(len(x[0]), 5)
    self.assertEqual(output['matrix_out'].dtype, np.double)
    self.assertEqual(output['matrix_out'][0, 0], 1)
    self.assertEqual(output['matrix_out'][0, 1], 2)
    self.assertEqual(output['matrix_out'][0, 2], 6)
    self.assertEqual(output['matrix_out'][0, 3], 4)
    self.assertEqual(output['matrix_out'][1, 0], 6)
    self.assertEqual(output['matrix_out'][1, 1], 7)
    self.assertEqual(output['matrix_out'][1, 2], 16)
    self.assertEqual(output['matrix_out'][1, 3], 9)
    self.assertEqual(output['matrix_out'][2, 0], 11)
    self.assertEqual(output['matrix_out'][2, 1], 12)
    self.assertEqual(output['matrix_out'][2, 2], 26)
    self.assertEqual(output['matrix_out'][2, 3], 14)

  def testNumpyUmatrix(self):
    """
    Same as testNumpyMatrix() but with an unsigned matrix.
    """
    x = np.random.randint(0, high=500, size=[100, 5])
    z = copy.copy(x)

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 umatrix_in=z)

    self.assertEqual(output['umatrix_out'].shape[0], 100)
    self.assertEqual(output['umatrix_out'].shape[1], 4)
    self.assertEqual(output['umatrix_out'].dtype, np.long)
    for i in [0, 1, 3]:
      for j in range(100):
        self.assertEqual(x[j, i], output['umatrix_out'][j, i])

    for j in range(100):
      self.assertEqual(2 * x[j, 2], output['umatrix_out'][j, 2])

  def testNumpyUmatrixForceCopy(self):
    """
    Same as testNumpyMatrix() but with an unsigned matrix.
    """
    x = np.random.randint(0, high=500, size=[100, 5])

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 umatrix_in=x,
                                 copy_all_inputs=True)

    self.assertEqual(output['umatrix_out'].shape[0], 100)
    self.assertEqual(output['umatrix_out'].shape[1], 4)
    self.assertEqual(output['umatrix_out'].dtype, np.long)
    for i in [0, 1, 3]:
      for j in range(100):
        self.assertEqual(x[j, i], output['umatrix_out'][j, i])

    for j in range(100):
      self.assertEqual(2 * x[j, 2], output['umatrix_out'][j, 2])

  def testArraylikeUmatrix(self):
    """
    Test that we can pass an arraylike unsigned matrix.
    """
    x = [[1, 2, 3, 4, 5],
         [6, 7, 8, 9, 10],
         [11, 12, 13, 14, 15]]

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 umatrix_in=x)

    self.assertEqual(output['umatrix_out'].shape[0], 3)
    self.assertEqual(output['umatrix_out'].shape[1], 4)
    self.assertEqual(output['umatrix_out'].dtype, np.long)
    self.assertEqual(output['umatrix_out'][0, 0], 1)
    self.assertEqual(output['umatrix_out'][0, 1], 2)
    self.assertEqual(output['umatrix_out'][0, 2], 6)
    self.assertEqual(output['umatrix_out'][0, 3], 4)
    self.assertEqual(output['umatrix_out'][1, 0], 6)
    self.assertEqual(output['umatrix_out'][1, 1], 7)
    self.assertEqual(output['umatrix_out'][1, 2], 16)
    self.assertEqual(output['umatrix_out'][1, 3], 9)
    self.assertEqual(output['umatrix_out'][2, 0], 11)
    self.assertEqual(output['umatrix_out'][2, 1], 12)
    self.assertEqual(output['umatrix_out'][2, 2], 26)
    self.assertEqual(output['umatrix_out'][2, 3], 14)

  def testArraylikeUmatrixForceCopy(self):
    """
    Test that we can pass an arraylike unsigned matrix.
    """
    x = [[1, 2, 3, 4, 5],
         [6, 7, 8, 9, 10],
         [11, 12, 13, 14, 15]]

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 umatrix_in=x,
                                 copy_all_inputs=True)

    self.assertEqual(output['umatrix_out'].shape[0], 3)
    self.assertEqual(output['umatrix_out'].shape[1], 4)
    self.assertEqual(len(x), 3)
    self.assertEqual(len(x[0]), 5)
    self.assertEqual(output['umatrix_out'].dtype, np.long)
    self.assertEqual(output['umatrix_out'][0, 0], 1)
    self.assertEqual(output['umatrix_out'][0, 1], 2)
    self.assertEqual(output['umatrix_out'][0, 2], 6)
    self.assertEqual(output['umatrix_out'][0, 3], 4)
    self.assertEqual(output['umatrix_out'][1, 0], 6)
    self.assertEqual(output['umatrix_out'][1, 1], 7)
    self.assertEqual(output['umatrix_out'][1, 2], 16)
    self.assertEqual(output['umatrix_out'][1, 3], 9)
    self.assertEqual(output['umatrix_out'][2, 0], 11)
    self.assertEqual(output['umatrix_out'][2, 1], 12)
    self.assertEqual(output['umatrix_out'][2, 2], 26)
    self.assertEqual(output['umatrix_out'][2, 3], 14)

  def testCol(self):
    """
    Test a column vector input parameter.
    """
    x = np.random.rand(100)
    z = copy.copy(x)

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 col_in=z)

    self.assertEqual(output['col_out'].shape[0], 100)
    self.assertEqual(output['col_out'].dtype, np.double)

    for i in range(100):
      self.assertEqual(output['col_out'][i], x[i] * 2)

  def testColForceCopy(self):
    """
    Test a column vector input parameter.
    """
    x = np.random.rand(100)

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 col_in=x,
                                 copy_all_inputs=True)

    self.assertEqual(output['col_out'].shape[0], 100)
    self.assertEqual(output['col_out'].dtype, np.double)

    for i in range(100):
      self.assertEqual(output['col_out'][i], x[i] * 2)

  def testUcol(self):
    """
    Test an unsigned column vector input parameter.
    """
    x = np.random.randint(0, high=500, size=100)
    z = copy.copy(x)

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 ucol_in=z)

    self.assertEqual(output['ucol_out'].shape[0], 100)
    self.assertEqual(output['ucol_out'].dtype, np.long)
    for i in range(100):
      self.assertEqual(output['ucol_out'][i], x[i] * 2)

  def testUcolForceCopy(self):
    """
    Test an unsigned column vector input parameter.
    """
    x = np.random.randint(0, high=500, size=100)

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 ucol_in=x,
                                 copy_all_inputs=True)

    self.assertEqual(output['ucol_out'].shape[0], 100)
    self.assertEqual(output['ucol_out'].dtype, np.long)
    for i in range(100):
      self.assertEqual(output['ucol_out'][i], x[i] * 2)

  def testRow(self):
    """
    Test a row vector input parameter.
    """
    x = np.random.rand(100)
    z = copy.copy(x)

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 row_in=z)

    self.assertEqual(output['row_out'].shape[0], 100)
    self.assertEqual(output['row_out'].dtype, np.double)

    for i in range(100):
      self.assertEqual(output['row_out'][i], x[i] * 2)

  def testRowForceCopy(self):
    """
    Test a row vector input parameter.
    """
    x = np.random.rand(100)

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 row_in=x,
                                 copy_all_inputs=True)

    self.assertEqual(output['row_out'].shape[0], 100)
    self.assertEqual(output['row_out'].dtype, np.double)

    for i in range(100):
      self.assertEqual(output['row_out'][i], x[i] * 2)

  def testUrow(self):
    """
    Test an unsigned row vector input parameter.
    """
    x = np.random.randint(0, high=500, size=100)
    z = copy.copy(x)

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 urow_in=z)

    self.assertEqual(output['urow_out'].shape[0], 100)
    self.assertEqual(output['urow_out'].dtype, np.long)

    for i in range(100):
      self.assertEqual(output['urow_out'][i], x[i] * 2)

  def testUrowForceCopy(self):
    """
    Test an unsigned row vector input parameter.
    """
    x = np.random.randint(0, high=500, size=100)

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 urow_in=x,
                                 copy_all_inputs=True)

    self.assertEqual(output['urow_out'].shape[0], 100)
    self.assertEqual(output['urow_out'].dtype, np.long)

    for i in range(100):
      self.assertEqual(output['urow_out'][i], x[i] * 2)

  def testMatrixAndInfoNumpy(self):
    """
    Test that we can pass a matrix with all numeric features.
    """
    x = np.random.rand(100, 10)
    z = copy.copy(x)

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 matrix_and_info_in=z)

    self.assertEqual(output['matrix_and_info_out'].shape[0], 100)
    self.assertEqual(output['matrix_and_info_out'].shape[1], 10)

    for i in range(10):
      for j in range(100):
        self.assertEqual(output['matrix_and_info_out'][j, i], x[j, i] * 2.0)

  def testMatrixAndInfoNumpyForceCopy(self):
    """
    Test that we can pass a matrix with all numeric features.
    """
    x = np.random.rand(100, 10)

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 matrix_and_info_in=x,
                                 copy_all_inputs=True)

    self.assertEqual(output['matrix_and_info_out'].shape[0], 100)
    self.assertEqual(output['matrix_and_info_out'].shape[1], 10)

    for i in range(10):
      for j in range(100):
        self.assertEqual(output['matrix_and_info_out'][j, i], x[j, i] * 2.0)

  def testMatrixAndInfoPandas(self):
    """
    Test that we can pass a matrix with some categorical features.
    """
    x = pd.DataFrame(np.random.rand(10, 4), columns=list('abcd'))
    x['e'] = pd.Series(['a', 'b', 'c', 'd', 'a', 'b', 'e', 'c', 'a', 'b'],
        dtype='category')
    z = copy.copy(x)

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 matrix_and_info_in=z)

    self.assertEqual(output['matrix_and_info_out'].shape[0], 10)
    self.assertEqual(output['matrix_and_info_out'].shape[1], 5)

    cols = list('abcde')

    for i in range(4):
      for j in range(10):
        self.assertEqual(output['matrix_and_info_out'][j, i], z[cols[i]][j] * 2)

    for j in range(10):
      self.assertEqual(output['matrix_and_info_out'][j, 4], z[cols[4]][j])

  def testMatrixAndInfoPandasForceCopy(self):
    """
    Test that we can pass a matrix with some categorical features.
    """
    x = pd.DataFrame(np.random.rand(10, 4), columns=list('abcd'))
    x['e'] = pd.Series(['a', 'b', 'c', 'd', 'a', 'b', 'e', 'c', 'a', 'b'],
        dtype='category')

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 matrix_and_info_in=x,
                                 copy_all_inputs=True)

    self.assertEqual(output['matrix_and_info_out'].shape[0], 10)
    self.assertEqual(output['matrix_and_info_out'].shape[1], 5)

    cols = list('abcde')

    for i in range(4):
      for j in range(10):
        self.assertEqual(output['matrix_and_info_out'][j, i], x[cols[i]][j] * 2)

    for j in range(10):
      self.assertEqual(output['matrix_and_info_out'][j, 4], x[cols[4]][j])

  def testIntVector(self):
    """
    Test that we can pass a vector of ints and get back that same vector but
    with the last element removed.
    """
    x = [1, 2, 3, 4, 5]

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 vector_in=x)

    self.assertEqual(output['vector_out'], [1, 2, 3, 4])

  def testStringVector(self):
    """
    Test that we can pass a vector of strings and get back that same vector but
    with the last element removed.
    """
    x = ['one', 'two', 'three', 'four', 'five']

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 str_vector_in=x)

    self.assertEqual(output['str_vector_out'],
        ['one', 'two', 'three', 'four'])

  def testModel(self):
    """
    First create a GaussianKernel object, then send it back and make sure we get
    the right double value.
    """
    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 build_model=True)

    output2 = test_python_binding(string_in='hello',
                                  int_in=12,
                                  double_in=4.0,
                                  model_in=output['model_out'])

    self.assertEqual(output2['model_bw_out'], 20.0)

  def testModelForceCopy(self):
    """
    First create a GaussianKernel object, then send it back and make sure we get
    the right double value.
    """
    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 build_model=True)

    output2 = test_python_binding(string_in='hello',
                                  int_in=12,
                                  double_in=4.0,
                                  model_in=output['model_out'],
                                  copy_all_inputs=True)

    output3 = test_python_binding(string_in='hello',
                                  int_in=12,
                                  double_in=4.0,
                                  model_in=output['model_out'])

    self.assertEqual(output2['model_bw_out'], 20.0)
    self.assertEqual(output3['model_bw_out'], 20.0)

if __name__ == '__main__':
  unittest.main()
