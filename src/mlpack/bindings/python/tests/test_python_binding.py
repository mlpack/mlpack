#!/usr/bin/env python
"""
test_python_binding.py

Test that passing types to Python bindings works successfully.
"""
import unittest
import pandas as pd
import numpy as np

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

    output = test_python_binding(string_in='hello',
                                 int_in=12,
                                 double_in=4.0,
                                 matrix_in=x)

    self.assertEqual(output['matrix_out'].shape[0], 100)
    self.assertEqual(output['matrix_out'].shape[1], 4)
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

if __name__ == '__main__':
  unittest.main()
