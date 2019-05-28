#!/usr/bin/env python
"""
@file test_python_model.py
@author Mehul Kumar Nirala

Test that python interpretation of mlpack models works successfully.

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""
import unittest
import pandas as pd
import numpy as np
from mlpack import logistic_regression, perceptron
from mlpack import pythonize_model

# Fake test data
x = np.random.rand(10,10)
y = [1,0,1,0,1,0,1,0,0,0]

class TestPythonInterpretation(unittest.TestCase):
  """
  This class tests the basic functionality of the Python interpretation model.
  """

  def testRunForLogisticPerceptron(self):
    """
    Test the output parameters for Perceptron.
    """
    output = perceptron(labels=y, training=x)
    python_interpretation = pythonize_model.transform(output['output_model'])

    weights = python_interpretation['p']['weights']
    biases = python_interpretation['p']['biases']

    self.assertEqual(len(weights['item']), 20)
    self.assertEqual(len(biases['item']), 2)
    self.assertEqual(weights['vec_state'], 'matrix')
    self.assertEqual(weights['n_elem'], 2.0 * 10.0)
    self.assertEqual(weights['n_cols'], 2.0)
    self.assertEqual(weights['n_rows'], 10.0)
    self.assertEqual(biases['vec_state'], 'column vector')
    self.assertEqual(biases['n_elem'], 1.0 * 2.0)
    self.assertEqual(biases['n_cols'], 1.0)
    self.assertEqual(biases['n_rows'], 2.0)

  def testRunForLogisticRegression(self):
    """
    Test the output parameters for Lgistic Regression.
    """
    output = logistic_regression(labels=y, training=x)
    python_interpretation = pythonize_model.transform(output['output_model'])
    weights = python_interpretation['parameters']['item']
    self.assertEqual(len(weights), 11)
    self.assertEqual(python_interpretation['parameters']['n_cols'], 11.0)
    self.assertEqual(python_interpretation['parameters']['n_rows'], 1.0)

if __name__ == '__main__':
  unittest.main()
