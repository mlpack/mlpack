/**
 * @file uselapack_test.h
 *
 * Tests for LAPACK integration.
 */
#include<stdio.h>
#include "uselapack_test.h"

int main(int argc, char *argv[]) {
  NOTIFY("Testing float LAPACK");  
  TestVector<float>();
  TestMatrix<float>();
  TestVectorDot<float>();
  TestVectorSimpleMath<float>();
  TestMatrixSimpleMath<float>();
  TestMultiply<float>();
  TestInverse<float>();
  TestDeterminant<float>();
  TestQR<float>();
  TestEigen<float>();
  TestSchur<float>();
  TestSVD<float>();
  TestCholesky<float>();
  TestSolve<float>();
  TestLeastSquareFit<float>();
  NOTIFY("Testing double LAPACK\n");  
  TestVector<double>();
  TestMatrix<double>();
  TestVectorDot<double>();
  TestVectorSimpleMath<double>();
  TestMatrixSimpleMath<double>();
  TestMultiply<double>();
  TestInverse<double>();
  TestDeterminant<double>();
  TestQR<double>();
  TestEigen<double>();
  TestSchur<double>();
  TestSVD<double>();
  TestCholesky<double>();
  TestSolve<double>();
  TestLeastSquareFit<double>();

  NOTIFY("ALL TESTS PASSED!!!!\n");    
}

