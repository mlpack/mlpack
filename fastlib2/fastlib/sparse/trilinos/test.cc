/*
 * =====================================================================================
 *
 *       Filename:  test.cc
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  11/30/2007 11:32:26 AM EST
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 *
 * =====================================================================================
 */

#define HAVE_CONFIG_H
#include "fastlib/fastlib.h"
#ifdef F77_FUNC
#undef F77_FUNC
#endif
#include "trilinos/include/Epetra_SerialDenseMatrix.h"
#include "trilinos/include/Epetra_SerialComm.h"
#include "trilinos/include/Epetra_Version.h"


int main(int argc, char *argv[]) {

  Epetra_SerialComm Comm;
  // declare two dense matrix, whose dimensions are still not specified
  Epetra_SerialDenseMatrix A, B;
  
  // Total number of rows ans columns for dense matrix A
  int NumRowsA = 2, NumColsA = 2;

  // shape A
  A.Shape( NumRowsA, NumColsA );

  // set the element of A using the () operator.
  // Note that i is the row-index, and j the column-index
  for( int i=0 ; i<NumRowsA ; ++i ) 
    for( int j=0 ; j<NumColsA ; ++j ) 
      A(i,j) = i+100*j;

  // Epetra_SerialDenseMatrix overloads the << operator
  cout << A;

  // get matrix norms
  cout << "Inf norm of A = " << A.OneNorm() << endl;
  cout << "One norm of A = " << A.InfNorm() << endl;

  // now define an other matrix, B, for matrix multiplication
  int NumRowsB = 2, NumColsB=1;
  B.Shape(NumRowsB, NumColsB);

  // enter the values of B
  for( int i=0 ; i<NumRowsB ; ++i ) 
    for( int j=0 ; j<NumColsB ; ++j ) 
      B(i,j) = 11.0+i+100*j;

  cout << B;

  // define the matrix which will hold A * B
  Epetra_SerialDenseMatrix AtimesB;

  // same number of rows than A, same columns than B
  AtimesB.Shape(NumRowsA,NumColsB);  

  // A * B
  AtimesB.Multiply('N','N',1.0, A, B, 0.0);
  cout << AtimesB;

#

}
