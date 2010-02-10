//      2dPCA.cpp
//      
//      Copyright 2010 Long <tqlong@gmail.com>
//      
//      This program is free software; you can redistribute it and/or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation; either version 2 of the License, or
//      (at your option) any later version.
//      
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//      GNU General Public License for more details.
//      
//      You should have received a copy of the GNU General Public License
//      along with this program; if not, write to the Free Software
//      Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//      MA 02110-1301, USA.

#include <fastlib/fastlib.h>
#include "2dfactor.h"
#include <iostream>

namespace la {

  void meanMatrix(const ArrayList<Matrix>& imageList, Matrix& mean_out);
  void computeRowCovariance(const ArrayList<Matrix>& imageList, 
			    const Matrix& mean, 
			    Matrix& rowCovariance_out);
  void computeColCovariance(const ArrayList<Matrix>& imageList, 
			    const Matrix& mean, 
			    Matrix& colCovariance_out);
  // Perform EVD on covariance matrix of rows in ``imageList''
  // Return: colums of ``rowBasis_out'' are right eigenvectors of ``X''
  void row2dPCA(const ArrayList<Matrix>& imageList, 
		Vector& eigenValues_out, Matrix& rowBasis_out, Matrix& mean_out) {
    Matrix rowCovariance;

    //ot::Print(imageList[0]);
    //ot::Print(imageList[imageList.size()-1]);

    meanMatrix(imageList, mean_out);
    std::cout << "--1" << std::endl;
    computeRowCovariance(imageList, mean_out, rowCovariance);
    std::cout << "--2 X:" << rowCovariance.n_rows() 
	      << " x " << rowCovariance.n_cols() << std::endl;
    //ot::Print(rowCovariance);
    la::EigenvectorsInit(rowCovariance, &eigenValues_out, &rowBasis_out);
    std::cout << "--3" << std::endl;
    //ot::Print(rowBasis_out);
  }

  // Concatenate matrices in ``imageList'' column-wisely into ``X''
  // Perform SVD on ``X''
  // Return: colums of ``colBasis_out'' are right eigenvectors of ``X''

  void col2dPCA(const ArrayList<Matrix>& imageList, 
		Vector& eigenValues_out, Matrix& colBasis_out, Matrix& mean_out) {
    Matrix colCovariance;

    meanMatrix(imageList, mean_out);
    std::cout << "--1" << std::endl;
    computeColCovariance(imageList, mean_out, colCovariance);
    std::cout << "--2 X:" << colCovariance.n_rows() 
	      << " x " << colCovariance.n_cols() << std::endl;
    la::EigenvectorsInit(colCovariance, &eigenValues_out, &colBasis_out);
    std::cout << "--3" << std::endl;
  }

  void RowCol2dPCA(const ArrayList<Matrix>& imageList, 
		   Vector& rowEigenValues, Matrix& rowBasis, 
		   Vector& colEigenValues, Matrix& colBasis, Matrix& mean) {
    Matrix tmp;
    la::row2dPCA(imageList, rowEigenValues, rowBasis, tmp);
    ot::Print(rowBasis);
    la::col2dPCA(imageList, colEigenValues, colBasis, mean);
    ot::Print(colBasis);
  }

  void Get2dBasisMajor(double rowPart, const Vector& rowEigenValues, 
		       const Matrix& rowBasis, double colPart, 
		       const Vector& colEigenValues, const Matrix& colBasis,
		       Matrix& rowBasisMajor_out, Matrix& colBasisMajor_out) {
    double sCol = 0, pCol = 0;
    for (index_t col = 0; col < rowEigenValues.length(); col++)
      sCol += rowEigenValues[col];
    for (index_t col = 0; col < rowEigenValues.length(); col++) {
      pCol += rowEigenValues[col];
      if (pCol / sCol >= rowPart) {
	rowBasisMajor_out.Init(rowBasis.n_rows(), col+1);
	rowBasisMajor_out.CopyColumnFromMat(0, 0, col+1, rowBasis);
	break;
      }
    }

    double sRow = 0, pRow = 0;
    for (index_t row = 0; row < colEigenValues.length(); row++)
      sRow += colEigenValues[row];
    for (index_t row = 0; row < colEigenValues.length(); row++) {
      pRow += colEigenValues[row];
      if (pRow / sRow >= colPart) {
	colBasisMajor_out.Init(colBasis.n_rows(), row+1);
	colBasisMajor_out.CopyColumnFromMat(0, 0, row+1, colBasis);
	break;
      }
    }
  }

  void Project2dBasis(const Matrix& A, const Matrix& rowBasis, 
		      const Matrix& colBasis, Matrix& A_out) {
    A_out.Init(A.n_rows(), A.n_cols());
    A_out.SetZero();
  
    index_t n_rows = colBasis.n_cols();
    index_t n_cols = rowBasis.n_cols();

    Matrix tmp, Coeffs;
    la::MulTransAInit(colBasis, A, &tmp);
    la::MulInit(tmp, rowBasis, &Coeffs);

    for (index_t i_row = 0; i_row < n_rows; i_row++) {
      Vector col;
      colBasis.MakeColumnVector(i_row, &col);
      for (index_t i_col = 0; i_col < n_cols; i_col++) {
	Vector row;
	rowBasis.MakeColumnVector(i_col, &row);
	for (index_t A_row = 0; A_row < A.n_rows(); A_row++)
	  for (index_t A_col = 0; A_col < A.n_cols(); A_col++)
	    A_out.ref(A_row, A_col) += 
	      Coeffs.get(i_row, i_col) * col[A_row] * row[A_col];
      }
    }
  }

  void computeRowCovariance(const ArrayList<Matrix>& imageList, 
			    const Matrix& mean, 
			    Matrix& rowCovariance_out) {
    index_t n_matrix = imageList.size();
    DEBUG_ASSERT(n_matrix > 0);
    index_t n_rows = imageList[0].n_rows();
    index_t n_cols = imageList[0].n_cols();

    rowCovariance_out.Init(n_cols, n_cols);
    rowCovariance_out.SetZero();

    for (index_t i_matrix = 0; i_matrix < n_matrix; i_matrix++) {
      DEBUG_ASSERT(imageList[i_matrix].n_rows() == n_rows);
      DEBUG_ASSERT(imageList[i_matrix].n_cols() == n_cols);
      for (index_t i_row = 0; i_row < n_rows; i_row++) 
	for (index_t i_col1 = 0; i_col1 < n_cols; i_col1++)
	  for (index_t i_col2 = 0; i_col2 < n_cols; i_col2++)     
	    rowCovariance_out.ref(i_col1, i_col2) +=
	      (imageList[i_matrix].get(i_row, i_col1)-mean.get(i_row,i_col1))*
 	      (imageList[i_matrix].get(i_row, i_col2)-mean.get(i_row,i_col2))
	      /n_matrix/n_rows;
    }
  }

  void computeColCovariance(const ArrayList<Matrix>& imageList, 
			    const Matrix& mean, 
			    Matrix& colCovariance_out) {
    index_t n_matrix = imageList.size();
    DEBUG_ASSERT(n_matrix > 0);
    index_t n_rows = imageList[0].n_rows();
    index_t n_cols = imageList[0].n_cols();

    colCovariance_out.Init(n_rows, n_rows);
    colCovariance_out.SetZero();

    for (index_t i_matrix = 0; i_matrix < n_matrix; i_matrix++) {
      DEBUG_ASSERT(imageList[i_matrix].n_rows() == n_rows);
      DEBUG_ASSERT(imageList[i_matrix].n_cols() == n_cols);
      for (index_t i_col = 0; i_col < n_cols; i_col++)
	for (index_t i_row1 = 0; i_row1 < n_rows; i_row1++) 
	  for (index_t i_row2 = 0; i_row2 < n_rows; i_row2++) 
	    colCovariance_out.ref(i_row1, i_row2) +=
	      (imageList[i_matrix].get(i_row1, i_col)-mean.get(i_row1,i_col))*
 	      (imageList[i_matrix].get(i_row2, i_col)-mean.get(i_row2,i_col))
	      /n_matrix/n_cols;
    }
  }

  void meanMatrix(const ArrayList<Matrix>& imageList, Matrix& mean_out) {
    index_t n_matrix = imageList.size();
    DEBUG_ASSERT(n_matrix > 0);
    index_t n_rows = imageList[0].n_rows();
    index_t n_cols = imageList[0].n_cols();
    mean_out.Init(n_rows, n_cols);
    mean_out.SetZero();

    for (index_t i_matrix = 0; i_matrix < n_matrix; i_matrix++) {
      DEBUG_ASSERT(imageList[i_matrix].n_rows() == n_rows);
      DEBUG_ASSERT(imageList[i_matrix].n_cols() == n_cols);
      la::AddTo(imageList[i_matrix], &mean_out);
    }
    la::Scale(1.0/n_matrix, &mean_out);
  }

}; // namespace
