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

namespace la {

void rowMatrixAssemble(ArrayList<Matrix>& imageList, Matrix& mean,
					   Matrix& X_out);
void colMatrixAssemble(ArrayList<Matrix>& imageList, Matrix& mean,
					   Matrix& X_out);
void meanMatrix(ArrayList<Matrix>& imageList, Matrix& mean_out);

// Concatenate matrices in ``imageList'' row-wisely into ``X''
// Perform SVD on ``X''
// Return: colums of ``rowBasis_out'' are right eigenvectors of ``X''
void row2dPCA(ArrayList<Matrix>& imageList, 
			  Vector& eigenValues, Matrix& rowBasis_out, Matrix& mean_out) {
	Matrix X, U, VT;

	meanMatrix(imageList, mean_out);
	rowMatrixAssemble(imageList, mean_out, X);
	la::SVDInit(X, &eigenValues, &U, &VT);
	la::TransposeInit(VT, &rowBasis_out);
}

// Concatenate matrices in ``imageList'' column-wisely into ``X''
// Perform SVD on ``X''
// Return: colums of ``colBasis_out'' are right eigenvectors of ``X''

void col2dPCA(ArrayList<Matrix>& imageList, 
			  Vector& eigenValues, Matrix& colBasis_out, Matrix& mean_out) {
	Matrix X, U, VT, mean;

	meanMatrix(imageList, mean_out);
	colMatrixAssemble(imageList, mean_out, X);
	la::SVDInit(X, &eigenValues, &U, &VT);
	la::TransposeInit(VT, &colBasis_out);
}

void rowMatrixAssemble(ArrayList<Matrix>& imageList, Matrix& mean,
					   Matrix& X_out) {
	index_t n_matrix = imageList.size();
	DEBUG_ASSERT(n_matrix > 0);
	index_t n_rows = imageList[0].n_rows();
	index_t n_cols = imageList[0].n_cols();
	
	X_out.Init(n_matrix * n_rows, n_cols);
	
	for (index_t i_matrix = 0; i_matrix < n_matrix; i_matrix++) {
		DEBUG_ASSERT(imageList[i_matrix].n_rows() == n_rows);
		DEBUG_ASSERT(imageList[i_matrix].n_cols() == n_cols);
		for (index_t i_row = 0; i_row < n_rows; i_row++) 
			for (index_t i_col = 0; i_col < n_cols; i_col++)
				X_out.ref(i_matrix*n_rows+i_row, i_col) = 
					imageList[i_matrix].get(i_row, i_col) - mean.get(i_row, i_col);
	}
}

void colMatrixAssemble(ArrayList<Matrix>& imageList, Matrix& mean, 
					   Matrix& X_out) {
	index_t n_matrix = imageList.size();
	DEBUG_ASSERT(n_matrix > 0);
	index_t n_rows = imageList[0].n_rows();
	index_t n_cols = imageList[0].n_cols();
	
	X_out.Init(n_matrix * n_cols, n_rows);
	
	for (index_t i_matrix = 0; i_matrix < n_matrix; i_matrix++) {
		DEBUG_ASSERT(imageList[i_matrix].n_rows() == n_rows);
		DEBUG_ASSERT(imageList[i_matrix].n_cols() == n_cols);
		for (index_t i_row = 0; i_row < n_rows; i_row++) 
			for (index_t i_col = 0; i_col < n_cols; i_col++)
				X_out.ref(i_matrix*n_cols+i_col, i_row) = 
					imageList[i_matrix].get(i_row, i_col) - mean.get(i_row,i_col);
	}
}

void meanMatrix(ArrayList<Matrix>& imageList, Matrix& mean_out) {
	index_t n_matrix = imageList.size();
	DEBUG_ASSERT(n_matrix > 0);
	index_t n_rows = imageList[0].n_rows();
	index_t n_cols = imageList[0].n_cols();
	mean_out.Init(n_rows, n_cols);

	for (index_t i_matrix = 0; i_matrix < n_matrix; i_matrix++) {
		DEBUG_ASSERT(imageList[i_matrix].n_rows() == n_rows);
		DEBUG_ASSERT(imageList[i_matrix].n_cols() == n_cols);
		la::AddTo(imageList[i_matrix], &mean_out);
	}
	la::Scale(1.0/n_matrix, &mean_out);
}

}; // namespace
