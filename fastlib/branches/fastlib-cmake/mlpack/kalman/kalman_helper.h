/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file kalman_helper.h
 *
 * This contains helper functions which are repeatedly
 * used 
 *
 * @see kalman_helper.cc
 */

#ifndef KALMAN_HELPER_H
#define KALMAN_HELPER_H
#include "fastlib/fastlib.h"

/**
 * Create a struct for the params. of a causal 
 * linear dynamical system (lds)
 * x_{t+1} = a_mat*x_t + b_mat*u_t + w_t 
 * y_t     = c_mat*x_t + v_t
 * where
 * E(w_tw_t') = Q; E(v_tv_t') = R; E(w_tv_t') = S;
 *
 * Member matrices include a_mat, b_mat, c_mat
 * q_mat, r_mat, s_mat
 */
struct ssm {
  Matrix a_mat; 
  Matrix b_mat; 
  Matrix c_mat; 
  Matrix q_mat; 
  Matrix r_mat; 
  Matrix s_mat; 
};

/**
 * changes the value of v to 
 * a_mat*x + w, via a ptr
 */
void propagate_one_step(const Matrix& a_mat, const Vector& x, 
			const Vector& w, Vector* v);

/**
 * changes the value of v to 
 * returns a_mat*x + b_mat*u + w, via a ptr
 */
void propagate_one_step(const Matrix& a_mat, const Matrix& b_mat, 
			const Vector& x, const Vector &u, const Vector& w,
			Vector* v);

/** 
 * Changes values of matrix to
 * a_mat - b_mat*inv(c_mat)*d_mat 
 * via a ptr
 */
void schur(const Matrix& a_mat, const Matrix& b_mat, const Matrix& c_mat, 
	   const Matrix& d_mat, Matrix* mat);


/**
 * Prints matrix a_mat and its label "name" 
 */
void print_matrix(const Matrix& a_mat, const char* name);


/**
 * Initializes x_mat such that x_mat <- [a_mat | b_mat]
 */
void matrix_concatenate_col_init(const Matrix& a_mat, const Matrix& b_mat, 
				 Matrix* x_mat);


/**
 * Initializes x_mat such that x_mat <- [a_mat 
 *                                       --
 *                                    b_mat]
 */
void matrix_concatenate_row_init(const Matrix& a_mat, const Matrix& b_mat, 
				 Matrix* x_mat);


/**
 * x_mat is initialized to have the same values as a sub-matrix of a_mat
 * r(c) refers to the (r+1)-th((c+1)-th) row(column)
 * _in refers to the starting row(column)
 * _out refers to the ending row(column)
 */
void extract_sub_matrix_init(const Matrix& a_mat, const int& r_in, 
			     const int& r_out, const int& c_in, 
			     const int& c_out, Matrix* x_mat);


/**
 * x is initialized to have the same values as a sub-vector of vector v
 * r refers to the (r+1)-th row
 * _in refers to the starting row
 * _out refers to the ending row
 */
void extract_sub_vector_of_vector_init(const Vector& v, const int& r_in, 
				       const int& r_out, Vector* x);

/**
 * x is given  the same values as a sub-vector of vector v
 * r refers to the (r+1)-th row
 * _in refers to the starting row
 * _out refers to the ending row
 * done via a ptr
 */
void extract_sub_vector_of_vector(const Vector& v, const int& r_in, 
				  const int& r_out, Vector* x);


/**
 * Sets x_mat([r_in+1:r_out+1, c_in+1:c_out+1]) <-a_mat // in matlab notation
 * where
 * a_mat is a matrix or vector (as appropriate)
 * x_mat needs to be initialized a-priori and is passed as a ptr
 */
void set_portion_of_matrix(const Matrix& a_mat, const int& r_in,
			   const int& r_out, const int& c_in, 
			   const int& c_out, Matrix* x_mat);


/**
 * Sets x_mat(r_in+1:r_out+1, c+1) <-a_mat // in matlab notation
 * where
 * a_mat is a matrix or vector (as appropriate)
 * x_mat needs to be initialized a-priori and passed as a ptr
 */
void set_portion_of_matrix(const Vector& a_mat, const int& r_in, 
			   const int& r_out, const int& c, Matrix* x_mat);

/**
 * Returns vector realization of zero-mean multi-variate Gaussian 
 * with unit covariance; vector needs to be a-priori initialized
 * and passed by ref
 */
void RandVector(Vector &v);

/**
 * Returns vector realization of zero-mean multi-variate Gaussian 
 * with covariance noise_mat;vector needs to be a-priori initialized
 * and passed by ref
 */
void RandVector(const Matrix& noise_mat, Vector &v);   

#endif
