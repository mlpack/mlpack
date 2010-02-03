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
#include <iostream>
#include "kalman_helper.h"
#include "fastlib/fastlib.h"

using namespace std;

void propagate_one_step(const Matrix& a_mat, const Vector& x, 
			  const Vector& w, Vector* v) {
  Vector temp_1;
  la::MulInit(a_mat, x, &temp_1); 
  la::AddTo(w, &temp_1); 
  (*v).CopyValues(temp_1);
}; 

void propagate_one_step(const Matrix& a_mat, const Matrix& b_mat, 
			const Vector& x, const Vector &u, const Vector& w,
			Vector* v) {
  Vector temp_1, temp_2;
  la::MulInit(a_mat, x, &temp_1); 
  la::MulInit(b_mat, u, &temp_2);
  la::AddTo(temp_2, &temp_1); 
  la::AddTo(w, &temp_1); 
  (*v).CopyValues(temp_1);
}; 

void schur(const Matrix& a_mat, const Matrix& b_mat, const Matrix& c_mat, 
	   const Matrix& d_mat, Matrix* mat) {
  Matrix temp_1, temp_2, temp_3; 
  temp_1.Copy(c_mat); la::Inverse(&temp_1); // temp1 = inv(c_mat)
  la::MulInit(b_mat, temp_1, &temp_2); // temp_2 = b_mat*inv(c_mat)
  la::MulInit(temp_2, d_mat, &temp_3); // temp_3 = b_mat*inv(c_mat)*d_mat
  Matrix result; 
  result.Copy(a_mat);
  la::AddExpert(-1, temp_3, &result); 
  (*mat).CopyValues(result);
};

void print_matrix(const Matrix& a_mat, const char* name)
{
  std::cout<<std::endl<<"Printing Matrix.."<<name<<std::endl;;
  for (int r = 1; r<=(int)a_mat.n_rows();r++)
    {
      for (int c=1; c<=(int)a_mat.n_cols();c++)
	{
	  std::cout<<" "<<a_mat.get(r-1, c-1)<<" ";
	};
      std::cout<<std::endl<<std::endl;
    };
};

void matrix_concatenate_col_init(const Matrix& a_mat, const Matrix& b_mat, 
				 Matrix* x_mat) {
  int n_rows = a_mat.n_rows(), n_cols = a_mat.n_cols() + b_mat.n_cols();
  (*x_mat).Init(n_rows, n_cols);
  
  for (int r1 = 0; r1<n_rows; r1++) {
    for (int c1 = 0; c1<a_mat.n_cols(); c1++)
      {
	(*x_mat).set(r1, c1, a_mat.get(r1,c1));
      };
  };
  
  for (int r2 = 0; r2<n_rows; r2++) {
    for (int c2 = a_mat.n_cols(); c2<n_cols; c2++)
      {
	(*x_mat).set(r2, c2, b_mat.get(r2, c2-a_mat.n_cols() ));
      };
  };  
};


void matrix_concatenate_row_init(const Matrix& a_mat, const Matrix& b_mat, Matrix* x_mat) {
  int n_cols = a_mat.n_rows(), n_rows = a_mat.n_rows() + b_mat.n_rows();
  (*x_mat).Init(n_rows, n_cols);
  
  for (int r1 = 0; r1<a_mat.n_rows(); r1++) {
    for (int c1 = 0; c1<n_cols; c1++) {
      (*x_mat).set(r1, c1, a_mat.get(r1, c1));
    };
  };
  
  for (int r2 = a_mat.n_rows(); r2<n_rows; r2++) {
    for (int c2 = 0; c2<n_cols; c2++) {
      (*x_mat).set(r2, c2, b_mat.get(r2-a_mat.n_rows(), c2));
    };
  };  
};

void extract_sub_matrix_init(const Matrix& a_mat, const int& r_in, 
			     const int& r_out, const int& c_in, 
			     const int& c_out, Matrix* x_mat) {
  (*x_mat).Init(r_out-r_in+1, c_out-c_in +1);
  
  for (int r = 0; r< (*x_mat).n_rows(); r++) {
    for (int c = 0; c< (*x_mat).n_cols(); c++) {
      (*x_mat).set(r, c, a_mat.get(r_in +r, c_in +c));
    };
  };
};

void extract_sub_vector_of_vector_init(const Vector& v, const int& r_in, 
				       const int& r_out, Vector* x) {
  (*x).Init(r_out-r_in+1);
  double temp [r_out-r_in+1];
  for (int r = 0; r< (*x).length(); r++) {
    temp[r] = v.get(r_in+r);
  };
  (*x).CopyValues(temp); 
};

void extract_sub_vector_of_vector(const Vector& v, const int& r_in, 
				  const int& r_out, Vector* x) {
  double temp [r_out-r_in+1];
  for (int r = 0; r< (*x).length(); r++) {
    temp[r] = v.get(r_in+r);
  };
  (*x).CopyValues(temp); 
};

void set_portion_of_matrix(const Matrix& a_mat, const int& r_in,
			   const int& r_out, const int& c_in, 
			   const int& c_out, Matrix* x_mat) {
  for (int r=r_in; r<=r_out; r++) {
    for (int c=c_in; c<=c_out; c++) {
      (*x_mat).set(r, c, a_mat.get(r-r_in, c-c_in));
    };
  };  
};

void set_portion_of_matrix(const Vector& a, const int& r_in,const int& r_out, 
			   const int& c, Matrix* x_mat) {
  for (int r=r_in; r<=r_out; r++) {
    (*x_mat).set(r, c, a.get(r-r_in));
  };  
};

void RandVector(Vector &v) {  
  index_t d = v.length();
  v.SetZero();
  
  for(index_t i = 0; i+1 < d; i+=2) {
    double a = drand48();
    double b = drand48();
    double first_term = sqrt(-2 * log(a));
    double second_term = 2 * M_PI * b;
    v[i] =   first_term * cos(second_term);
    v[i+1] = first_term * sin(second_term);
  };
  
  if((d % 2) == 1) {
    v[d - 1] = sqrt(-2 * log(drand48())) * cos(2 * M_PI * drand48());
  };  
};

void RandVector(const Matrix& noise_mat, Vector &v) {  
  index_t d = v.length();
  v.SetZero();
  
  for(index_t i = 0; i+1 < d; i+=2) {
    double a = drand48();
    double b = drand48();
    double first_term = sqrt(-2 * log(a));
    double second_term = 2 * M_PI * b;
    v[i] =   first_term * cos(second_term);
    v[i+1] = first_term * sin(second_term);
  };
  
  if((d % 2) == 1) {
    v[d - 1] = sqrt(-2 * log(drand48())) * cos(2 * M_PI * drand48());
  };
  
  Matrix noise_mat_rt, noise_mat_rt_trans;   
  Vector temp;
  
  la::CholeskyInit(noise_mat, &noise_mat_rt_trans);
  la::TransposeInit(noise_mat_rt_trans, &noise_mat_rt);
  la::MulInit(noise_mat_rt, v, &temp);
  v.CopyValues(temp);
};

