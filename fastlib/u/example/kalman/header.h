#ifndef HEADER_H
#define HEADER_H
#include "fastlib/fastlib.h"

struct ssm{
   Matrix A; // A,B,C define
   Matrix B; // the deterministic
   Matrix C; // part of the ssm 
   Matrix D; 
   Matrix Q; // process noise cov
   Matrix R; // mst. noise cov
   Matrix S; // cross-correlation 
};

Vector propagate_one_step(const Matrix& a_mat, const Matrix& b_mat, const Vector& x, const Vector &u, const Vector& w);

void SsmTimeInvariantSignalGenerator(const int& T, const ssm& LDS, const Matrix& u, Matrix* w, Matrix* v, Matrix* x, Matrix *y);

Matrix schur(const Matrix& a_mat, const Matrix& b_mat, const Matrix& c_mat, const Matrix& d_mat);

void print_matrix(const Matrix& a_mat, const char* name);

void matrix_concatenate_col_init(const Matrix& a_mat, const Matrix& b_mat, Matrix* x_mat);

void matrix_concatenate_row_init(const Matrix& a_mat, const Matrix& b_mat, Matrix* x_mat);

void extract_sub_matrix_init(const Matrix& a_mat, const int& r_in, const int& r_out, const int& c_in, const int& c_out, Matrix* x_mat);

void extract_sub_vector_of_vector_init(const Vector& v, const int& r_in, const int& r_out, Vector* x);

void set_portion_of_matrix(const Matrix& a_mat, const int& r_in,const int& r_out, const int& c_in, const int& c_out,Matrix* x_mat);

void set_portion_of_matrix(const Vector& a_mat, const int& r_in,const int& r_out, const int& c, Matrix* x_mat);

void RandVector(Vector &v, const Matrix& noise_mat);

#endif
