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
#include "fastlib/fastlib.h"
#include "kalman_helper.h"
#include "kalman.h"

// Measurement Update
void KalmanFiltTimeInvariantMstUpdate(const ssm& lds, const Vector& y_t, 
				      const Vector& x_pred_t, 
				      const Matrix& p_pred_t,
				      const Vector& y_pred_t, 
				      const Matrix& inno_cov_t, 
				      Vector& x_hat_t, Matrix& p_hat_t, 
				      Matrix& k_gain_t) {
  // create some params to be used later
  // tempm's are temp. matrices
  // temp's are temp vectors 
  int ny = lds.c_mat.n_rows();
  int nx = lds.a_mat.n_rows(); 
  Matrix c_trans; la::TransposeInit(lds.c_mat, &c_trans); 
  Matrix tempm_0, tempm_1, tempm_2, tempm_3; 
  Vector temp_1; 

  // create required square-root matrices
  // a sqrt-matrix is s.t. x = x_sqrt*x_sqrt_trans
  // want: r_sqrt, r_sqrt_trans
  // want: p_pred_t_sqrt, p_pred_t_sqrt_trans
  Matrix r_sqrt, r_sqrt_trans; 
  Matrix p_pred_t_sqrt, p_pred_t_sqrt_trans;
  la::CholeskyInit(lds.r_mat, &r_sqrt_trans); 
  la::TransposeInit(r_sqrt_trans, &r_sqrt); 
  la::CholeskyInit(p_pred_t, &p_pred_t_sqrt_trans); 
  la::TransposeInit(p_pred_t_sqrt_trans, &p_pred_t_sqrt);

  // form pre-array
  // pre_mat = [r_sqrt_trans                , 0; 
  //            p_red_t_sqrt_trans*c_trans' , P_pred_t_sqrt_trans]
  // pre_mat of size (nx + ny, nx + ny)
  Matrix pre_mat(ny + nx, ny + nx); pre_mat.SetZero();
  set_portion_of_matrix(r_sqrt_trans, 0, ny-1, 0, ny-1, &pre_mat);
  set_portion_of_matrix(p_pred_t_sqrt_trans, ny, ny + nx - 1, ny, ny + nx - 1, 
			&pre_mat);
  la::MulInit(p_pred_t_sqrt_trans, c_trans, &tempm_0);
  set_portion_of_matrix(tempm_0, ny, ny + nx-1, 0, ny - 1, &pre_mat); 

  // form post-array
  // post_mat = [inno_cov_t_sqrt, 0;
  //             tempm_1,        p_hat_t_sqrt];
  // post_mat of size (nx + ny, nx + ny)
  Matrix qq, rr; la::QRInit(pre_mat, &qq, &rr);  
  Matrix post_mat; la::TransposeInit(rr, &post_mat);
  
  // extract p_hat_t_sqrt, k_gain_t  from post-array
  // k_gain_t = tempm_1*inv(inno_cov_t_sqrt)
  extract_sub_matrix_init(post_mat, ny, ny + nx - 1, 0, ny-1, &tempm_1);
  extract_sub_matrix_init(post_mat, 0, ny-1, 0, ny-1, &tempm_2);
  la::InverseInit(tempm_2, &tempm_3); 
  la::MulOverwrite(tempm_1,tempm_3, &k_gain_t); 
  Matrix p_hat_t_sqrt;
  extract_sub_matrix_init(post_mat, ny, ny + nx - 1, ny, ny + nx - 1, &p_hat_t_sqrt);   
  la::MulTransBOverwrite(p_hat_t_sqrt, p_hat_t_sqrt, &p_hat_t); 
  
  // Get state-estimates
  // x_hat_t = x_pred_t + k_gain_t*(y_t - y_pred_t); 
  // temp_1 = y_t - y_pred_t = innovations;
  la::ScaleInit(-1, y_pred_t, &temp_1); 
  la::AddTo(y_t, &temp_1); 
  la::MulOverwrite(k_gain_t, temp_1, &x_hat_t); 
  la::AddTo(x_pred_t, &x_hat_t); 
}; 

// Time Update
void KalmanFiltTimeInvariantTimeUpdate(const ssm& lds, const Vector& x_hat_t, 
				       const Matrix& p_hat_t, 
				       const Vector& y_t, const Vector& u_t, 
				       Vector& x_pred_t_next, 
				       Vector& y_pred_t_next, 
				       Matrix& p_pred_t_next, 
				       Matrix& inno_cov_t_next) {
  // create some params to be used later
  // tempm's are temp. matrices
  // temp's are temp vectors 
  Matrix tempm_1, tempm_2; 
  Vector temp_1, temp_2; 
  int nx = lds.a_mat.n_cols(); 

  // Due to non-zero s_mat, must compute a_eff_mat, q_eff_mat, e_mat
  // a_eff_mat = a_mat - s_mat*inv(r_mat)*c_mat
  // q_eff_mat = q_mat - s_mat*inv(r_mat)*(s_mat)'
  // e_mat     = s_mat*inv(r_mat)
  Matrix a_eff_mat(lds.a_mat.n_rows(), lds.a_mat.n_cols()); 
  Matrix q_eff_mat(lds.q_mat.n_rows(), lds.q_mat.n_cols()); 
  Matrix e_mat; 
  
  // a_eff_mat
  schur(lds.a_mat, lds.s_mat,  lds.r_mat, lds.c_mat, &a_eff_mat);
  
  // e_mat
  Matrix inv_r_mat; la::InverseInit(lds.r_mat, &inv_r_mat); 
  la::MulInit(lds.s_mat, inv_r_mat, &e_mat); 

  // q_eff_mat
  Matrix s_trans; 
  la::TransposeInit(lds.s_mat, &s_trans);
  schur(lds.q_mat, lds.s_mat, lds.r_mat, s_trans, &q_eff_mat);
    
  // Also create a_eff_trans to be used later
  Matrix a_eff_trans;  
  la::TransposeInit(a_eff_mat, &a_eff_trans);

  // form pre-array
  // pre_mat = [p_hat_t_sqrt_trans*a_eff_trans; q_eff_sqrt_trans]
  Matrix pre_mat(nx + nx, nx);
  Matrix q_eff_sqrt_trans;
  Matrix  p_hat_t_sqrt_trans;
  la::CholeskyInit(q_eff_mat, & q_eff_sqrt_trans);  
  la::CholeskyInit(p_hat_t, &p_hat_t_sqrt_trans);  
  la::MulInit(p_hat_t_sqrt_trans, a_eff_trans, &tempm_1);
  set_portion_of_matrix(tempm_1, 0, nx-1, 0, nx-1, &pre_mat);
  set_portion_of_matrix(q_eff_sqrt_trans, nx, 2*nx-1, 0, nx-1, &pre_mat);

  // Form post-array via QR decomp.
  Matrix qq, rr;
  la::QRInit(pre_mat, &qq, &rr); 
  Matrix post_mat;
  la::TransposeInit(rr, &post_mat);

  // Extract p_pred_t_next_sqrt from post-array
  Matrix p_pred_t_next_sqrt;
  extract_sub_matrix_init(post_mat, 0, nx-1, 0, nx-1, &p_pred_t_next_sqrt);
  la::MulTransBOverwrite(p_pred_t_next_sqrt, p_pred_t_next_sqrt, 
			 &p_pred_t_next);
  // Generate inno_cov_t_next  = r_mat + c_mat*p_pred_t_next*c_mat';
  la::MulInit(lds.c_mat, p_pred_t_next_sqrt, &tempm_2); 
  la::MulTransBOverwrite(tempm_2, tempm_2, &inno_cov_t_next); 
  la::AddTo(lds.r_mat, &inno_cov_t_next); 

  // Generate x_pred_t_next = a_eff_mat*x_hat_t + e_mat*y_t + b_mat*u_t; 
  // y_pred_t_next = C*x_pred_t_next;
  la::MulInit(a_eff_mat, x_hat_t, &temp_1); 
  la::MulInit(e_mat, y_t, &temp_2); 
  la::AddTo(temp_1, &temp_2);
  la::MulOverwrite(lds.b_mat, u_t, &x_pred_t_next); 
  la::AddTo(temp_2, &x_pred_t_next); 
  la::MulOverwrite(lds.c_mat, x_pred_t_next, &y_pred_t_next);
};
