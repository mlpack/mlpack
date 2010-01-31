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
 * @file kalman.h
 * This file contains two functions that computer
 * the Time update and Measurement update steps
 * associated with the time-invariant kalman filter
 *
 * @see kalman.cc
 */

#ifndef KALMAN_H
#define KALMAN_H
#include "fastlib/fastlib.h"
#include "kalman_helper.h"

/**
 * Time Update for the Kalman Filter
 * 
 * x_{t|t} = x_{t|t-1} + k_gain_{t}*(y_t - y_{t|t-1})
 * k_gain_{t} = p_{t|t-1}*c_mat'*inv(inno_cov_{t|t-1})
 * p_{t|t} = (I - k_gain_t*c_mat)*p_{t|t-1}
 *
 * In reality, the update eq. for p_{t|t} and k_gain_{t}
 * are computed using a sq-rt algo. to ensure pos. definiteness.
 * see: "Linear Estimation" by Kailath, Sayed, Hassibi, 2000, Prentice-Hall
 * Chp 12 for more details
 *
*/
void KalmanFiltTimeInvariantMstUpdate(const ssm& lds, const Vector& y_t, 
				      const Vector& x_pred_t, 
				      const Matrix& p_pred_t, 
				      const Vector& y_pred_t, 
				      const Matrix& inno_cov_t, 
				      Vector& x_hat_t, Matrix& p_hat_t, 
				      Matrix& k_gain_t);

/**
 * Measurement Update for the Kalman Filter
 *
 * x_{t+1|t} = a_eff_mat*x_{t|t} + b_mat*u_t + e_mat*y_t
 * p_{t+1|t} = a_eff_mat*p_{t|t}*a_eff_mat' + q_eff_mat
 * y_{t+1|t} = c_mat*x_{t+1|t} 
 * inno_cov_{t+1|t} = c_mat*p_{t+1|t}*c_mat' + r_mat
 * where a_eff_mat = a_mat - s_mat*inv(r_mat)*c_mat
 * e_mat     = s_mat*inv(r_mat) due to non-zero s_mat
 * q_eff_mat = q_mat - s_mat*inv(r_mat)*s_mat'
 *
 * In reality, the update eqn. for p_{t+1|t} is implemented
 * using a square-root algorithm to ensure pos. definiteness
 * see: "Linear Estimation" by Kailath, Sayed, Hassibi, 2000, Prentice-Hall
 * Chp 12 for more details
 */
void KalmanFiltTimeInvariantTimeUpdate(const ssm& lds, const Vector& x_hat_t, 
				       const Matrix& p_hat_t, 
				       const Vector& y_t, const Vector& u_t, 
				       Vector& x_pred_t_next, 
				       Vector& y_pred_t_next, 
				       Matrix& p_pred_t_next, 
				       Matrix& inno_cov_t_next);

#endif

