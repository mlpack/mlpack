#ifndef KALMAN_H
#define KALMAN_H
#include "fastlib/fastlib.h"
#include "header.h"

void KalmanFiltTimeInvariantMstUpdate(const ssm& LDS, const Vector& y_t, const Vector& u_t, const Vector& x_pred_t, const Matrix& P_pred_t, const Vector& y_pred_t, const Matrix& inno_cov_t, Vector& x_hat_t, Matrix& P_hat_t, Matrix& K_t);

void KalmanFiltTimeInvariantTimeUpdate(const ssm& LDS, const Vector& x_hat_t, const Matrix& P_hat_t, const Vector& y_t, const Vector& u_t, const Vector& u_t_next, Vector& x_pred_t_next, Vector& y_pred_t_next, Matrix& P_pred_t_next, Matrix& inno_cov_t_next);

void KalmanFiltTimeInvariant(const int& T, const ssm& LDS, const Matrix& u, const Matrix& y, Matrix* x_pred, Matrix P_pred[], Matrix* x_hat, Matrix P_hat[], Matrix* y_pred, Matrix inno_cov[], Matrix K[] );

#endif

