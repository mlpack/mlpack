#include <iostream>
#include "fastlib/fastlib.h"
#include "header.h"
#include "kalman.h"

using namespace std;

void KalmanFiltTimeInvariantMstUpdate(const ssm& LDS, const Vector& y_t, const Vector& u_t, const Vector& x_pred_t, const Matrix& P_pred_t, const Vector& y_pred_t, const Matrix& inno_cov_t, Vector& x_hat_t, Matrix& P_hat_t, Matrix& K_t)
{
  // K_t = P_pred_t*C'*inv(inno_cov_t);
  // P_hat_t = P_pred_t - K_t*C_t*P_pred_t;
  // x_hat_t = x_pred_t + K_t*(y_t - y_pred_t)

  int ny = LDS.C.n_rows();
  int nx = LDS.A.n_rows(); 
  Matrix Ctrans; la::TransposeInit(LDS.C, &Ctrans); 

  // Get covariance matrices > 0;
  Matrix tempM_0, tempM_1, tempM_2, tempM_3; // temporary Matrices

  // create required square-root matrices
  Matrix R_sqrt, R_sqrt_trans, P_pred_t_sqrt, P_pred_t_sqrt_trans;
  la::CholeskyInit(LDS.R,&R_sqrt_trans); la::TransposeInit(R_sqrt_trans,&R_sqrt); 
  la::CholeskyInit(P_pred_t,&P_pred_t_sqrt_trans); la::TransposeInit(P_pred_t_sqrt_trans,&P_pred_t_sqrt);

  // form PRE-ARRAY
  Matrix pre_mat(ny+nx,ny+nx); pre_mat.SetZero();

  set_portion_of_matrix(R_sqrt_trans,0,ny-1,0,ny-1,&pre_mat);
  set_portion_of_matrix(P_pred_t_sqrt_trans,ny,ny+nx-1,ny,ny+nx-1,&pre_mat);
  la::MulInit(P_pred_t_sqrt_trans,Ctrans, &tempM_0);
  set_portion_of_matrix(tempM_0,ny,ny+nx-1,0,ny-1,&pre_mat); // pre_mat = [R_sqrt_trans, 0; P_pred_t_sqrt_trans*C';P_pred_t_sqrt_trans]

  Matrix qq, rr; la::QRInit(pre_mat,&qq,&rr);
  
  Matrix post_mat; la::TransposeInit(rr, &post_mat);// post_mat = [inno_cov_t_sqrt,0;tempM_1, P_hat_t_sqrt];
   
  extract_sub_matrix_init(post_mat,ny,ny+nx-1,0,ny-1,&tempM_1);
  extract_sub_matrix_init(post_mat,0,ny-1,0,ny-1,&tempM_2);
  la::InverseInit(tempM_2,&tempM_3); //temp_3 = inv(tempM_2);
  la::MulOverwrite(tempM_1,tempM_3,&K_t); // K_t = tempM_1*tempM_3;
   
  Matrix P_hat_t_sqrt;
  extract_sub_matrix_init(post_mat, ny, ny+nx-1,ny, ny+nx-1, &P_hat_t_sqrt);   
  la::MulTransBOverwrite(P_hat_t_sqrt, P_hat_t_sqrt, &P_hat_t); //P_hat_t = P_hat_t_sqrt*P_hat_t_sqrt_trans;
  
  // Get state-estimates
  Vector temp_1; // temporary Vectors;    
  la::ScaleInit(-1,y_pred_t,&temp_1); // temp_1  = -y_pred_t;
  la::AddTo(y_t,&temp_1); // temp_1 = y_t - y_pred_t = innovations;
  la::MulOverwrite(K_t,temp_1, &x_hat_t); la::AddTo(x_pred_t,&x_hat_t); // x_hat_t = x_pred_t + K_t*(y_t - C*x_pred_t - D*u_t); y       
}; 
 
void KalmanFiltTimeInvariantTimeUpdate(const ssm& LDS, const Vector& x_hat_t, const Matrix& P_hat_t, const Vector& y_t, const Vector& u_t, const Vector& u_t_next, Vector& x_pred_t_next, Vector& y_pred_t_next, Matrix& P_pred_t_next, Matrix& inno_cov_t_next)
{
 
  // x_pred_t_next = A*x_hat_t + B*u_t + E*y_t;
  // y_pred_t_next = C*x_pred_t_next + D*u_t_next;
  // P_pred_t_next = A*P_hat_t*A' + Q;
  // inno_cov_t_next = C*P_pred_t_next*C' + R;

  // Due to potentially non-zero S, need to compute A_eff, E, Q_eff
  Matrix A_eff(LDS.A.n_rows(),LDS.A.n_cols()), Q_eff(LDS.Q.n_rows(),LDS.Q.n_cols()), E;
  A_eff = schur(LDS.A,LDS.S,LDS.R,LDS.C ); // A_eff = A - S*inv(R)*C
  Matrix invR; la::InverseInit(LDS.R, &invR); 
  la::MulInit(LDS.S,invR,&E); // E = S*inv(R);
  Matrix Strans; la::TransposeInit(LDS.S,&Strans);
  Q_eff = schur(LDS.Q,LDS.S,LDS.R,Strans); //Q_eff = Q - S*inv(R)*S'

  Matrix A_efftrans;  la::TransposeInit(A_eff,&A_efftrans);
  Matrix Ctrans; la::TransposeInit(LDS.C, &Ctrans); 

  Matrix tempM_5, tempM_6; // temporary Matrices
  Vector temp_2, temp_3, temp_4; // temporary vectors

  int nx = LDS.A.n_cols(); 

  // form PRE-Array
  Matrix pre_mat(nx+nx,nx);
  Matrix Q_eff_sqrt_trans, P_hat_t_sqrt_trans;
  la::CholeskyInit(Q_eff, &Q_eff_sqrt_trans);  
  la::CholeskyInit(P_hat_t, &P_hat_t_sqrt_trans);  

  la::MulInit(P_hat_t_sqrt_trans,A_efftrans, &tempM_5);
  set_portion_of_matrix(tempM_5, 0, nx-1, 0, nx-1, &pre_mat);
  set_portion_of_matrix(Q_eff_sqrt_trans, nx, 2*nx-1,0,nx-1,&pre_mat);

  // Q-R Decomposition
  Matrix qq, rr;
  la::QRInit(pre_mat, &qq, &rr);
  
  // Form Post-Array
  Matrix post_mat;
  la::TransposeInit(rr, &post_mat);

  Matrix P_pred_t_next_sqrt;
  extract_sub_matrix_init(post_mat,0,nx-1,0,nx-1,&P_pred_t_next_sqrt);
  la::MulTransBOverwrite(P_pred_t_next_sqrt, P_pred_t_next_sqrt, &P_pred_t_next);

  la::MulInit(LDS.C,P_pred_t_next_sqrt, &tempM_6); la::MulTransBOverwrite(tempM_6, tempM_6, &inno_cov_t_next); 
  la::AddTo(LDS.R, &inno_cov_t_next); // inno_cov_t_next  = R + C*P_pred_t_next*C';

  la::MulInit(A_eff,x_hat_t,&temp_2); la::MulInit(E,y_t,&temp_3); la::AddTo(temp_2, &temp_3); // temp_3 = A_eff*x_hat_t + E*y_t;     
  la::MulOverwrite(LDS.B,u_t,&x_pred_t_next); la::AddTo(temp_3,&x_pred_t_next); // x_pred_t_next = A_eff*x_hat_t + E*y_t + B*u_t; 
  la::MulOverwrite(LDS.C,x_pred_t_next, &y_pred_t_next); la::MulInit(LDS.D,u_t_next,&temp_4); 
  la::AddTo(temp_4,&y_pred_t_next); // y_pred_t_next = C*x_pred_t_next + D*u_t_next;
};

void KalmanFiltTimeInvariant(const int& T, const ssm& LDS, const Matrix& u, const Matrix& y, Matrix* x_pred, Matrix P_pred[], Matrix* x_hat, Matrix P_hat[], Matrix* y_pred, Matrix inno_cov[], Matrix K[] )
{
  Vector y_t, u_t, u_t_next, x_pred_t ,x_pred_t_next, x_hat_t, y_pred_t, y_pred_t_next; // aliases, no need to .Init(length);

  for (int t =0; t<=T ;t++) {
  //Assign References recalling that arrays are by def, passed by reference
  y.MakeColumnVector(t, &y_t);
  u.MakeColumnVector(t, &u_t);
  u.MakeColumnVector(t+1, &u_t_next);
  (*x_hat).MakeColumnVector(t, &x_hat_t);
  (*x_pred).MakeColumnVector(t, &x_pred_t);
  (*x_pred).MakeColumnVector(t+1, &x_pred_t_next);
  (*y_pred).MakeColumnVector(t, &y_pred_t);
  (*y_pred).MakeColumnVector(t+1, &y_pred_t_next);
     
  // Measurement Update
  KalmanFiltTimeInvariantMstUpdate(LDS, y_t, u_t, x_pred_t, P_pred[t], y_pred_t, inno_cov[t], x_hat_t, P_hat[t], K[t]);

  // Time Update
  KalmanFiltTimeInvariantTimeUpdate(LDS, x_hat_t, P_hat[t], y_t, u_t, u_t_next, x_pred_t_next, y_pred_t_next, P_pred[t+1], inno_cov[t+1]);
 
      
  if (t == T){ // last measurement update for t = T+1
    Vector y_T_next, u_T_next, x_hat_T_next;

    y.MakeColumnVector(T+1,&y_T_next);
    y.MakeColumnVector(T+1,&u_T_next);
    (*x_hat).MakeColumnVector(T+1,&x_hat_T_next);

    KalmanFiltTimeInvariantMstUpdate(LDS, y_T_next, u_T_next, x_pred_t_next, P_pred[t+1], y_pred_t_next, inno_cov[t+1], x_hat_T_next, P_hat[t+1], K[t+1]);     
  } else{
    // Destruct so that re-init can take place
    y_t.Destruct(); u_t.Destruct(); u_t_next.Destruct();
    x_hat_t.Destruct(); 
    x_pred_t.Destruct();  x_pred_t_next.Destruct();
    y_pred_t.Destruct();  y_pred_t_next.Destruct(); 
   }; 
  }; 
};
