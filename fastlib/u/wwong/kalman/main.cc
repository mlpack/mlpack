#include <iostream>
#include "fastlib/fastlib.h"
#include "header.h"
#include "kalman.h"

using namespace std;

int main(int argc, char* argv[])  
{

  fx_init(argc, argv);

  /* Signal generation*/

/*
  // Option 1: generate signals from a time-invariant ssm 

  int t_duration = 1000;
   ssm LDS; //Initialize a SSM
  Matrix& A=LDS.A;
   A.Init(2,2); A.set(0,0,0.9); A.set(0,1,0.0);A.set(1,0,0.0); A.set(1,1,0.9);
  Matrix& B=LDS.B;
   B.Init(2,1); B.set(0,0,1.0); B.set(1,0,1.0);
  Matrix& C=LDS.C;
   C.Init(1,2); C.set(0,0,1.0); C.set(0,1,1.0);
  Matrix& D=LDS.D;
   D.Init(1,1); D.set(0,0, 1.0);
  Matrix& Q=LDS.Q;
   Q.Init(2,2); Q.set(0,0,8.8125); Q.set(0,1,69.125);Q.set(1,0,69.125); Q.set(1,1,683.25);
  Matrix& R=LDS.R;
   R.Init(1,1); R.set(0,0,691.06);
  Matrix& S=LDS.S;
   S.Init(2,1); S.set (0,0,30); S.set(1,0,54.75);
 
  int nx = A.n_rows(); int ny= C.n_rows();  int nu = B.n_cols();
  
  Matrix x(nx,t_duration+1+1); x.SetAll(0); 
  Matrix y(ny,t_duration+1+1); y.SetAll(0);
  Matrix u(nu,t_duration+1+1); u.SetAll(1);
  Matrix w(nx,t_duration+1+1); w.SetAll(0);
  Matrix v(ny,t_duration+1+1); v.SetAll(0);
  SsmTimeInvariantSignalGenerator(t_duration, LDS, u, &w, &v, &x, &y);

  // Kalman filtering
  Matrix x_pred(nx,t_duration+1+1),x_hat(nx,t_duration+1+1), p_pred[t_duration+1+1], p_hat[t_duration+1+1], k_gain[t_duration+1+1], y_pred(ny,t_duration+1+1), inno_cov[t_duration+1+1];
  
  for (int t =0; t<=t_duration+1;t++ ){
  (p_pred[t]).Init(nx,nx);
  (p_hat[t]).Init(nx,nx);
  (k_gain[t]).Init(nx,ny);
   (inno_cov[t]).Init(ny,ny);
   };

  // Initialize x_pred_(0|-1), p_{0|-1}, y_pred_{0|-1}, inno_cov_{0|-1}
  x_pred.set(0,0,1); x_pred.set(1,0,1);
  p_pred[0].set(0,0,0.001); p_pred[0].set(0,1,0); p_pred[0].set(1,0,0); p_pred[0].set(1,1,0.001);           

  Matrix tempM, Ctrans;
  la::TransposeInit(LDS.C, &Ctrans);

  la::MulInit(LDS.C,p_pred[0], &tempM); // tempM = C*P_pred_0;
  la::MulOverwrite(tempM, Ctrans, &inno_cov[0]); la::AddTo(LDS.R, &inno_cov[0]); // inno_cov_0  = R + C*P_pred_0*C';

  Vector x_pred_0; x_pred.MakeColumnVector(0, &x_pred_0);
  Vector y_pred_0; y_pred.MakeColumnVector(0, &y_pred_0);
  Vector u_0;  u.MakeColumnVector(0,&u_0);
  Vector temp;
  la::MulOverwrite(LDS.C, x_pred_0, &y_pred_0); la::MulInit(LDS.D, u_0, &temp); 
  la::AddTo(temp, &y_pred_0); // y_pred_0 = C*x_pred_0 + D*u_0;

  // Start Kalman Filtering
  KalmanFiltTimeInvariant(t_duration, LDS, u, y, &x_pred, p_pred, &x_hat, p_hat, &y_pred, inno_cov, k_gain);
*/

  // Option 2: Load Data from file (data generate from matlab code)
  const char* A_in = fx_param_str(NULL, "A_in", "A_in");
  const char* B_in = fx_param_str(NULL, "B_in", "B_in");
  const char* C_in = fx_param_str(NULL, "C_in", "C_in");
  const char* D_in = fx_param_str(NULL, "D_in", "D_in");
  const char* Q_in = fx_param_str(NULL, "Q_in", "Q_in");
  const char* R_in = fx_param_str(NULL, "R_in", "R_in");
  const char* S_in = fx_param_str(NULL, "S_in", "S_in");
  const char* u_in = fx_param_str(NULL, "u_in", "u_in");
  const char* w_in = fx_param_str(NULL, "w_in", "w_in");
  const char* v_in = fx_param_str(NULL, "v_in", "v_in");
  const char* y_in = fx_param_str(NULL, "y_in", "y_in");
  const char* x_in = fx_param_str(NULL, "x_in", "x_in");
  const char* t_in = fx_param_str(NULL, "t_in", "T_in");
  
  ssm LDS; Matrix u, w, v, y, x, t_duration_; 

  data::Load(A_in,&LDS.A); data::Load(B_in,&LDS.B); data::Load(C_in,&LDS.C); data::Load(D_in,&LDS.D); data::Load(Q_in,&LDS.Q); data::Load(R_in,&LDS.R); data::Load(S_in,&LDS.S);  data::Load(u_in,&u);  data::Load(w_in,&w);  data::Load(v_in,&v);  data::Load(y_in,&y);  data::Load(x_in,&x); 
  data::Load(t_in,&t_duration_);

  int t_duration = (int)t_duration_.get(0,0);
  int nx = LDS.A.n_rows(); int ny= LDS.C.n_rows(); 

  //  Kalman Filtering //
  Matrix x_pred(nx,t_duration+1+1),x_hat(nx,t_duration+1+1), p_pred[t_duration+1+1], p_hat[t_duration+1+1], k_gain[t_duration+1+1], y_pred(ny,t_duration+1+1), inno_cov[t_duration+1+1];
  for (int t =0; t<=t_duration+1;t++ )  {
  (p_pred[t]).Init(nx,nx);
  (p_hat[t]).Init(nx,nx);
  (k_gain[t]).Init(nx,ny);
   (inno_cov[t]).Init(ny,ny);
   };

  // Initialize x_pred_(0|-1), P_{0|-1}, y_pred_{0|-1}, inno_cov_{0|-1}
  const char* x_pred_0_in   = fx_param_str(NULL, "x_pred_0_in",  "x_pred_0_in");
  const char* p_pred_0_in   = fx_param_str(NULL, "p_pred_0_in",  "P_pred_0_in");
  const char* y_pred_0_in   = fx_param_str(NULL, "y_pred_0_in",  "y_pred_0_in");
  const char* inno_cov_0_in = fx_param_str(NULL, "inno_cov_0_in","inno_cov_0_in");
 
  Matrix x_pred_0_, y_pred_0_, p_pred_0_, inno_cov_0_;
    
  data::Load(x_pred_0_in,&x_pred_0_);
  data::Load(y_pred_0_in,&y_pred_0_);
  data::Load(p_pred_0_in,&p_pred_0_);
  data::Load(inno_cov_0_in,&inno_cov_0_);
  
  set_portion_of_matrix(x_pred_0_,0,nx-1,0,0,&x_pred);
  set_portion_of_matrix(y_pred_0_,0,ny-1,0,0,&y_pred);
  p_pred[0]   = p_pred_0_;
  inno_cov[0] = inno_cov_0_;

  // Start Kalman Filtering
  KalmanFiltTimeInvariant(t_duration, LDS, u, y, &x_pred, p_pred, &x_hat, p_hat, &y_pred, inno_cov, k_gain);
  
  /*Store Results*/     
  Matrix x_, y_, x_hat_, x_pred_, y_pred_;
  la::TransposeInit(x, &x_);
  la::TransposeInit(y, &y_);
  la::TransposeInit(x_hat, &x_hat_);
  la::TransposeInit(x_pred, &x_pred_);
  la::TransposeInit(y_pred, &y_pred_);

  cout<<"hi"<<endl;
  data::Save("x_hat", x_hat_);
  data::Save("x_pred",x_pred_);
  data::Save("y_pred",y_pred_);

  //  might want to store p_hat[t], p_pred[t], inno_cov[t], k_gain[t] but just save time-invariant results p_hat[T+1], p_pred[T+1], k_gain[T+1]

  Matrix k_gain_end;
  la::TransposeInit(k_gain[t_duration+1],&k_gain_end);
  
  data::Save("K_end", k_gain_end);
  data::Save("P_pred_end",p_pred[t_duration+1]);
  data::Save("P_hat_end", p_hat[t_duration+1]);
  data::Save("inno_cov_end", inno_cov[t_duration+1]);
 
  //  fx_done();
};


/* Old Stuff To Delete Later
void kalman_filt_time_invariant(const int& T, const ssm& LDS, const Matrix& u, const Matrix& y, Matrix* x_pred, Matrix P_pred[], Matrix* x_hat, Matrix P_hat[], Matrix* y_pred, Matrix inno_cov[], Matrix K[] )
{

  cout<<" Due to potentially non-zero S, need to compute A_eff, E, Q_eff"<<endl;
// Due to potentially non-zero S, need to compute A_eff, E, Q_eff
Matrix A_eff(LDS.A.n_rows(),LDS.A.n_cols()), Q_eff(LDS.Q.n_rows(),LDS.Q.n_cols()), E;
A_eff = schur(LDS.A,LDS.S,LDS.R,LDS.C ); // A_eff = A - S*inv(R)*C
Matrix invR; la::InverseInit(LDS.R, &invR); 
la::MulInit(LDS.S,invR,&E); // E = S*inv(R);
Matrix Strans; la::TransposeInit(LDS.S,&Strans);
Q_eff = schur(LDS.Q,LDS.S,invR,Strans); //Q_eff = R - S*inv(R)*S'

//Code starts
Vector y_t, u_t, x_pred_t ,x_pred_t_next, x_hat_t, y_pred_t; // aliases, no need to .Init(length);
Matrix K_t, P_pred_t, P_pred_t_next, P_hat_t, inno_cov_t;  //aliases

// Declare required temp. variables
Matrix Ctrans; la::TransposeInit(LDS.C,&Ctrans);
Matrix A_efftrans;  la::TransposeInit(A_eff,&A_efftrans);
Vector temp_1, temp_2, temp_3, temp_4;
Matrix tempM_1, tempM_2, tempM_3, tempM_4, tempM_5, tempM_6;

  for (int t =0; t<=T ;t++)
  {
  //Assign References recalling that arrays are by def, passed by reference
    y.MakeColumnVector(t, &y_t);
    u.MakeColumnVector(t, &u_t);
    (*x_hat).MakeColumnVector(t,&x_hat_t);
    P_hat_t.Alias(P_hat[t]);
    (*x_pred).MakeColumnVector(t,&x_pred_t);
    P_pred_t.Alias(P_pred[t]);
    (*x_pred).MakeColumnVector(t+1,&x_pred_t_next);
    P_pred_t_next.Alias(P_pred[t+1]);
    (*y_pred).MakeColumnVector(t,&y_pred_t);
    inno_cov_t.Alias(inno_cov[t]);
    K_t.Alias(K[t]);

  // Measurement Update: 
 
   la::MulInit(LDS.C,P_pred_t, &tempM_1); // tempM_1 = C*P_pred_t;
   la::MulInit(tempM_1, Ctrans, &tempM_2); la::AddTo(LDS.R,&tempM_2); la::Inverse(&tempM_2); // tempM_2 = inv(R + C*P_pred_t*C');
   la::MulInit(P_pred_t,Ctrans,&tempM_3); // tempM_3 = P_pred_t*Ctrans;
   la::MulOverwrite(tempM_3, tempM_2, &K_t);   // K_t = P_pred_t*C'*inv(R + C*P_pred_t*C');
   la::MulInit(K_t,LDS.C,&tempM_4); // tempM_4 = K_t*C;
   la::MulInit(tempM_4, P_pred_t, &tempM_5);  la::Scale(-1,&tempM_5); // tempM_5 = -K_t*C*P_pred_t;
   la::AddOverwrite(P_pred_t, tempM_5 , &P_hat_t); // P_hat_t = P_pred_t - K_t*C*P_pred_t; 

   la::MulInit(LDS.D,u_t,&temp_1);   // temp_1 = D*u_t;   
   la::MulOverwrite(LDS.C,x_pred_t,&y_pred_t); la::AddTo(temp_1, &y_pred_t); // y_pred_t = C*x_pred_t + D*u_t;
   la::ScaleInit(-1,y_pred_t,&temp_2); // temp_2  = -(C*x_pred_t + D*u_t);
   la::AddTo(y_t,&temp_2); // temp_2 = y_t - C*xpred_t - D*u_t = innovations;
   la::MulOverwrite(K_t,temp_2, &x_hat_t);  la::AddTo(x_pred_t,&x_hat_t); // x_hat_t = x_pred_t + K_t*(y_t - C*x_pred_t - D*u_t); 
      
  // Time Update: 

    la::MulInit(A_eff,P_hat_t,&tempM_6); // tempM_6 = A_eff*P_hat_t;
    la::MulOverwrite(tempM_6, A_efftrans, &P_pred_t_next);  la::AddTo(Q_eff,&P_pred_t_next); // P_pred_t_next = A_eff*P_hat_t*A_eff' + Qeff

    la::MulInit(A_eff,x_hat_t,&temp_3);
    la::MulInit(E,y_t,&temp_4); la::AddTo(temp_3, &temp_4); //temp_4 = A_eff*x_hat_t + E*y_t;     
    la::MulOverwrite(LDS.B,u_t,&x_pred_t_next); la::AddTo(temp_4,&x_pred_t_next); // x_pred_t_next = A_eff*x_hat_t + E*y_t + B*u_t; 

  if (t == T) // last measurement update
  {
   y_t.Destruct(); u_t.Destruct(); 
   x_hat_t.Destruct(); P_hat_t.Destruct(); 
   x_pred_t.Destruct(); P_pred_t.Destruct(); 
   y_pred_t.Destruct(); inno_cov_t.Destruct(); K_t.Destruct();
   temp_1.Destruct(); temp_2.Destruct(); tempM_1.Destruct(); tempM_2.Destruct(); tempM_3.Destruct(); tempM_4.Destruct(); tempM_5.Destruct();  

   y.MakeColumnVector(t+1, &y_t);
   u.MakeColumnVector(t+1, &u_t);
   (*x_hat).MakeColumnVector(t+1,&x_hat_t);
   P_hat_t.Alias(P_hat[t+1]);
   (*x_pred).MakeColumnVector(t+1,&x_pred_t);
   P_pred_t.Alias(P_pred[t+1]);
   (*y_pred).MakeColumnVector(t+1,&y_pred_t);
   inno_cov_t.Alias(inno_cov[t+1]);   
   K_t.Alias(K[t+1]);

   la::MulInit(LDS.C,P_pred_t, &tempM_1); // tempM_1 = C*P_pred_{T+1};
   la::MulInit(tempM_1, Ctrans, &tempM_2); la::AddTo(LDS.R,&tempM_2); la::Inverse(&tempM_2); // tempM_2 = inv(R + C*P_pred_{T+1}*C');
   la::MulInit(P_pred_t,Ctrans,&tempM_3); // tempM_3 = P_pred_{T+1}*Ctrans;
   la::MulOverwrite(tempM_3, tempM_2, &K_t);   // K_{T+1} = P_pred_{T+1}*C'*inv(R + C*P_pred_{T+1}*C');
   la::MulInit(K_t,LDS.C,&tempM_4); // tempM_4 = K_{T+1}*C;
   la::MulInit(tempM_4, P_pred_t, &tempM_5);    la::Scale(-1,&tempM_5); // tempM_5 = -K_{T+1}*C*P_pred{T+1};
   la::AddOverwrite(P_pred_t, tempM_5 , &P_hat_t); // P_hat_{T+1} = P_pred_{T+1} - K_{T+1}*C*P_pred_{T+1}; 

   la::MulInit(LDS.D,u_t,&temp_1);   // temp_1 = D*u_{T+1};   
   la::MulOverwrite(LDS.C,x_pred_t,&y_pred_t); la::AddTo(temp_1, &y_pred_t); // y_pred_{T+1} = C*x_pred_{T+1} + D*u_t;
   la::ScaleInit(-1,y_pred_t,&temp_2); // temp_2  = -(C*x_pred_{T+1} + D*u_{T+1});
   la::AddTo(y_t,&temp_2); // temp_2 = y_{T+1} - C*xpred_{T+1} - D*u_{T+1} = innovations;
   la::MulOverwrite(K_t,temp_2, &x_hat_t);  la::AddTo(x_pred_t,&x_hat_t); // x_hat_{T+1} = x_pred_{T+1} + K_{T+1}*(y_{T+1} - C*x_pred_{T+1} - D*u_{T+1});        
  }
  else
  {
  // Destruct so that re-init can take place
  y_t.Destruct(); u_t.Destruct();
  x_hat_t.Destruct(); P_hat_t.Destruct();
  x_pred_t.Destruct(); P_pred_t.Destruct(); x_pred_t_next.Destruct(); P_pred_t_next.Destruct(); 
  y_pred_t.Destruct(); inno_cov_t.Destruct();
  K_t.Destruct();
  temp_1.Destruct(); temp_2.Destruct(); temp_3.Destruct(); temp_4.Destruct();
  tempM_1.Destruct(); tempM_2.Destruct(); tempM_3.Destruct(); tempM_4.Destruct(); tempM_5.Destruct();  tempM_6.Destruct();   
  };
  }; //end of for loop
};
*/

