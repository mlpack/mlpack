#include "../hshmm/lds.h"


void Inference(Matrix Y) {

  // assume that A_, C_, Q_, R_, mu_0_, Sigma_0_ are sensibly initialized


  ArrayList<Matrix> V_t_given_t;
  V_t_given_t.Init(T+1);
  for(int t = 0; t <= T; t++) {
    V_t_given_t[t].Init(n_dims_latent_, n_dims_latent_);
  }

  ArrayList<Matrix> V_t_given_t_minus_1;
  V_t_given_t_minus_1.Init(T+1);
  for(int t = 1; t <= T; t++) {
    V_t_given_t_minus_1[t].Init(n_dims_latent_, n_dims_latent_);
  }

  // variables:
  Matrix X_t_given_t; // R^{(T+1) x n}
  X_t_given_t.Init(n_dims_latent_, T + 1);
  Vector x_t_minus_1_given_t_minus_1; // R^n
  //x_t_minus_1_given_t_minus_1.Init(n_dims_latent_); // this will be an alias
  Vector x_t_given_t_minus_1; // R^n
  x_t_given_t_minus_1.Init(n_dims_latent_);
  Vector x_t_given_t; // R^n
  //x_t_given_t.Init(n_dims_latent_); // this will ben an alias

  /*
  Matrix V_t_minus_1_given_t_minus_1; // R^{n x n}
  //V_t_minus_1_given_t_minus_1.Init(n_dims_latent_, n_dims_latent_); // Init'd below via a Copy
  Matrix V_t_given_t_minus_1; // R^{n x n}
  V_t_given_t_minus_1.Init(n_dims_latent_, n_dims_latent_);
  Matrix V_t_given_t; // R^{n x n}
  V_t_given_t.Init(n_dims_latent_, n_dims_latent_);
  */

  Matrix K_t; // R^{n x d}
  K.Init(n_dims_latent_, n_dims_obs_);

  /*
    V_t_given_t_minus_1 is the predicted estimate covariance
    In general, this will be A V A^T + Q
    V_0_given_0 is by definition Sigma_0
    V_1_given_0 is then A Sigma_0 A^T + Q
    x_0_given_0 is by definition mu_0
    x_1_given_0 is clearly A mu_0 (= A x_0_given_0)
  */

  // Kalman filter (forward algorithm)

  X_t_given_t.CopyVectorToColumn(0, mu_0);
  //x_t_minus_1_given_t_minus_1.Copy(mu_0_);
  V_t_minus_1_given_t_minus_1.Copy(Sigma_0_);

  Matrix temp_o_by_l;
  temp_o_by_l.Init(n_dims_obs_, n_dims_latent_);
  Matrix temp_l_by_l;
  temp_l_by_l.Init(n_dims_latent, n_dims_latent);
  Matrix temp1_o_by_o;
  temp1_o_by_o.Init(n_dims_obs, n_dims_obs);
  Matrix temp2_o_by_o;
  temp2_o_by_o.Init(n_dims_obs, n_dims_obs);

  Vector ideal;
  ideal.Init(n_dims_obs);
  Vector innovation;
  innovation.Init(n_dims_obs);


  for(int t = 1; t <= T; t++) {
    X_t_given_t.MakeColumnVector(t - 1, &x_t_minus_1_given_t_minus_1);
    la::MulOverwrite(A_, x_t_minus_1_given_t_minus_1, &x_t_given_t_minus_1);
    
    la::MulOverwrite(A_, V_t_given_t[t-1], &temp_l_by_l);
    la::MulTransBOverwrite(temp_l_by_l, A_, &(V_t_given_t_minus_1[t]));
    la::AddTo(Q_, &(V_t_given_t_minus_1[t]));
    
    la::MulOverwrite(C, V_t_given_t_minus_1[t], &temp1_o_by_o);
    la::MulTransBOverwrite(temp1_o_by_o, C, &temp2_o_by_o);
    la::AddTo(R, &temp2_o_by_o);
    la::Inverse(&temp2_o_by_o);
    la::MulTransAOverwrite(C, temp2_o_by_o, &temp1_o_by_o);
    la::MulOverwrite(V_t_given_t_minus_1[t], temp1_o_by_o, &K_t);
    
    Vector ideal, innovation;
    la::MulOverwrite(C, x_t_given_t_minus_1, &ideal);
    // y^t is the t^th column of Y
    Y.MakeColumnVector(t - 1, &y_t); //Y^(0) is actually y_{t=1}
    la::SubOverwrite(ideal, y_t, &innovation);
    X_t_given_t.MakeColumnVector(t, &x_t_given_t);
    la::MulOverwrite(K_t, innovation, &x_t_given_t);
    la::AddTo(x_t_given_t_minus_1, &x_t_given_t);

    la::MulOverwrite(K_t, C, &temp_o_by_l);
    la::MulOverwrite(temp_o_by_l, V_t_given_t_minus_1[t], &temp_l_by_l);
    la::SubOverwrite(temp1_o_by_o, V_t_given_t_minus_1[t], &(V_t_given_t[t]));
  }

  // Note that K_t is equal to K_T

  ArrayList<Matrix> V_hat_t;
  V_hat_t.Init(T+1);
  for(int t = 1; t <= T; t++) {
    V_hat_t[t].Init(n_dims_latent_, n_dims_latent_);
  }

  ArrayList<Matrix> V_hat_t_given_t_minus_1;
  V_hat_t_given_t_minus_1.Init(T+1);
  for(int t = 2; t <= T; t++) {
    V_hat_t_given_t_minus_1[t].Init(n_dims_latent_, n_dims_latent_);
  }

  V_hat_t_given_t_minus_1 = (I - K_T C) A V_t_given_t[T-1]
  _ - 
  

  



  
  



}

void Learn() {
}
