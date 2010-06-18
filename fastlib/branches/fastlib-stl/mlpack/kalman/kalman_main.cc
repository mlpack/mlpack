#include <iostream>
#include <fastlib/fastlib.h>
#include "kalman_helper.h"
#include "kalman.h"

#include <armadillo>
#include <fastlib/base/arma_compat.h>

using namespace std;

/**
 * @file kalman_main.cc
 *
 * This file contains a sample demo of a time-invariant 
 * kalman filter at work
 * 
 * User needs to input a dynamical system to interact with kf
 * A lds with the same params. as the kf is assumed here
 *
 * User specified inputs are in the form of *.csv files contained
 * in the same folder as this file main.cc. A Matlab m.file (test_kf.m)
 * is provided to facilitate this process. See comments therein
 *
 * Inputs: 
 * t_in: duration of expt.
 * a_in, b_in, c_in, q_in, r_in, s_in: sys. params
 * x_pred_0_in, p_pred_0_in: initializes kf
 * y_pred_0_in, inno_cov_0_in: initalizes kf
 * 
 * Outputs: mainly for comparison with matlab code see test_kf.m
 * These will be stored as *.csv files in the directory
 * w_out, v_out: stochastic signals
 * u_out, x_out, y_out: system signals
 * x_pred_out, p_pred_end_out: 1-step ahead predictions & err. cov.
 * x_hat_out, p_hat_end_out: filtered state estimates & err cov.
 * y_pred_out, inno_cov_end_out: 1-step ahead mst. predictions & err cov.
 * k_gain_end_out: kalman gain
 * note that only the final err. covs are returned because k.f. converges
 * 
 * @see kalman.h
 */

int main(int argc, char* argv[]) {

  fx_init(argc, argv, NULL);
 
  ///////////LOADING DATA AND SETTING PARAMETERS//////////////////////////

  // User-specified: duration of experiment 
  // Note that the experiment will run from t=0 to t_tot
  const char* t_in = fx_param_str(NULL, "t_in", "t_in");
  Matrix t_tot_mat;
  arma::mat tmp;
  data::Load(t_in, tmp);
  arma_compat::armaToMatrix(tmp, t_tot_mat);
  int t_tot = (int)t_tot_mat.get(0, 0);
  
  // User-specified: system parameters. Stored in *.csv files 
  // a_mat, b_mat, c_mat, d_mat, q_mat, r_mat, s_mat
  // are the typical time-invariant system params
  // Proceed to Load init matrices within a struct called lds
  const char* a_in = fx_param_str(NULL, "a_in", "a_in");
  const char* b_in = fx_param_str(NULL, "b_in", "b_in");
  const char* c_in = fx_param_str(NULL, "c_in", "c_in");
  const char* q_in = fx_param_str(NULL, "q_in", "q_in");
  const char* r_in = fx_param_str(NULL, "r_in", "r_in");
  const char* s_in = fx_param_str(NULL, "s_in", "s_in");  
  
  ssm lds;
  data::Load(a_in, tmp);
  arma_compat::armaToMatrix(tmp, lds.a_mat);
  data::Load(b_in, tmp); 
  arma_compat::armaToMatrix(tmp, lds.b_mat);
  data::Load(c_in, tmp); 
  arma_compat::armaToMatrix(tmp, lds.c_mat);
  data::Load(q_in, tmp); 
  arma_compat::armaToMatrix(tmp, lds.q_mat);
  data::Load(r_in, tmp); 
  arma_compat::armaToMatrix(tmp, lds.r_mat);
  data::Load(s_in, tmp);  
  arma_compat::armaToMatrix(tmp, lds.s_mat);
	     
  // User-specified: kf parameters. Stored in *.csv files 
  // Includes x_{0|-1}, p_pred_{0|-1}, y_{0|-1}, inno_cov_{0|-1}
  // {t|t-1} means var. at time t given info. up to t-1
  const char* x_pred_0_in = fx_param_str(NULL, "x_pred_0_in", "x_pred_0_in");
  const char* p_pred_0_in = fx_param_str(NULL, "p_pred_0_in", "p_pred_0_in");
  const char* y_pred_0_in = fx_param_str(NULL, "y_pred_0_in", "y_pred_0_in");
  const char* inno_cov_0_in = fx_param_str(NULL, "inno_cov_0_in", 
					   "inno_cov_0_in");
  
  Matrix x_pred_0; 
  data::Load(x_pred_0_in, tmp);
  arma_compat::armaToMatrix(tmp, x_pred_0);
  Matrix y_pred_0;
  data::Load(y_pred_0_in, tmp); 
  arma_compat::armaToMatrix(tmp, y_pred_0);
  Matrix p_pred_0;
  data::Load(p_pred_0_in, tmp);
  arma_compat::armaToMatrix(tmp, p_pred_0);
  Matrix inno_cov_0;
  data::Load(inno_cov_0_in, tmp);
  arma_compat::armaToMatrix(tmp, inno_cov_0);

  // Set up parameters to be used by signal generator. 
  // Assumed to be an lds with the same params. as the KF
  // for the purpose of demonstration
  // assume that x is initially zero
  int nx = lds.a_mat.n_rows(); 
  int ny = lds.c_mat.n_rows(); 
  int nu = lds.b_mat.n_cols();
  Matrix x(nx, t_tot + 1); x.SetZero();
  Matrix y(ny, t_tot + 1); 
  Matrix u(nu, t_tot + 1); 
  Matrix w(nx, t_tot + 1); 
  Matrix v(ny, t_tot + 1); 

  // Set up params. for the KF including:
  // estimates: x_{t|t-1}, x_{t|t}, y_{t|t-1}
  // error covariances: p_pred_{t|t-1}, p_hat{t|t-1}, inno_cov_{t|t-1}
  // Kalman gain
  // Also initialize values and sizes where appropriate
  Matrix x_pred(nx, t_tot + 1); 
  Matrix x_hat(nx, t_tot + 1);
  Matrix y_pred(ny, t_tot + 1);
  Matrix p_pred[t_tot + 1]; 
  Matrix p_hat[t_tot + 1];
  Matrix inno_cov[t_tot + 1];
  Matrix k_gain[t_tot + 1];
  set_portion_of_matrix(x_pred_0, 0, nx-1, 0, 0, &x_pred);
  set_portion_of_matrix(y_pred_0, 0, ny-1, 0, 0, &y_pred);
  for (int t =0; t<=t_tot; t++ ) {
    p_pred[t].Init(nx, nx);
    p_hat[t].Init(nx, nx);
    k_gain[t].Init(nx, ny);
    inno_cov[t].Init(ny, ny);
  };    
  p_pred[0]   = p_pred_0;
  inno_cov[0] = inno_cov_0;

  // Define other params. to be used
  // noise_matrix = [q_mat s_mat; s_mat' r_mat]
  Matrix noise_mat(nx + ny, nx + ny), s_trans;
  la:: TransposeInit(lds.s_mat, &s_trans);
  set_portion_of_matrix(lds.q_mat, 0, nx - 1, 0, nx - 1, &noise_mat);
  set_portion_of_matrix(lds.s_mat, 0, nx - 1, nx, nx + ny -1, &noise_mat);
  set_portion_of_matrix(s_trans, nx, nx + ny - 1,0, nx - 1, &noise_mat);
  set_portion_of_matrix(lds.r_mat, nx, nx + ny - 1, nx, nx + ny - 1, &noise_mat); 


  ///////////////////BEGIN DEMO /////////////////////////
  for (int t=0; t<=t_tot; t++) {
    /*
      Signal generation. As a demo, the system will also 
      be an lds with the same parameters as the k.f. 
      although this doesn't need to be necessarily true */
    
    // Generate noise
    // Also make aliases which do not need to be init'd
    Vector w_t; w.MakeColumnVector(t, &w_t); 
    Vector v_t; v.MakeColumnVector(t, &v_t);    
    Vector w_v_t; w_v_t.Init(nx+ny);
    RandVector(noise_mat, w_v_t);
    extract_sub_vector_of_vector(w_v_t, 0, nx-1, &w_t);
    extract_sub_vector_of_vector(w_v_t, nx, nx+ny-1, &v_t);  

    // y_t = c_mat*x_{t} +  v_t 
    // Also make aliases which do not need to be init'd
    Vector x_t; x.MakeColumnVector(t, &x_t);
    Vector y_t; y.MakeColumnVector(t, &y_t);
    propagate_one_step(lds.c_mat, x_t, v_t, &y_t);

    /*
      Perform measurement update */
    // Alg. takes (lds, y_t, x_{t|t-1}, p_{t|t-1}, inno_cov_{t|t-1})
    // outputs (x_{t|t}, p_{t|t}, k_gain{t})
    // Also make aliases which do not need to  be init'd
    Vector x_pred_t; x_pred.MakeColumnVector(t, &x_pred_t);
    Vector y_pred_t; y_pred.MakeColumnVector(t, &y_pred_t);
    Vector x_hat_t; x_hat.MakeColumnVector(t, &x_hat_t);
    KalmanFiltTimeInvariantMstUpdate(lds, y_t, x_pred_t, p_pred[t], 
				     y_pred_t, inno_cov[t], x_hat_t, 
				     p_hat[t], k_gain[t]);

    /* The system might want to do sth. here
       Ex. u_t = f(x_{t|t})
       Here, u_t is assumed to be a normally distributed vec.
       Also make aliases*/
    Vector u_t; 
    u.MakeColumnVector(t, &u_t);
    RandVector(u_t);


    /* Perform Time update 
     Alg. takes (lds, x_{t|t}, p_{t|t}, y_t, u_t)
     outputs (x_{t+1|t}, y_{t+1|t}, p_{t+1|t}, inno_cov{t+1|t}) 
     Also make aliases that do not need to be init'd. 
     But do so if not at final time step */
    if (t <  t_tot) { 
      Vector x_pred_t_next; x_pred.MakeColumnVector(t + 1, &x_pred_t_next);
      Vector y_pred_t_next; y_pred.MakeColumnVector(t + 1, &y_pred_t_next);
      KalmanFiltTimeInvariantTimeUpdate(lds, x_hat_t, p_hat[t], y_t, u_t, 
					x_pred_t_next, y_pred_t_next, 
					p_pred[t + 1], inno_cov[t + 1]);      
      
    };
    /* Propagate states 1-step forward 
       Again the true plant/ system is up to the user to define
       x_{t+1} = a_mat*x_{t} + b_mat*u_{t} + w_{t} 
       unless we are at the last time step */      
    if (t <  t_tot) { 
      Vector x_t_next; x.MakeColumnVector(t+1, &x_t_next);
      propagate_one_step(lds.a_mat, lds.b_mat, x_t, u_t, w_t, &x_t_next);
    };
  }; /* demo */
  
  ///////////////////SAVE RESULTS TO A FILE///////////////
  
  /* We will store the results so that a similar Matlab
     implementation can be used as a comparison
     To save:
     w, v (stochastic realizations)
     u, x, y (actions)
     x_pred, p_pred[t_tot] (expect this to converge)
     x_hat, p_hat[t_tot] (expect this to converge)
     y_pred, inno_cov[t_tot] (expect this to converge)
     k_gain[t_tot] (expect this to converge)
     
     These will be saved as separate *.csv files
     See test_kf.m for comparison tests 
     and plotting capabilities */
  
  Matrix w_trans, v_trans; 
  Matrix u_trans, x_trans, y_trans; 
  Matrix x_pred_trans;
  Matrix x_hat_trans; 
  Matrix y_pred_trans; 
  Matrix k_gain_end_trans;
    
  la::TransposeInit(w, &w_trans);
  la::TransposeInit(v, &v_trans);
  la::TransposeInit(u, &u_trans);
  la::TransposeInit(x, &x_trans);
  la::TransposeInit(y, &y_trans);
  la::TransposeInit(x_pred, &x_pred_trans);
  la::TransposeInit(x_hat, &x_hat_trans);
  la::TransposeInit(y_pred, &y_pred_trans);
  la::TransposeInit(k_gain[t_tot], &k_gain_end_trans);
 
  arma_compat::matrixToArma(w_trans, tmp); 
  data::Save("w_out", tmp);
  arma_compat::matrixToArma(v_trans, tmp); 
  data::Save("v_out", tmp);
  arma_compat::matrixToArma(u_trans, tmp); 
  data::Save("u_out", tmp);
  arma_compat::matrixToArma(x_trans, tmp); 
  data::Save("x_out", tmp);
  arma_compat::matrixToArma(y_trans, tmp); 
  data::Save("y_out", tmp);
  arma_compat::matrixToArma(x_pred_trans, tmp); 
  data::Save("x_pred_out", tmp);
  arma_compat::matrixToArma(p_pred[t_tot], tmp); 
  data::Save("p_pred_end_out", tmp);
  arma_compat::matrixToArma(x_hat_trans, tmp); 
  data::Save("x_hat_out", tmp);
  arma_compat::matrixToArma(p_hat[t_tot], tmp); 
  data::Save("p_hat_end_out", tmp);
  arma_compat::matrixToArma(y_pred_trans, tmp); 
  data::Save("y_pred_out", tmp);
  arma_compat::matrixToArma(inno_cov[t_tot], tmp); 
  data::Save("inno_cov_end_out", tmp);
  arma_compat::matrixToArma(k_gain_end_trans, tmp); 
  data::Save("k_gain_end_out", tmp);             
  
  fx_done(fx_root);
}; /* main */
