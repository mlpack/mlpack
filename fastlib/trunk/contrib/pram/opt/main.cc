#include <fastlib/fastlib.h>
#include "phi.h"
#include "optimizers_reloaded.h"

long double test_function(Vector& theta, const Matrix& data, Vector* grad,
			  Vector *v, Vector *hv) {

  // First we make a model out of the theta
  // so mu <- theta[1,..,dim]
  // U = [ theta[dim+1] 0 0 ...;
  //       theta[dim+2] theta[dim+3] 0 ...;
  //       theta[dim+4] theta[dim+5] theta[dim+6] 0 ...;
  //       .....
  //     ];
  // Sigma = U' * U;

  if (v != NULL) {
    DEBUG_ASSERT(hv != NULL);
  }
  else {
    DEBUG_ASSERT(hv == NULL);
  }
  
  index_t dim = data.n_rows(), n = data.n_cols();
  Matrix sigma, r_lower, r_upper;
  ArrayList<Matrix> d_sigma;
  Vector mu;
  double s_min = 0.01; 
  double *temp_mu;
  index_t sigma_params = dim*(dim+1)/2;

  // obtaining the mu values
  temp_mu = (double*)malloc(dim * sizeof(double));

  for(index_t i = 0; i < dim; i++) {
    temp_mu[i] = theta.get(i);
  }
  mu.Copy(temp_mu, dim);

  // obtaining the sigma and d_sigma values
  d_sigma.Init(sigma_params);

  // the sigma values
  r_lower.Init(dim, dim);
  r_lower.SetAll(0.0);
  for(index_t i = 0; i < dim; i++) { 
    for(index_t j = 0; j < i; j++) {
      r_lower.set(i, j, theta[dim + i*(i+1)/2 + j]);
    }
    // adding small value to the diagonal of the 
    // covariance matrix to stop it from going to 
    // infinity by obtaining zero determinant of 
    // covariance
    r_lower.set(i, i, theta[dim + i*(i+1)/2 + i] + s_min);
  }
  la::TransposeInit(r_lower, &r_upper);
  la::MulInit(r_lower, r_upper, &sigma);


  long double l_theta = 0.0, f_theta, tmp_val;
  Vector g_mu, g_sigma, x;
  x.Init(dim);
  g_mu.Init(dim);
  g_mu.SetZero();
  g_sigma.Init(sigma_params);
  g_sigma.SetZero();
  
  if (v != NULL) {
    
    // v_vec_mu = [v_1, v_2, ... v_dim]
    // v_mat_sigma = [v_dim+1 0 0....; 
    //                v_dim+2 v_dim+3 0 .....;
    //                ..;
    //                ...v_(dim+(sigma_params) - 1) v_(dim+(sigma_params))]

    Vector v_vec_mu;
    v_vec_mu.Init(dim);
    for (index_t i = 0; i < dim; i++) {
      v_vec_mu.ptr()[i] = (*v)[i];
      DEBUG_ASSERT(v_vec_mu[i] == (*v)[i]);
    }


    Matrix v_sig_upper, v_sig_lower;
    v_sig_upper.Init(dim, dim);
    v_sig_upper.SetZero();
    for (index_t i = 0; i < dim; i++) {
      for (index_t j = 0; j < i+1; j++) {
	v_sig_upper.set(j, i, (*v)[dim + i*(i+1)/2 + j]);
      }
    }
    la::TransposeInit(v_sig_upper, &v_sig_lower);
  

    // the d_sigma values and the d_dr_d_sigma values
    Matrix d_sigma_d_r, temp_mat_a, temp_mat_b;
    ArrayList<Matrix> d_dr_d_sigma;

    d_dr_d_sigma.Init(sigma_params);
    d_sigma_d_r.Init(dim, dim);
    //d_sigma_d_r_t.Init(dim, dim);
    temp_mat_a.Init(dim, dim);
    temp_mat_b.Init(dim, dim);

    for(index_t i = 0; i < dim; i++) {
      for(index_t j = 0; j < i+1; j++) {
	d_sigma_d_r.SetAll(0.0);
	d_sigma_d_r.set(i, j, 1.0);
	//la::TransposeOverwrite(d_sigma_d_r, &d_sigma_d_r_t);

	la::MulOverwrite(d_sigma_d_r, r_upper, &temp_mat_a);
	la::MulOverwrite(d_sigma_d_r, v_sig_upper, &temp_mat_b);
	//la::MulOverwrite(r_lower, d_sigma_d_r_t, &temp_mat_b);
	la::TransposeInit(temp_mat_a, &d_sigma[i*(i+1)/2 + j]);
	la::TransposeInit(temp_mat_b, &d_dr_d_sigma[i*(i+1)/2]);
	//la::AddInit(temp_mat_a, temp_mat_b, &d_sigma[i*(i+1)/2 + j]);
	la::AddTo(temp_mat_a, &d_sigma[i*(i+1)/2 + j]);
	la::AddTo(temp_mat_b, &d_dr_d_sigma[i*(i+1)/2]);
      }
    }

    DEBUG_ASSERT(d_sigma.size() == sigma_params);
    DEBUG_ASSERT(d_dr_d_sigma.size() == sigma_params);

    // forming inv_sigma
    Matrix inv_sigma;
    la::InverseInit(sigma, &inv_sigma);

    // d_sigma_d_rv = (v_mat_sigma_t * r_upper + r_t * v_mat_sigma)
    Matrix d_sigma_d_rv, v_t_r, r_t_v;
    la::MulInit(v_sig_lower, r_upper, &v_t_r);
    la::MulInit(r_lower, v_sig_upper, &r_t_v);
    la::AddInit(v_t_r, r_t_v, &d_sigma_d_rv);

    // forming inv_sig * d_sig_d_rv * inv_sig
    Matrix inv_sig_d_sig_d_rv, inv_sig_d_sig_d_rv_inv_sig;
    la::MulInit(inv_sigma, d_sigma_d_rv, &inv_sig_d_sig_d_rv);
    la::MulInit(inv_sig_d_sig_d_rv, inv_sigma, &inv_sig_d_sig_d_rv_inv_sig);

    // forming inv_sig * v_vec_mu
    Vector inv_sig_v_vec_mu;
    la::MulInit(inv_sigma, v_vec_mu, &inv_sig_v_vec_mu);

    // Arraylists for inv_sigma * d_sigma_d_r_ij, 
    //                d_sigma_d_r_ij * inv_sigma,
    //                inv_sigma * d_sigma_d_r_ij * inv_sigma
    //                inv_sigma * d_dr_d_sigma_dr_ij
    //                inv_sigma * d_dr_d_sigma_dr_ij * inv_sigma
    ArrayList<Matrix> inv_s_ds_dr_ij, ds_dr_ij_inv_s, 
      inv_s_ds_dr_ij_inv_s, inv_s_d_dr_d_s, inv_s_d_dr_d_s_inv_s, 
      inv_s_d_s_d_rv_inv_s_d_s;
    inv_s_ds_dr_ij.Init(sigma_params);
    ds_dr_ij_inv_s.Init(sigma_params);
    inv_s_ds_dr_ij_inv_s.Init(sigma_params);
    inv_s_d_dr_d_s.Init(sigma_params); 
    inv_s_d_dr_d_s_inv_s.Init(sigma_params);
    inv_s_d_s_d_rv_inv_s_d_s.Init(sigma_params);

    for (index_t i = 0; i < sigma_params; i++) {
      la::MulInit(inv_sigma, d_sigma[i], &inv_s_ds_dr_ij[i]);
      la::MulInit(d_sigma[i], inv_sigma, &ds_dr_ij_inv_s[i]);
      la::MulInit(inv_s_ds_dr_ij[i], inv_sigma, &inv_s_ds_dr_ij_inv_s[i]);
      la::MulInit(inv_sigma, d_dr_d_sigma[i], &inv_s_d_dr_d_s[i]);
      la::MulInit(inv_s_d_dr_d_s[i], inv_sigma, &inv_s_d_dr_d_s_inv_s[i]);
      la::MulInit(inv_sig_d_sig_d_rv_inv_sig, d_sigma[i], &inv_s_d_s_d_rv_inv_s_d_s[i]);
    }

    // trace vector for all the sigma_params r_ij
    Vector trace_r;
    trace_r.Init(sigma_params);

    for (index_t i = 0; i < sigma_params; i++) {
      Matrix temp;
      la::AddInit(inv_s_d_s_d_rv_inv_s_d_s[i], inv_s_d_dr_d_s[i], &temp);

      DEBUG_ASSERT(temp.n_cols() == dim);
      DEBUG_ASSERT(temp.n_rows() == dim);
      double trace = 0.0;
      for (index_t j = 0; j < dim; j++) {
	trace += temp.get(j, j);
      }
      trace_r.ptr()[i] = trace;
    }

    // form hv_mu and hv_sigma
    Vector hv_mu, hv_sigma;
    hv_mu.Init(dim);
    hv_mu.SetZero();
    hv_sigma.Init(sigma_params);
    hv_sigma.SetZero();

    // calculating the value of the function for each data point
    // and adding it up
    // f_theta(x_i) = -log phi(x_i, mu, sigma);
    // g_mu(x_i) = (-1/phi(x_i, mu, sigma)) * d phi / d mu;
    // g_sigma(x_i) = (-1/phi(x_i, mu, sigma)) * d phi / d sigma;
    // l_theta = \sum_{i=1}^N  f_theta(x_i);
    // g_l_theta = grad = \sum_{i=1}^N [g_mu(x_i) g_sigma(x_i)]

     for(index_t i = 0; i < n; i++) {
      Vector d_phi_d_mu, d_phi_d_sigma;
      x.CopyValues(data.GetColumnPtr(i));
      tmp_val = phi(x, mu, sigma, d_sigma, 
		    &d_phi_d_mu, &d_phi_d_sigma);
      f_theta = -log(tmp_val);

      double alpha = -1.0 / tmp_val;
      la::AddExpert(alpha, d_phi_d_mu, &g_mu);
      la::AddExpert(alpha, d_phi_d_sigma, &g_sigma);

      l_theta += f_theta;

      Vector diff, hv_mu_i, hv_sigma_i;
    
      la::SubInit(mu, x, &diff);
      hv_sigma_i.Init(sigma_params);
    
      // forming hv_mu for x_i
      la::MulInit(inv_sig_d_sig_d_rv_inv_sig, diff, &hv_mu_i);
      la::SubFrom(inv_sig_v_vec_mu, &hv_mu_i);
      la::Scale(-1.0, &hv_mu_i);

      // forming hv_sigma for x_i
      Vector temp_vec;
      Matrix temp_mat;

      temp_vec.Init(dim);
      temp_mat.Init(dim, dim);

      for (index_t i = 0; i < sigma_params; i++) {

	double hv_sig_r_ij;

	la::MulOverwrite(v_vec_mu,inv_s_ds_dr_ij_inv_s[i], &temp_vec);
	hv_sig_r_ij = - la::Dot(temp_vec, diff);

	la::MulOverwrite(inv_sig_d_sig_d_rv_inv_sig, ds_dr_ij_inv_s[i], &temp_mat);
	la::MulOverwrite(diff, temp_mat, &temp_vec);
	hv_sig_r_ij += la::Dot(temp_vec, diff);

	la::MulOverwrite(diff, inv_s_d_dr_d_s_inv_s[i], &temp_vec);
	hv_sig_r_ij += la::Dot(temp_vec, diff);

	la::MulOverwrite(inv_s_ds_dr_ij[i], inv_sig_d_sig_d_rv_inv_sig, &temp_mat);
	la::MulOverwrite(diff, temp_mat, &temp_vec);
	hv_sig_r_ij += la::Dot(temp_vec, diff);

	la::MulOverwrite(inv_s_ds_dr_ij_inv_s[i], v_vec_mu, &temp_vec);
	hv_sig_r_ij -= la::Dot(diff, temp_vec);

	hv_sigma_i.ptr()[i] =-0.5 * (hv_sig_r_ij - trace_r[i]);
      }

      la::AddTo(hv_mu_i, &hv_mu);
      la::AddTo(hv_sigma_i, &hv_sigma);
    }

    // forming hv_theta
    hv->Init(dim + sigma_params);
  
    for (index_t i = 0; i < dim; i++) {
      hv->ptr()[i] = hv_mu[i];
      DEBUG_ASSERT(hv->get(i) == hv_mu[i]);
    }
  
    for (index_t i = 0; i < sigma_params; i++) {
      hv->ptr()[dim + i] = hv_sigma[i];
      DEBUG_ASSERT(hv->get(dim + i) == hv_sigma[i]);
    }

  }
  else {

    // the d_sigma values and the d_dr_d_sigma values
    Matrix d_sigma_d_r, temp_mat_a;

    d_sigma_d_r.Init(dim, dim);
    //d_sigma_d_r_t.Init(dim, dim);
    temp_mat_a.Init(dim, dim);

    for(index_t i = 0; i < dim; i++) {
      for(index_t j = 0; j < i+1; j++) {
	d_sigma_d_r.SetAll(0.0);
	d_sigma_d_r.set(i, j, 1.0);
	//la::TransposeOverwrite(d_sigma_d_r, &d_sigma_d_r_t);

	la::MulOverwrite(d_sigma_d_r, r_upper, &temp_mat_a);
	//la::MulOverwrite(r_lower, d_sigma_d_r_t, &temp_mat_b);
	la::TransposeInit(temp_mat_a, &d_sigma[i*(i+1)/2 + j]);
	//la::AddInit(temp_mat_a, temp_mat_b, &d_sigma[i*(i+1)/2 + j]);
	la::AddTo(temp_mat_a, &d_sigma[i*(i+1)/2 + j]);
      }
    }

    DEBUG_ASSERT(d_sigma.size() == sigma_params);

    // calculating the value of the function for each data point
    // and adding it up
    // f_theta(x_i) = -log phi(x_i, mu, sigma);
    // g_mu(x_i) = (-1/phi(x_i, mu, sigma)) * d phi / d mu;
    // g_sigma(x_i) = (-1/phi(x_i, mu, sigma)) * d phi / d sigma;
    // l_theta = \sum_{i=1}^N  f_theta(x_i);
    // g_l_theta = grad = \sum_{i=1}^N [g_mu(x_i) g_sigma(x_i)]

    for(index_t i = 0; i < n; i++) {
      Vector d_phi_d_mu, d_phi_d_sigma;
      x.CopyValues(data.GetColumnPtr(i));
      tmp_val = phi(x, mu, sigma, d_sigma, 
		    &d_phi_d_mu, &d_phi_d_sigma);
      f_theta = -log(tmp_val);

      double alpha = -1.0 / tmp_val;
      la::AddExpert(alpha, d_phi_d_mu, &g_mu);
      la::AddExpert(alpha, d_phi_d_sigma, &g_sigma);

      l_theta += f_theta;
    }
  }

  // forming the gradient grad
  double *temp_grad;
  temp_grad = (double*)malloc(theta.length() * sizeof(double));
  for(index_t i = 0; i < dim; i++) {
    temp_grad[i] = g_mu.get(i);
  }

  for(index_t i = 0; i < sigma_params; i++) {
    temp_grad[dim+i] = g_sigma.get(i);
  }
  grad->CopyValues(temp_grad);

  return l_theta;
}

long double test_function(Vector& point, const Matrix& data, Vector *grad) {
  return test_function(point, data, grad, NULL, NULL);
}

int main(int argc, char* argv[]) {

  fx_init(argc, argv);

  const char *datafile = fx_param_str_req(NULL, "data");

  Matrix data_points;
  data::Load(datafile, &data_points);

  double temp_array[] = {4, -2, 3, 1, 2};
  double *real_theta_array;
  //long double function_val = 4607.5;
  Vector real_theta, grad;
  index_t len = 5;

  real_theta_array = (double*)malloc(5 * sizeof(double));
  for(index_t i = 0; i < 5; i++) {
    real_theta_array[i] = temp_array[i];
  }
  real_theta.Copy(real_theta_array, len);
  grad.Init(5);

  
  //long double val = test_function(real_theta, data_points, &grad);
  //printf("%Lf %Lf\n", val, function_val);

  datanode *opt_module = fx_submodule(NULL,"opt","opt");
  fx_param_int(opt_module,"param_space_dim", 5);

  QuasiNewton opt;
  //SMDSS opt;

  fx_param_str(opt_module, "method", "QuasiNewton");
  //fx_param_str(opt_module, "method", "SMD_SingleStep");
  opt.Init(test_function, data_points, opt_module);
  double *pt;
  double p[] = {1, 2, 3, 1, 6};
  pt = (double*)malloc(5 * sizeof(double));
  for(index_t i = 0; i < 5; i++) {
    pt[i] = p[i];
  }
  fx_timer_start(opt_module,"opt_time");
  opt.Eval(pt);
  fx_timer_stop(opt_module,"opt_time");

  printf("theta : [");
  for(index_t i = 0; i < 5; i++) {
  printf(" %lf,",pt[i]);
  }
  printf("\b ]\n");

  Vector calc_theta;
  calc_theta.Copy(pt, 5);
  long double min_ob = test_function(calc_theta, data_points, &grad);
  printf("%Lf\n",min_ob);
  //fx_silence();
  fx_done();

  return 1;
}

/**
 * The actual minimum obtained by the Quasi Newton 
 * method is 4599.772730
 * The minima is [3.870753, -1.952005, 2.819059, 0.831117, 2.048447]
 * The time required was : 0.255378 sec
 * Iterations through the data : 19
 * 
 * True values:
 * [4 -2 3 1 2]
 * likelihood value : 4607.5 (1000)
 * likelihood value : 463460.00 (100000)
 * 
 * Quasi Newton: 36.496303
 * theta : [ 4.009524, -1.989121, -3.021029, -1.006317, 1.992536 ]
 * minimum value : 463457.311250
 * Iterations through the data : 25
 */
