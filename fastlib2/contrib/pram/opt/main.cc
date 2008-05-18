#include <fastlib/fastlib.h>
#include "phi.h"
#include "optimizers_reloaded.h"

long double test_function(Vector& theta, const Matrix& data, Vector* grad) {

  // First we make a model out of the theta
  // so mu <- theta[1,..,dim]
  // U = [ theta[dim+1] 0 0 ...;
  //       theta[dim+2] theta[dim+3] 0 ...;
  //       theta[dim+4] theta[dim+5] theta[dim+6] 0 ...;
  //       .....
  //     ];
  // Sigma = U' * U;
  
  index_t dim = data.n_rows(), n = data.n_cols();
  Matrix sigma, lower_triangle_matrix, upper_triangle_matrix;
  ArrayList<Matrix> d_sigma;
  Vector mu;
  double s_min = 0.01; 
  double *temp_mu;

  // obtaining the mu values
  temp_mu = (double*)malloc(dim * sizeof(double));

  for(index_t i = 0; i < dim; i++) {
    temp_mu[i] = theta.get(i);
  }
  mu.Copy(temp_mu, dim);

  //printf("Mu : [");
  //for(index_t i = 0; i < dim; i++) {
  //printf("%lf ",mu.get(i));
  //}
  //printf("\b];\n");

  // obtaining the sigma and d_sigma values
  d_sigma.Init(dim*(dim+1)/2);

  // the sigma values
  lower_triangle_matrix.Init(dim, dim);
  lower_triangle_matrix.SetAll(0.0);
  for(index_t i = 0; i < dim; i++) { 
    for(index_t j = 0; j < i; j++) {
      lower_triangle_matrix.set(i, j, theta[dim + i*(i+1)/2 + j]);
    }
    // adding small value to the diagonal of the 
    // covariance matrix to stop it from going to 
    // infinity by obtaining zero determinant of 
    // covariance
    lower_triangle_matrix.set(i, i, theta[dim + i*(i+1)/2 + i] + s_min);
  }
  la::TransposeInit(lower_triangle_matrix, &upper_triangle_matrix);
  la::MulInit(lower_triangle_matrix, upper_triangle_matrix, &sigma);

  // the d_sigma values
  Matrix d_sigma_d_r, d_sigma_d_r_t, temp_matrix1, temp_matrix2;
  d_sigma_d_r.Init(dim, dim);
  d_sigma_d_r_t.Init(dim, dim);
  temp_matrix1.Init(dim, dim);
  temp_matrix2.Init(dim, dim);

  for(index_t i = 0; i < dim; i++) {
    for(index_t j = 0; j < i+1; j++) {
      d_sigma_d_r.SetAll(0.0);
      d_sigma_d_r.set(i, j, 1.0);
      la::TransposeOverwrite(d_sigma_d_r, &d_sigma_d_r_t);

      la::MulOverwrite(d_sigma_d_r, upper_triangle_matrix, &temp_matrix1);
      la::MulOverwrite(lower_triangle_matrix, d_sigma_d_r_t, &temp_matrix2);
      la::AddInit(temp_matrix1, temp_matrix2, &d_sigma[i*(i+1)/2 + j]);
    }
  }

  //printf("Sigma : [");
  //for(index_t i = 0; i < dim; i++) {
  //for(index_t j = 0; j < dim; j++) {
  //  printf("%lf ",sigma.get(i,j));
  //}
  //printf("\b;");
  //}
  //printf("\b]\n");

  // calculating the value of the function for each data point
  // and adding it up
  // f_theta(x_i) = -log phi(x_i, mu, sigma);
  // g_mu(x_i) = (-1/phi(x_i, mu, sigma)) * d phi / d mu;
  // g_sigma(x_i) = (-1/phi(x_i, mu, sigma)) * d phi / d sigma;
  // l_theta = \sum_{i=1}^N  f_theta(x_i);
  // g_l_theta = grad = \sum_{i=1}^N [g_mu(x_i) g_sigma(x_i)]

  long double l_theta = 0.0, f_theta, tmp_val;
  Vector g_mu, g_sigma, x;
  x.Init(dim);
  g_mu.Init(dim);
  g_mu.SetZero();
  g_sigma.Init(dim*(dim+1)/2);
  g_sigma.SetZero();

  //printf("calc func val\n");
  //fflush(NULL);
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

  //printf("%Lf setting grad val\n", l_theta);
  //fflush(NULL);
  double *temp_grad;
  temp_grad = (double*)malloc(theta.length() * sizeof(double));
  for(index_t i = 0; i < dim; i++) {
    temp_grad[i] = g_mu.get(i);
  }

  for(index_t i = 0; i < dim*(dim+1)/2; i++) {
    temp_grad[dim+i] = g_sigma.get(i);
  }
  grad->CopyValues(temp_grad);
  return l_theta;
}

int main(int argc, char* argv[]) {

  fx_init(argc, argv, NULL);

  const char *datafile = fx_param_str_req(NULL, "data");

  Matrix data_points;
  data::Load(datafile, &data_points);

  double temp_array[] = {4, -2, 3, 1, 2};
  double *real_theta_array;
  long double function_val = 4607.5;
  Vector real_theta, grad;
  index_t len = 5;

  real_theta_array = (double*)malloc(5 * sizeof(double));
  for(index_t i = 0; i < 5; i++) {
    real_theta_array[i] = temp_array[i];
  }
  real_theta.Copy(real_theta_array, len);
  grad.Init(5);

  //printf("entering func\n");
  //fflush(NULL);
  long double val = test_function(real_theta, data_points, &grad);
  //printf("exiting func\n");
  //fflush(NULL);
  printf("%Lf %Lf\n", val, function_val);

  datanode *opt_module = fx_submodule(NULL,"opt");
  fx_param_int(opt_module,"param_space_dim", 5);

  //QuasiNewton opt;
  SMD_SingleStep opt;

  // fx_param_str(opt_module, "method", "QuasiNewton");
  fx_param_str(opt_module, "method", "SMD_SingleStep");
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
  fx_done(NULL);

  return 1;
}

/**
 * The actual minimum obtained by the Quasi Newton 
 * method is 4599.772730
 * The minima is [3.870753, -1.952005, 2.819059, 0.831117, 2.048447]
 * The time required was : 0.255378 sec
 * Iterations through the data : 19
 */
