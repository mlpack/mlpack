#include "fastlib/fastlib.h"
#include "mmk.h"
#include "ppk.h"

const fx_entry_doc test_obs_kernel_entries[] = {
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc test_obs_kernel_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc test_obs_kernel_doc = {
  test_obs_kernel_entries, test_obs_kernel_submodules,
  "This is for testing the observable kernel.\n"
};


int main(int argc, char *argv[]) {
  fx_module* root = fx_init(argc, argv, &test_obs_kernel_doc);

  Gaussian f;
  Gaussian g;

  int n_dims = 1;

  f.Init(n_dims);
  g.Init(n_dims);

  Vector mu_f;
  mu_f.Init(n_dims);
  mu_f.SetZero();
  int i;
  for(i = 0; i < n_dims; i++) {
    mu_f[i] = 0;
  }
  f.SetMu(mu_f);

  Vector mu_g;
  mu_g.Init(n_dims);
  mu_g.SetZero();
  for(i = 0; i < n_dims; i++) {
    mu_g[i] = 0;
  }
  g.SetMu(mu_g);

  Matrix sigma_f;
  sigma_f.Init(n_dims, n_dims);
  sigma_f.SetZero();
  for(i = 0; i < n_dims; i++) {
    sigma_f.set(i, i, 1);
  }
  sigma_f.set(0, 0, 1);
  f.SetSigma(sigma_f);

  Matrix sigma_g;
  sigma_g.Init(n_dims, n_dims);
  sigma_g.SetZero();
  for(i = 0; i < n_dims; i++) {
    sigma_g.set(i, i, 1);
  }
  sigma_g.set(0, 0, 1);
  g.SetSigma(sigma_g);



  f.PrintDebug("f");
  printf("\n");
  g.PrintDebug("g");
  

  int num_lambdas = 180;
  double lambda_increase_factor = 1.1;
  Vector lambda_array;
  /*
  lambda_array.Init(num_lambdas);
  
  lambda_array[0] = 1e-5;
  for(int i = 1; i < num_lambdas; i++) {
    lambda_array[i] = lambda_array[i-1] * lambda_increase_factor;
  }
  */
  
  // mean shift
  /*
  lambda_array.Init(4);
  lambda_array[0] = 1e-5;
  lambda_array[1] = 1e-2;
  lambda_array[2] = 5e-2;
  lambda_array[3] = 5e-1;
  */

  // covariance shift
  
  lambda_array.Init(4);
  lambda_array[0] = 1e-5;
  lambda_array[1] = 5e-3;
  lambda_array[2] = 5e-2;
  lambda_array[3] = 5e-1;
  

  int num_samples = 3000;
  //double mean_shift_increment = 0.01;
  double covariate_shift_increment = 0.01;
  Matrix results;
  results.Init(num_lambdas + 2, num_samples);

  MeanMapKernel mmk;
  mmk.Init(1); // set lambda = 1

  PPK ppk;
  ppk.Init(1); // set rho = 1

  double mmk_sim;
  double ppk_sim;
  for(int i = 0; i < num_samples; i++) {
    //results.set(0, i, g.mu()[0]);
    results.set(0, i, g.sigma().get(0, 0));
    ppk_sim = ppk.Compute(f, g);
    results.set(1, i, ppk_sim);
    
    for(int j = 0; j < num_lambdas; j++) {
      mmk.SetLambda(lambda_array[j]);
      mmk_sim = mmk.Compute(f, g);
      results.set(j + 2, i, mmk_sim);
    }
    
    //mu_g[0] += mean_shift_increment;
    //g.SetMu(mu_g);
    
    sigma_g.set(0, 0, sigma_g.get(0,0) + covariate_shift_increment);
    g.SetSigma(sigma_g);
  }


  Matrix lambda_array_matrix_alias;
  lambda_array_matrix_alias.AliasRowVector(lambda_array);
  data::Save("lambda_array.txt", lambda_array_matrix_alias);

  data::Save("sim_results.txt", results);



  fx_done(root);

  return SUCCESS_PASS;
}

