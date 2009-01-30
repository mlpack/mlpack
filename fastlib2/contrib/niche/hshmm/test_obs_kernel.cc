#include "hmm_distance.h"

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

  Distribution f;
  Distribution g;

  int n_dims = 1;

  f.Init(n_dims);
  g.Init(n_dims);

  Vector mu_f;
  mu_f.Init(n_dims);
  mu_f.SetZero();
  mu_f[0] = 0.1;
  f.SetMu(mu_f);

  Vector mu_g;
  mu_g.Init(n_dims);
  mu_g.SetZero();
  mu_g[0] = -0.1;
  g.SetMu(mu_g);

  Matrix sigma_f;
  sigma_f.Init(n_dims, n_dims);
  sigma_f.SetZero();
  for(int i = 0; i < n_dims; i++) {
    sigma_f.set(i, i, 1);
  }
  sigma_f.set(0, 0, 1.2);
  f.SetSigma(sigma_f);

  Matrix sigma_g;
  sigma_g.Init(n_dims, n_dims);
  sigma_g.SetZero();
  for(int i = 0; i < n_dims; i++) {
    sigma_g.set(i, i, 1);
  }
  sigma_g.set(0, 0, 0.8);
  g.SetSigma(sigma_g);

  double sim = HMM_Distance::ObservableKernel(f, g);
  
  printf("similarity(f,g) = %.9f\n", sim);


  fx_done(root);

  return SUCCESS_PASS;
}

