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

  int n_dims = 2;

  f.Init(n_dims);
  g.Init(n_dims);

  Vector mu;
  mu.Init(n_dims);
  mu.SetZero();
  f.SetMu(mu);
  g.SetMu(mu);

  Matrix sigma;
  sigma.Init(n_dims, n_dims);
  sigma.SetZero();
  for(int i = 0; i < n_dims; i++) {
    sigma.set(i, i, 1);
  }
  f.SetSigma(sigma);
  g.SetSigma(sigma);

  double sim = HMM_Distance::ObservableKernel(f, g);
  
  printf("similarity(f,g) = %f\n", sim);

  fx_done(root);

  return SUCCESS_PASS;
}

