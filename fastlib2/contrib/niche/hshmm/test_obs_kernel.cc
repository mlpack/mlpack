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
  g.PrintDebug("g");
  

  Matrix results;
  int num_samples = 1000;
  double increment = 0.01;
  results.Init(2, num_samples);

  double sim;
  for(int j = 0; j < num_samples; j++) {
    sim = HMM_Distance::ObservableKernel(f, g);
    results.set(0, j, g.mu_[0]);
    results.set(1, j, sim);
    g.mu_[0] += increment;
  }

  data::Save("sim_results.txt", results);
  



  fx_done(root);

  return SUCCESS_PASS;
}

