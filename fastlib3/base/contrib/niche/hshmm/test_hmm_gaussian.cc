#include "fastlib/fastlib.h"
#include "hmm.h"
#include "gaussian.h"
#include "mmk.h"

const fx_entry_doc hshmm_main_entries[] = {
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hshmm_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hshmm_main_doc = {
  hshmm_main_entries, hshmm_main_submodules,
  "This is used for a Hilbert-Schmidt embedding of an HMM.\n"
};



int main(int argc, char *argv[]) {
  fx_module* root = fx_init(argc, argv, &hshmm_main_doc);

  int n_dims = 2;
  int T = 10;


  HMM<Gaussian> hmm_a;
  hmm_a.Init(4, n_dims, T);
  hmm_a.RandomlyInitialize();
  hmm_a.ComputeCumulativePTransition();
  hmm_a.ComputeStateProbabilities();

  hmm_a.cumulative_p_transition().PrintDebug("cumulative_p_transition");
  hmm_a.state_probabilities().PrintDebug("state_probabilities");
  hmm_a.state_cumulative_probabilities().PrintDebug("state_cumulative_probabilities");


  HMM<Gaussian> hmm_b;
  hmm_b.Init(4, n_dims, T);
  hmm_b.RandomlyInitialize();
  hmm_b.ComputeCumulativePTransition();
  hmm_b.ComputeStateProbabilities();

  hmm_a.PrintDebug("a");
  hmm_b.PrintDebug("b");

  printf("Ready to MMK\n");
  
  MeanMapKernel mmk;
  mmk.Init(1, T);
  
  printf("hmm_dist(hmm_a, hmm_b) = %e\n",
	 mmk.Compute(hmm_a, hmm_b));


  fx_done(root);

  return SUCCESS_PASS;
}

