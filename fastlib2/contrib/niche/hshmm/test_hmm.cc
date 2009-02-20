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

  HMM<Gaussian> hmm_a;

  int T = 10;

  hmm_a.Init(4, 2, T);
  
  hmm_a.RandomlyInitialize();
  hmm_a.ComputeCumulativePTransition();
    

  //  hmm.CustomInitialize();
  //  hmm.PrintDebug();

  hmm_a.ComputeStateProbabilities();

  hmm_a.cumulative_p_transition().PrintDebug("cumulative_p_transition");
  
  hmm_a.state_probabilities().PrintDebug("state_probabilities");
  
  hmm_a.state_cumulative_probabilities().PrintDebug("state_cumulative_probabilities");

  /*  Matrix random_draws;
  random_draws.Init(1,1000);
  for(int i = 0; i < 1000; i++) {
    random_draws.set(0, i, hmm.DrawStateGivenLastState(1));
  }

  data::Save("random_draws.txt", random_draws);
  */

  
  printf("HMM_b\n");

  HMM<Gaussian> hmm_b;

  hmm_b.Init(4, 2, T);

  hmm_b.RandomlyInitialize();
  hmm_b.ComputeCumulativePTransition();
  
  //  hmm.CustomInitialize();

  //  hmm.PrintDebug();
  hmm_b.ComputeStateProbabilities();

  hmm_a.PrintDebug("a");
  hmm_b.PrintDebug("b");

  printf("Ready to MMK\n");
  
  MeanMapKernel mmk;
  mmk.Init(1, 70);

  
  printf("hmm_dist(hmm_a, hmm_b) = %e\n",
	 mmk.Compute(hmm_a, hmm_b));
  

  


  fx_done(root);

  


  return SUCCESS_PASS;
}

