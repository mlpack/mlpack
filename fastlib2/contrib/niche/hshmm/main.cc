#include "hshmm.h"

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

  fx_done(root);

  HMM hmm;
  hmm.Init(4, 2, 25);

  hmm.RandomlyInitialize();
  hmm.CustomInitialize();

  hmm.PrintDebug();

  hmm.ComputeStateProbabilities();

  hmm.cumulative_p_transition().PrintDebug("cumulative_p_transition");

  hmm.state_probabilities().PrintDebug("state_probabilities");

  hmm.state_cumulative_probabilities().PrintDebug("state_cumulative_probabilities");

  Matrix random_draws;
  random_draws.Init(1,1000);
  for(int i = 0; i < 1000; i++) {
    random_draws.set(0, i, hmm.DrawStateGivenLastState(1));
  }

  data::Save("random_draws.txt", random_draws);

  






  // draw a state from P(q_t | theta)

  


  return SUCCESS_PASS;
}

