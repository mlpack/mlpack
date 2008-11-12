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
  hmm.Init(4, 2);

  hmm.RandomlyInitialize();

  hmm.PrintDebug();

  Matrix state_probabilities;
  hmm.CalculateStateProbabilities(5, &state_probabilities);

  state_probabilities.PrintDebug("state_probabilities");


  return SUCCESS_PASS;
}

