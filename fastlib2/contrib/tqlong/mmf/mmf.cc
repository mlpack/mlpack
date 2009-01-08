
#include <fastlib/fastlib.h>
#include "support.h"
#include "discreteHMM.h"

using namespace hmm_support;

void DiscreteHMM::TrainMMF(const ArrayList<Vector>& list_data_seq, 
			   int max_iteration, double tolerance) {
  
}

const fx_entry_doc mmf_main_entries[] = {
  {"seqfile", FX_REQUIRED, FX_STR, NULL,
   "  Output file for the data sequences.\n"},
  //  {"guess", FX_PARAM, FX_STR, NULL,
  //   "  File containing guessing HMM model profile.\n"},
  {"numstate", FX_REQUIRED, FX_INT, NULL,
   "  If no guessing profile specified, at least provide the"
   " number of states.\n"},
  {"profile", FX_REQUIRED, FX_STR, NULL,
   "  Output file containing trained HMM profile.\n"},
  {"maxiter", FX_PARAM, FX_INT, NULL,
   "  Maximum number of iterations, default = 500.\n"},
  {"tolerance", FX_PARAM, FX_DOUBLE, NULL,
   "  Error tolerance on log-likelihood as a stopping criteria.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc mmf_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc mmf_main_doc = {
  mmf_main_entries, mmf_main_submodules,
  "This is a program training HMM models from data sequences using MMF. \n"
};

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &mmf_main_doc);
  const char* seqin = fx_param_str_req(NULL, "seqfile");
  const char* proout = fx_param_str_req(NULL, "profile");

  ArrayList<Vector> seqs;
  load_vector_list(seqin, &seqs);

  int numstate = fx_param_int_req(NULL, "numstate");
  printf("Randomly generate parameters: NUMSTATE = %d\n", numstate);

  int maxiter = fx_param_int(NULL, "maxiter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  DiscreteHMM hmm;
  hmm.InitFromData(seqs, numstate);

  hmm.TrainMMF(seqs, maxiter, tol);

  hmm.SaveProfile(proout);
  fx_done(NULL);
}
