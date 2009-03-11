#include <fastlib/fastlib.h>
#include "discreteHMM.h"
#include "gaussianHMM.h"

const fx_entry_doc hmm_train_main_entries[] = {
  {"type", FX_REQUIRED, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"fileTR", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM transition.\n"},
  {"fileE", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM emission.\n"},
  {"fileSEQ", FX_PARAM, FX_STR, NULL,
   "  Input file for the sequences.\n"},
  {"outTR", FX_PARAM, FX_STR, NULL,
   "  Output file for the transition.\n"},
  {"outE", FX_PARAM, FX_STR, NULL,
   "  Output file for the emission.\n"},
  {"tol", FX_PARAM, FX_DOUBLE, NULL,
   "  Error tolerance, default = 1e-5.\n"},
  {"maxiter", FX_PARAM, FX_INT, NULL,
   "  Maximum number of iterations, default = 500.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hmm_train_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hmm_train_main_doc = {
  hmm_train_main_entries, hmm_train_main_submodules,
  "This is a program generating sequences from HMM models.\n"
};

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &hmm_train_main_doc );
  const char* type = fx_param_str_req(fx_root, "type");
  const char* fileTR = fx_param_str_req(fx_root, "fileTR");
  const char* fileE = fx_param_str_req(fx_root, "fileE");
  const char* fileSEQ = fx_param_str(fx_root, "fileSEQ", "seq.out");
  const char* outTR = fx_param_str(fx_root, "outTR", "tr.out");
  const char* outE = fx_param_str(fx_root, "outE", "e.out");
  double tolerance = fx_param_double(fx_root, "tol", 1e-5);
  index_t maxIteration = fx_param_int(fx_root, "maxiter", 500);
  
  TextLineReader f;
  f.Open(fileSEQ);
  if (strcmp(type, "discrete") == 0) {
    DiscreteHMM hmm;
    hmm.LoadTransition(fileTR);
    hmm.LoadEmission(fileE);
    ArrayList<DiscreteHMM::OutputSeq> seqs;
    DiscreteHMM::readSEQs(f, &seqs);
    hmm.Train(seqs, tolerance, maxIteration);
    hmm.Save(outTR, outE);
  }
  else if (strcmp(type, "gaussian") == 0) {
    GaussianHMM hmm;
    //hmm.LoadTransition(fileTR);
    //hmm.LoadEmission(fileE);
    srand(time(NULL));
    ArrayList<GaussianHMM::OutputSeq> seqs;
    GaussianHMM::readSEQs(f, &seqs);
    hmm.InitRandom(seqs[0][0].length(), 2);
    printf("numSeq = %d\n", seqs.size());
    hmm.Train(seqs, tolerance, maxIteration);
    hmm.Save(outTR, outE);
  }
  f.Close();
  
  fx_done(NULL);
}

