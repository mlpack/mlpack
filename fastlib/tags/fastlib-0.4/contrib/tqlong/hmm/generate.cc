#include <fastlib/fastlib.h>
#include "discreteHMM.h"
#include "gaussianHMM.h"

const fx_entry_doc hmm_generate_main_entries[] = {
  {"type", FX_REQUIRED, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"fileTR", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM transition.\n"},
  {"fileE", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM emission.\n"},
  {"length", FX_PARAM, FX_INT, NULL,
   "  Sequence length, default = 10.\n"},
  {"lenmax", FX_PARAM, FX_INT, NULL,
   "  Maximum sequence length, default = length\n"},
  {"numseq", FX_PARAM, FX_INT, NULL,
   "  Number of sequance, default = 10.\n"},
  {"fileSEQ", FX_PARAM, FX_STR, NULL,
   "  Output file for the generated sequences.\n"},
  //{"statefile", FX_PARAM, FX_STR, NULL,
  // "  Output file for the generated state sequences.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hmm_generate_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hmm_generate_main_doc = {
  hmm_generate_main_entries, hmm_generate_main_submodules,
  "This is a program generating sequences from HMM models.\n"
};

void printSEQ(FILE* f, const ArrayList<index_t>& seq) {
  for (index_t i = 0; i < seq.size(); i++)
    fprintf(f, "%d,", seq[i]);
  fprintf(f, "\n");
}

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &hmm_generate_main_doc );
  const char* type = fx_param_str_req(fx_root, "type");
  const char* fileTR = fx_param_str_req(fx_root, "fileTR");
  const char* fileE = fx_param_str_req(fx_root, "fileE");
  int length = fx_param_int(fx_root, "length", 10);
  int lenmax = fx_param_int(fx_root, "lenmax", length);
  int numseq = fx_param_int(fx_root, "numseq", 10);
  const char* fileSEQ = fx_param_str(fx_root, "fileSEQ", "seq.out");
  
  FILE* f = fopen(fileSEQ, "w");
  if (strcmp(type, "discrete") == 0) {
    DiscreteHMM hmm;
    hmm.LoadTransition(fileTR);
    hmm.LoadEmission(fileE);
  	
    for (int i_seq = 0; i_seq < numseq; i_seq++) {
      ArrayList<index_t> seq;
      int len = (length >= lenmax)?length:rand()%(lenmax-length)+length;
      hmm.Generate(len, &seq, NULL);
      printSEQ(f, seq);
    }
  }
  else if (strcmp(type, "gaussian") == 0) {
    GaussianHMM hmm;
    hmm.LoadTransition(fileTR);
    hmm.LoadEmission(fileE);
    for (int i_seq = 0; i_seq < numseq; i_seq++) {
      GaussianHMM::OutputSeq seq;
      int len = (length >= lenmax)?length:rand()%(lenmax-length)+length;
      hmm.Generate(len, &seq, NULL);
      GaussianHMM::printSEQ(f, seq);
    }
  }
  fclose(f);
  fx_done(NULL);
}

