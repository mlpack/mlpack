#include <fastlib/fastlib.h>
#include "discreteHMM.h"
#include "gaussianHMM.h"

const fx_entry_doc hmm_decode_main_entries[] = {
  {"type", FX_REQUIRED, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"fileTR", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM transition.\n"},
  {"fileE", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM emission.\n"},
  {"fileSEQ", FX_PARAM, FX_STR, NULL,
   "  Output file for the sequences.\n"},
  {"fileLOG", FX_PARAM, FX_STR, NULL,
   "  Output file for the log-likelihood.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hmm_decode_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hmm_decode_main_doc = {
  hmm_decode_main_entries, hmm_decode_main_submodules,
  "This is a program generating sequences from HMM models.\n"
};



int main(int argc, char* argv[]) {
  fx_init(argc, argv, &hmm_decode_main_doc );
  const char* type = fx_param_str_req(fx_root, "type");
  const char* fileTR = fx_param_str_req(fx_root, "fileTR");
  const char* fileE = fx_param_str_req(fx_root, "fileE");
  const char* fileSEQ = fx_param_str(fx_root, "fileSEQ", "seq.out");
  const char* fileLOG = fx_param_str(fx_root, "fileLOG", "log.out");
  
  TextLineReader f;
  f.Open(fileSEQ);
  FILE* fout = fopen(fileLOG, "w");
  if (strcmp(type, "discrete") == 0) {
    DiscreteHMM hmm;
    hmm.LoadTransition(fileTR);
    hmm.LoadEmission(fileE);
    while (1) {
      DiscreteHMM::OutputSeq seq;
      DiscreteHMM::readSEQ(f, &seq);
      if (seq.size() == 0) break;
      Matrix pStates, fs, bs;
      Vector s;
      double logSeq = hmm.Decode(seq, &pStates, &fs, &bs, &s);
      fprintf(fout, "%f\n", logSeq);
    }
  }
  else if (strcmp(type, "gaussian") == 0) {
    GaussianHMM hmm;
    hmm.LoadTransition(fileTR);
    hmm.LoadEmission(fileE);
    while (1) {
      GaussianHMM::OutputSeq seq;
      GaussianHMM::readSEQ(f, &seq);
      printf("length = %d\n", seq.size());
      if (seq.size() == 0) break;
      Matrix pStates, fs, bs, pOutput;
      Vector s;
      double logSeq = hmm.Decode(seq, &pStates, &fs, &bs, &s, &pOutput);
      fprintf(fout, "%f\n", logSeq);
    }
  }
  f.Close();
  fclose(fout);
  
  fx_done(NULL);
}

