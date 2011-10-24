/**
 * @file viterbi.cc
 *
 * This file contains the program to compute the most probable state sequences
 * in a Hidden Markov Model of given sequences 
 * Model.
 *
 * Usage:
 *   viterbi --type=TYPE --profile=PROFILE --seqfile=FILE [OPTCLINS]
 * See the usage() function for complete option list
 */

#include "fastlib/fastlib.h"
#include "support.h"
#include "discreteHMM.h"
#include "gaussianHMM.h"
#include "mixgaussHMM.h"
#include "mixtureDST.h"

using namespace hmm_support;

bool viterbi_discrete();
bool viterbi_gaussian();
bool viterbi_mixture();
void usage();

const fx_entry_doc hmm_viterbi_main_entries[] = {
  {"type", FX_REQUIRED, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"profile", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM profile.\n"},
  {"seqfile", FX_PARAM, FX_STR, NULL,
   "  Output file for the data sequences.\n"},
  {"statefile", FX_PARAM, FX_STR, NULL,
   "  Output file for the most probable state sequences.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hmm_viterbi_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hmm_viterbi_main_doc = {
  hmm_viterbi_main_entries, hmm_viterbi_main_submodules,
  "This is a program computing the most probable state sequences \n"
  "of data sequences from HMM models.\n"
};

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &hmm_viterbi_main_doc);
  bool s = true;
  if (fx_param_exists(NULL,"type")) {
    const char* type = fx_param_str_req(NULL, "type");
    if (strcmp(type, "discrete")==0)
      s = viterbi_discrete();
    else if (strcmp(type, "gaussian")==0)
      s = viterbi_gaussian();
    else if (strcmp(type, "mixture")==0)
      s = viterbi_mixture();
    else {
      printf("Unrecognized type: must be: discrete | gaussian | mixture !!!\n");
      s = false;
    }
  }
  else {
    printf("Unrecognized type: must be: discrete | gaussian | mixture  !!!\n");
    s = false;
  }
  if (!(s)) usage();
  fx_done(NULL);
}

void usage() {
  printf("\n"
	 "Usage:\n"
	 "  viterbi --type=={discrete|gaussian|mixture} OPTCLINS\n"
	 "[OPTCLINS]\n"
	 "  --profile=file   : file contains HMM profile\n"
	 "  --seqfile=file   : file contains input sequences\n"
	 "  --statefile=file : output file for state sequences\n"
	 );
}

bool viterbi_mixture() {
  if (!fx_param_exists(NULL, "profile")) {
    printf("--profile must be defined.\n");
    return false;
  }
  const char* profile = fx_param_str_req(NULL, "profile");
  const char* seqin = fx_param_str(NULL, "seqfile", "seq.mix.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.viterbi.mix.out");

  MixtureofGaussianHMM hmm;
  hmm.InitFromFile(profile);

  ArrayList<Matrix> seqs;
  load_matrix_list(seqin, &seqs);

  TextWriter w_state;
  if (!(w_state.Open(stateout))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout);
    return false;
  }

  for (int i = 0; i < seqs.size(); i++) {
    Vector states;
    char s[100];

    hmm.ComputeViterbiStateSequence(seqs[i], &states);
    
    sprintf(s, "%% viterbi state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");
  }
  return true;
}

bool viterbi_gaussian() {
  if (!fx_param_exists(NULL, "profile")) {
    printf("--profile must be defined.\n");
    return false;
  }
  const char* profile = fx_param_str_req(NULL, "profile");
  const char* seqin = fx_param_str(NULL, "seqfile", "seq.gauss.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.viterbi.gauss.out");
  GaussianHMM hmm;
  hmm.InitFromFile(profile);

  ArrayList<Matrix> seqs;
  load_matrix_list(seqin, &seqs);

  TextWriter w_state;
  if (!(w_state.Open(stateout))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout);
    return false;
  }

  for (int i = 0; i < seqs.size(); i++) {
    Vector states;
    char s[100];
    hmm.ComputeViterbiStateSequence(seqs[i], &states);
    
    sprintf(s, "%% viterbi state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");
  }
  return true;
}

bool viterbi_discrete() {
  if (!fx_param_exists(NULL, "profile")) {
    printf("--profile must be defined.\n");
    return false;
  }
  const char* profile = fx_param_str_req(NULL, "profile");
  const char* seqin = fx_param_str(NULL, "seqfile", "seq.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.viterbi.out");

  DiscreteHMM hmm;

  hmm.InitFromFile(profile);

  ArrayList<Vector> seqs;
  load_vector_list(seqin, &seqs);

  TextWriter w_state;
  if (!(w_state.Open(stateout))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout);
    return false;
  }

  for (int i = 0; i < seqs.size(); i++) {
    Vector states;
    char s[100];
    
    hmm.ComputeViterbiStateSequence(seqs[i], &states);
    
    sprintf(s, "%% viterbi state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");
  }
  return true;
}
