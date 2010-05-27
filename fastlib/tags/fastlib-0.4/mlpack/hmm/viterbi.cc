/**
 * @file viterbi.cc
 *
 * This file contains the program to compute the most probable state sequences
 * in a Hidden Markov Model of given sequences 
 * Model.
 *
 * Usage:
 *   viterbi --type=TYPE --profile=PROFILE --seqfile=FILE [OPTIONS]
 * See the usage() function for complete option list
 */

#include <fastlib/fastlib.h>
#include "support.h"
#include "discreteHMM.h"
#include "gaussianHMM.h"
#include "mixgaussHMM.h"
#include "mixtureDST.h"
#include "hmm_documentation.h"

using namespace hmm_support;

success_t viterbi_discrete();
success_t viterbi_gaussian();
success_t viterbi_mixture();

const fx_entry_doc hmm_viterbi_main_entries[] = {
  {"input_model", FX_REQUIRED, FX_STR, NULL,
   "Input file containing a trained HMM profile\n"},
  {"input_sequence_file", FX_REQUIRED, FX_STR, NULL,
   "Input file of data sequences to evaluate\n"},
  {"output_state_file", FX_PARAM, FX_STR, NULL,
   "Output file for the most probable state sequences for each input sequence\n"
   "     (default output_state.<type>.seq)\n"},
  {"type", FX_REQUIRED, FX_STR, NULL,
   "HMM type: discrete | gaussian | mixture\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hmm_viterbi_main_submodules[] = {
  {"formats", &hmm_format_doc,
   "Documentation for file formats used by this program and other MLPACK HMM tools\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hmm_viterbi_main_doc = {
  hmm_viterbi_main_entries, hmm_viterbi_main_submodules,
  "The hmm_viterbi utility is used to find the most likely state sequences\n"
  "corresponding to given input sequences.  For a given input HMM profile and a\n"
  "given input file of sequences, this utility will find the most probable\n"
  "hidden state sequence and output that to a file.\n"
  "\n"
  "For more information on the file formats used by this and other MLPACK HMM\n"
  "utilities, see the 'formats' submodule documentation (--help=formats).\n"
};

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &hmm_viterbi_main_doc);
  success_t s = SUCCESS_PASS;
  if (fx_param_exists(NULL,"type")) {
    const char* type = fx_param_str_req(NULL, "type");
    if (strcmp(type, "discrete")==0)
      s = viterbi_discrete();
    else if (strcmp(type, "gaussian")==0)
      s = viterbi_gaussian();
    else if (strcmp(type, "mixture")==0)
      s = viterbi_mixture();
    else {
      FATAL("Unrecognized HMM type; must be 'discrete', 'gaussian', or 'mixture'.\n");
      s = SUCCESS_FAIL;
    }
  }
  else {
    FATAL("Unrecognized HMM type; must be 'discrete', 'gaussian', or 'mixture'.\n");
    s = SUCCESS_FAIL;
  }
  
  fx_done(NULL);
}

success_t viterbi_mixture() {
  const char* profile = fx_param_str_req(NULL, "input_model");
  const char* seqin = fx_param_str_req(NULL, "input_sequence_file");
  const char* stateout = fx_param_str(NULL, "output_state_file", "output_state.mix.seq");

  MixtureofGaussianHMM hmm;
  hmm.InitFromFile(profile);

  ArrayList<Matrix> seqs;
  load_matrix_list(seqin, &seqs);

  TextWriter w_state;
  if (!PASSED(w_state.Open(stateout))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout);
    return SUCCESS_FAIL;
  }

  for (int i = 0; i < seqs.size(); i++) {
    Vector states;
    char s[100];

    hmm.ComputeViterbiStateSequence(seqs[i], &states);
    
    sprintf(s, "%% viterbi state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");
  }
  return SUCCESS_PASS;
}

success_t viterbi_gaussian() {
  const char* profile = fx_param_str_req(NULL, "input_model");
  const char* seqin = fx_param_str_req(NULL, "input_sequence_file");
  const char* stateout = fx_param_str(NULL, "output_state_file", "output_state.gauss.seq");
  GaussianHMM hmm;
  hmm.InitFromFile(profile);

  ArrayList<Matrix> seqs;
  load_matrix_list(seqin, &seqs);

  TextWriter w_state;
  if (!PASSED(w_state.Open(stateout))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout);
    return SUCCESS_FAIL;
  }

  for (int i = 0; i < seqs.size(); i++) {
    Vector states;
    char s[100];
    hmm.ComputeViterbiStateSequence(seqs[i], &states);
    
    sprintf(s, "%% viterbi state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");
  }
  return SUCCESS_PASS;
}

success_t viterbi_discrete() {
  const char* profile = fx_param_str_req(NULL, "input_model");
  const char* seqin = fx_param_str_req(NULL, "input_sequence_file");
  const char* stateout = fx_param_str(NULL, "output_state_file", "output_state.dis.seq");

  DiscreteHMM hmm;

  hmm.InitFromFile(profile);

  ArrayList<Vector> seqs;
  load_vector_list(seqin, &seqs);

  TextWriter w_state;
  if (!PASSED(w_state.Open(stateout))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout);
    return SUCCESS_FAIL;
  }

  for (int i = 0; i < seqs.size(); i++) {
    Vector states;
    char s[100];
    
    hmm.ComputeViterbiStateSequence(seqs[i], &states);
    
    sprintf(s, "%% viterbi state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");
  }
  return SUCCESS_PASS;
}
