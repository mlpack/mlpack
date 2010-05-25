/**
 * @file generate.cc
 *
 * This file contains the program to generate sequences from a Hidden Markov
 * Model.
 *
 * Usage:
 *   generate --type=TYPE --profile=PROFILE [OPTIONS]
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

success_t generate_discrete();
success_t generate_gaussian();
success_t generate_mixture();

const fx_entry_doc hmm_generate_main_entries[] = {
  {"input_model", FX_REQUIRED, FX_STR, NULL,
   "A .hmm file containing an HMM profile.\n"},
  {"output_observation_file", FX_PARAM, FX_STR, NULL,
   "Output file for the generated observed variable sequences.\n"
   "     (default observed.<type>.seq)\n"},
  {"output_state_file", FX_PARAM, FX_STR, NULL,
   "Output file for the generated hidden state sequences.\n"
   "     (default state.<type>.seq)\n"},
  {"type", FX_REQUIRED, FX_STR, NULL,
   "HMM type: discrete | gaussian | mixture\n"},
  {"length", FX_PARAM, FX_INT, NULL,
   "Sequence length (default 10)\n"},
  {"lenmax", FX_PARAM, FX_INT, NULL,
   "Maximum sequence length (default <length>)\n"},
  {"numseq", FX_PARAM, FX_INT, NULL,
   "Number of sequences to generate (default 10).\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hmm_generate_main_submodules[] = {
  {"formats", &hmm_format_doc,
  "Documentation for file formats used by this program and other MLPACK HMM tools\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hmm_generate_main_doc = {
  hmm_generate_main_entries, hmm_generate_main_submodules,
  "The hmm_generate utility is used to generate a random sequence from an input\n"
  "HMM profile, which is given as a parameter.  Then, random sequences are\n"
  "generated and stored in two files; one file (specified by\n"
  "--output_observation_file) stores the observed variable sequences, and the\n"
  "other (specified by --output_state_file) stores the hidden state sequences.\n"
  "\n"
  "The maximum sequence length parameter (lenmax) can be used to generate a\n"
  "series of sequences of varying length.  For instance, if numseq is 3, length\n"
  "is 10, and lenmax is 12, the first sequence will be 10 states long; the second\n"
  "will be 11 states long; the last will be 12 states long.  However, lenmax must\n"
  "always be greater than length.\n"
  "\n"
  "For more information on the formats used by this utility, see the help in\n"
  "the formats submodule (--help=formats).\n"
};

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &hmm_generate_main_doc );
  success_t s = SUCCESS_PASS;
  if (fx_param_exists(NULL,"type")) {
    const char* type = fx_param_str_req(NULL, "type");
    if (strcmp(type, "discrete")==0)
      s = generate_discrete();
    else if (strcmp(type, "gaussian")==0) 
      s = generate_gaussian();
    else if (strcmp(type, "mixture")==0)
      s = generate_mixture();
    else {
      FATAL("Unrecognized type; must be 'discrete', 'gaussian', or 'mixture'.\n");
      return SUCCESS_PASS;
    }
  }
  else {
    FATAL("Unrecognized type; must be 'discrete', 'gaussian', or 'mixture'.\n");
    s = SUCCESS_FAIL;
  }

  fx_done(NULL);
}

success_t generate_mixture() {
  const char* profile = fx_param_str_req(NULL, "input_model");
  const int seqlen = fx_param_int(NULL, "length", 10);
  const int seqlmax = fx_param_int(NULL, "lenmax", seqlen);
  const int numseq = fx_param_int(NULL, "numseq", 10);
  const char* seqout = fx_param_str(NULL, "output_observation_file", "observed.mix.seq");
  const char* stateout = fx_param_str(NULL, "output_state_file", "state.mix.seq");

  DEBUG_ASSERT_MSG(seqlen <= seqlmax, "lenmax must bigger than length");
  DEBUG_ASSERT_MSG(numseq > 0, "numseq must be positive");

  double step = (double) (seqlmax-seqlen) / numseq;

  MixtureofGaussianHMM hmm;
  hmm.InitFromFile(profile);
  
  TextWriter w_seq, w_state;
  if (!PASSED(w_seq.Open(seqout))) {
    NONFATAL("Couldn't open '%s' for writing.", seqout);
    return SUCCESS_FAIL;
  }

  if (!PASSED(w_state.Open(stateout))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout);
    return SUCCESS_FAIL;
  }

  double L = seqlen;
  for (int i = 0; i < numseq; i++, L+=step) {
    Matrix seq;
    Vector states;
    char s[100];

    hmm.GenerateSequence((int)L, &seq, &states);
    
    sprintf(s, "%% sequence %d", i);
    print_matrix(w_seq, seq, s, "%E,");    
    sprintf(s, "%% state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");    
  }

  //printf("---END---");
  return SUCCESS_PASS;
}

success_t generate_gaussian() {
  const char* profile = fx_param_str_req(NULL, "input_model");
  const int seqlen = fx_param_int(NULL, "length", 10);
  const int seqlmax = fx_param_int(NULL, "lenmax", seqlen);
  const int numseq = fx_param_int(NULL, "numseq", 10);
  const char* seqout = fx_param_str(NULL, "output_observation_file", "observed.gauss.seq");
  const char* stateout = fx_param_str(NULL, "output_state_file", "state.gauss.seq");

  DEBUG_ASSERT_MSG(seqlen <= seqlmax, "lenmax must bigger than length");
  DEBUG_ASSERT_MSG(numseq > 0, "numseq must be positive");

  double step = (double) (seqlmax-seqlen) / numseq;

  GaussianHMM hmm;
  hmm.InitFromFile(profile);
  
  TextWriter w_seq, w_state;
  if (!PASSED(w_seq.Open(seqout))) {
    NONFATAL("Couldn't open '%s' for writing.", seqout);
    return SUCCESS_FAIL;
  }

  if (!PASSED(w_state.Open(stateout))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout);
    return SUCCESS_FAIL;
  }

  double L = seqlen;
  for (int i = 0; i < numseq; i++, L+=step) {
    Matrix seq;
    Vector states;
    char s[100];

    hmm.GenerateSequence((int)L, &seq, &states);
    
    sprintf(s, "%% sequence %d", i);
    print_matrix(w_seq, seq, s, "%E,");    
    sprintf(s, "%% state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");    
  }
  return SUCCESS_PASS;
}

success_t generate_discrete() {
  const char* profile = fx_param_str_req(NULL, "input_model");
  const int seqlen = fx_param_int(NULL, "length", 10);
  const int seqlmax = fx_param_int(NULL, "lenmax", seqlen);
  const int numseq = fx_param_int(NULL, "numseq", 10);
  const char* seqout = fx_param_str(NULL, "output_observation_file", "observed.dis.out");
  const char* stateout = fx_param_str(NULL, "output_state_file", "state.dis.out");

  DEBUG_ASSERT_MSG(seqlen <= seqlmax, "lenmax must bigger than length");
  DEBUG_ASSERT_MSG(numseq > 0, "numseq must be positive");

  double step = (double) (seqlmax-seqlen) / numseq;

  DiscreteHMM hmm;
  hmm.InitFromFile(profile);

  TextWriter w_seq, w_state;
  if (!PASSED(w_seq.Open(seqout))) {
    NONFATAL("Couldn't open '%s' for writing.", seqout);
    return SUCCESS_FAIL;
  }

  if (!PASSED(w_state.Open(stateout))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout);
    return SUCCESS_FAIL;
  }

  double L = seqlen;
  for (int i = 0; i < numseq; i++, L+=step) {
    Vector seq, states;
    char s[100];

    hmm.GenerateSequence((int)L, &seq, &states);
    
    sprintf(s, "%% sequence %d", i);
    print_vector(w_seq, seq, s, "%.0f,");    
    sprintf(s, "%% state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");    
  }
  return SUCCESS_PASS;
}
