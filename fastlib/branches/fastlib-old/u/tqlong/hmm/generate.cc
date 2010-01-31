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

#include "fastlib/fastlib.h"
#include "support.h"
#include "discreteHMM.h"
#include "gaussianHMM.h"
#include "mixgaussHMM.h"
#include "mixtureDST.h"

using namespace hmm_support;

success_t generate_discrete();
success_t generate_gaussian();
success_t generate_mixture();
void usage();

int main(int argc, char* argv[]) {
  fx_init(argc, argv);
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
      printf("Unrecognized type: must be: discrete | gaussian | mixture !!!\n");
      return SUCCESS_PASS;
    }
  }
  else {
    printf("Unrecognized type: must be: discrete | gaussian | mixture  !!!\n");
    s = SUCCESS_FAIL;
  }
  if (!PASSED(s)) usage();
  fx_done();
}

void usage() {
  printf("\nUsage:\n");
  printf("  generate --type=={discrete|gaussian|mixture} OPTIONS\n");
  printf("[OPTIONS]\n");
  printf("  --profile=file   : file contains HMM profile\n");
  printf("  --length=NUM     : sequence length\n");
  printf("  --lenmax=NUM     : maximum sequence length, default = length\n");
  printf("  --numseq=NUM     : number of sequence\n");
  printf("  --seqfile=file   : output file for generated sequences\n");
  printf("  --statefile=file : output file for generated state sequences\n");
}

success_t generate_mixture() {
  if (!fx_param_exists(NULL, "profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
  const char* profile = fx_param_str_req(NULL, "profile");
  const int seqlen = fx_param_int(NULL, "length", 10);
  const int seqlmax = fx_param_int(NULL, "lenmax", seqlen);
  const int numseq = fx_param_int(NULL, "numseq", 10);
  const char* seqout = fx_param_str(NULL, "seqfile", "seq.mix.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.mix.out");

  DEBUG_ASSERT_MSG(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  DEBUG_ASSERT_MSG(numseq > 0, "NUMSEQ must be positive");

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
  if (!fx_param_exists(NULL, "profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
  const char* profile = fx_param_str_req(NULL, "profile");
  const int seqlen = fx_param_int(NULL, "length", 10);
  const int seqlmax = fx_param_int(NULL, "lenmax", seqlen);
  const int numseq = fx_param_int(NULL, "numseq", 10);
  const char* seqout = fx_param_str(NULL, "seqfile", "seq.gauss.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.gauss.out");

  DEBUG_ASSERT_MSG(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  DEBUG_ASSERT_MSG(numseq > 0, "NUMSEQ must be positive");

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
  if (!fx_param_exists(NULL, "profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
  const char* profile = fx_param_str_req(NULL, "profile");
  const int seqlen = fx_param_int(NULL, "length", 10);
  const int seqlmax = fx_param_int(NULL, "lenmax", seqlen);
  const int numseq = fx_param_int(NULL, "numseq", 10);
  const char* seqout = fx_param_str(NULL, "seqfile", "seq.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.out");

  DEBUG_ASSERT_MSG(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  DEBUG_ASSERT_MSG(numseq > 0, "NUMSEQ must be positive");

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
