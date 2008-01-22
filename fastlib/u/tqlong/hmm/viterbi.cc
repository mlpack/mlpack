#include "fastlib/fastlib.h"
#include "support.h"
#include "discreteHMM.h"
#include "gaussianHMM.h"
#include "mixgaussHMM.h"
#include "mixtureDST.h"

success_t viterbi_discrete();
success_t viterbi_gaussian();
success_t viterbi_mixture();
void usage();

int main(int argc, char* argv[]) {
  fx_init(argc, argv);
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
      printf("Unrecognized type: must be: discrete | gaussian | mixture !!!\n");
      s = SUCCESS_FAIL;
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
  printf("\n"
	 "Usage:\n"
	 "  viterbi --type=={discrete|gaussian|mixture} OPTIONS\n"
	 "[OPTIONS]\n"
	 "  --profile=file   : file contains HMM profile\n"
	 "  --seqfile=file   : file contains input sequences\n"
	 "  --statefile=file : output file for state sequences\n"
	 );
}

success_t viterbi_mixture() {
  if (!fx_param_exists(NULL, "profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
  const char* profile = fx_param_str_req(NULL, "profile");
  const char* seqin = fx_param_str(NULL, "seqfile", "seq.mix.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.viterbi.mix.out");

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
  if (!fx_param_exists(NULL, "profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
  const char* profile = fx_param_str_req(NULL, "profile");
  const char* seqin = fx_param_str(NULL, "seqfile", "seq.gauss.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.viterbi.gauss.out");
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
  if (!fx_param_exists(NULL, "profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
  const char* profile = fx_param_str_req(NULL, "profile");
  const char* seqin = fx_param_str(NULL, "seqfile", "seq.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.viterbi.out");

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
