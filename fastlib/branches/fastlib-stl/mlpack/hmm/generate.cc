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
#include <fastlib/fx/io.h>
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

/*const fx_entry_doc hmm_generate_main_entries[] = {
  {"type", FX_REQUIRED, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"profile", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM profile.\n"},
  {"length", FX_PARAM, FX_INT, NULL,
   "  Sequence length, default = 10.\n"},
  {"lenmax", FX_PARAM, FX_INT, NULL,
   "  Maximum sequence length, default = length\n"},
  {"numseq", FX_PARAM, FX_INT, NULL,
   "  Number of sequance, default = 10.\n"},
  {"seqfile", FX_PARAM, FX_STR, NULL,
   "  Output file for the generated sequences.\n"},
  {"statefile", FX_PARAM, FX_STR, NULL,
   "  Output file for the generated state sequences.\n"},
  FX_ENTRY_DOC_DONE
}; */

PARAM_STRING_REQ("type", "HMM type : discrete | gaussian | mixture.", "hmm");
PARAM_STRING_REQ("profile", "A file containing HMM profile.", "hmm");
PARAM_STRING_REQ("seqfile", "Output file for the generated sequences.", "hmm");
PARAM_STRING_REQ("statefile", "Output file for the generated state sequences.",
		"hmm");

PARAM_INT("length", "Sequence length, default = 10.", "hmm", 10);
PARAM_INT("lenmax", "Maximum sequence length, default = 10", "hmm", 10);
PARAM_INT("numseq", "Number of sequance, default = 10.\n", "hmm", 10);

PARAM_MODULE("hmm", "This is a program generating sequences from HMM models.");

/* const fx_submodule_doc hmm_generate_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
}; */

/* const fx_module_doc hmm_generate_main_doc = {
  hmm_generate_main_entries, hmm_generate_main_submodules,
  "This is a program generating sequences from HMM models.\n"
}; */

using namespace mlpack;

int main(int argc, char* argv[]) {

  IO::ParseCommandLine(argc, argv);
  success_t s = SUCCESS_PASS;
  if (IO::CheckValue("hmm/type")) {
    const char* type = IO::GetValue<std::string>("hmm/type").c_str();
    if (strcmp(type, "discrete")==0)
      s = generate_discrete();
    else if (strcmp(type, "gaussian")==0) 
      s = generate_gaussian();
    else if (strcmp(type, "mixture")==0)
      s = generate_mixture();
    else {
      IO::Fatal << "Unrecognized type: must be: " << 
		"discrete | gaussian | mixture !!!" << std::endl;
      return SUCCESS_PASS;
    }
  }
  else {
    IO::Fatal << "Unrecognized type: must be: " << 
		"discrete | gaussian | mixture  !!!" << std::endl;
    s = SUCCESS_FAIL;
  }
  if (!PASSED(s)) usage();

}

void usage() {
  IO::Info << std::endl << "Usage:" << std::endl;
  IO::Info << "  generate --hmm/type=={discrete|gaussian|mixture} OPTIONS" << std::endl;
  IO::Info << "[OPTIONS]" << std::endl;
  IO::Info << "  --hmm/profile=file   : file contains HMM profile" << std::endl;
  IO::Info << "  --hmm/length=NUM     : sequence length" << std::endl;
  IO::Info << "  --hmm/lenmax=NUM     : maximum sequence length, default = length" << std::endl;
  IO::Info << "  --hmm/numseq=NUM     : number of sequence" << std::endl;
  IO::Info << "  --hmm/seqfile=file   : output file for generated sequences" << std::endl;
  IO::Info << "  --hmm/statefile=file : output file for generated state sequences" << std::endl;
}

success_t generate_mixture() {
  if (!IO::CheckValue("hmm/profile")) {
    IO::Fatal << "--hmm/profile must be defined." << std::endl;
    return SUCCESS_FAIL;
  }
  const char* profile = IO::GetValue<std::string>("hmm/profile").c_str();
  const int seqlen = IO::GetValue<int>("hmm/length");
  const int seqlmax = IO::GetValue<int>("hmm/lenmax");
  const int numseq = IO::GetValue<int>("hmm/numseq");
  const char* seqout = IO::GetValue<std::string>("hmm/seqfile").c_str();
  const char* stateout = IO::GetValue<std::string>("hmm/statefile").c_str();

  DEBUG_ASSERT_MSG(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  DEBUG_ASSERT_MSG(numseq > 0, "NUMSEQ must be positive");

  double step = (double) (seqlmax - seqlen) / numseq;

  MixtureofGaussianHMM hmm;
  hmm.InitFromFile(profile);
  
  TextWriter w_seq, w_state;
  if (!PASSED(w_seq.Open(seqout))) {
    IO::Warn << "Couldn't open '" << seqout << "' for writing." << std::endl;
    return SUCCESS_FAIL;
  }

  if (!PASSED(w_state.Open(stateout))) {
    IO::Warn << "Couldn't open '" << stateout << "' for writing." << std::endl;
    return SUCCESS_FAIL;
  }

  double L = seqlen;
  for (int i = 0; i < numseq; i++, L += step) {
    arma::mat seq;
    arma::vec states;
    char s[100];

    hmm.GenerateSequence((int)L, seq, states);
    
    sprintf(s, "%% sequence %d", i);
    print_matrix(w_seq, seq, s, "%E,");    
    sprintf(s, "%% state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");    
  }

  //printf("---END---");
  return SUCCESS_PASS;
}

success_t generate_gaussian() {
  if (!IO::CheckValue("hmm/profile")) {
    IO::Fatal << "--hmm/profile must be defined." << std::endl;
    return SUCCESS_FAIL;
  }
  const char* profile = IO::GetValue<std::string>("hmm/profile").c_str();
  const int seqlen = IO::GetValue<int>("hmm/length");
  const int seqlmax = IO::GetValue<int>("hmm/lenmax");
  const int numseq = IO::GetValue<int>("hmm/numseq");
  const char* seqout = IO::GetValue<std::string>("hmm/seqfile").c_str();
  const char* stateout = IO::GetValue<std::string>("hmm/statefile").c_str();

  DEBUG_ASSERT_MSG(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  DEBUG_ASSERT_MSG(numseq > 0, "NUMSEQ must be positive");

  double step = (double) (seqlmax - seqlen) / numseq;

  GaussianHMM hmm;
  hmm.InitFromFile(profile);
  
  TextWriter w_seq, w_state;
  if (!PASSED(w_seq.Open(seqout))) {
    IO::Warn << "Couldn't open '" << seqout << "' for writing." << std::endl;
    return SUCCESS_FAIL;
  }

  if (!PASSED(w_state.Open(stateout))) {
    IO::Warn << "Couldn't open '" << stateout << "' for writing." << std::endl;
    return SUCCESS_FAIL;
  }

  double L = seqlen;
  for (int i = 0; i < numseq; i++, L+=step) {
    arma::mat seq;
    arma::vec states;
    char s[100];

    hmm.GenerateSequence((int) L, seq, states);
    
    sprintf(s, "%% sequence %d", i);
    print_matrix(w_seq, seq, s, "%E,");    
    sprintf(s, "%% state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");    
  }
  return SUCCESS_PASS;
}

success_t generate_discrete() {
  if (!IO::CheckValue("hmm/profile")) {
    IO::Fatal << "--hmm/profile must be defined." << std::endl;
    return SUCCESS_FAIL;
  }
  const char* profile = IO::GetValue<std::string>("hmm/profile").c_str();
  const int seqlen = IO::GetValue<int>("hmm/length");
  const int seqlmax = IO::GetValue<int>("hmm/lenmax");
  const int numseq = IO::GetValue<int>("hmm/numseq");
  const char* seqout = IO::GetValue<std::string>("hmm/seqfile").c_str();
  const char* stateout = IO::GetValue<std::string>("hmm/statefile").c_str();

  DEBUG_ASSERT_MSG(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  DEBUG_ASSERT_MSG(numseq > 0, "NUMSEQ must be positive");

  double step = (double) (seqlmax - seqlen) / numseq;

  DiscreteHMM hmm;
  hmm.InitFromFile(profile);

  TextWriter w_seq, w_state;
  if (!PASSED(w_seq.Open(seqout))) {
    IO::Warn << "Couldn't open '" << seqout << "' for writing." << std::endl;
    return SUCCESS_FAIL;
  }

  if (!PASSED(w_state.Open(stateout))) {
    IO::Warn << "Couldn't open '" << stateout << "' for writing." << std::endl;
    return SUCCESS_FAIL;
  }

  double L = seqlen;
  for (int i = 0; i < numseq; i++, L+=step) {
    arma::vec seq, states;
    char s[100];

    hmm.GenerateSequence((int) L, seq, states);
    
    sprintf(s, "%% sequence %d", i);
    print_vector(w_seq, seq, s, "%.0f,");    
    sprintf(s, "%% state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");    
  }
  return SUCCESS_PASS;
}
