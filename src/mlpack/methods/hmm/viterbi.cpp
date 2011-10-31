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
#include <mlpack/core.h>

#include "support.hpp"
#include "discreteHMM.hpp"
#include "gaussianHMM.hpp"
#include "mixgaussHMM.hpp"
#include "mixtureDST.hpp"

using namespace mlpack;
using namespace hmm;
using namespace hmm_support;

bool viterbi_discrete();
bool viterbi_gaussian();
bool viterbi_mixture();
void usage();

/* const fx_entry_doc hmm_viterbi_main_entries[] = {
  {"type", FX_REQUIRED, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"profile", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM profile.\n"},
  {"seqfile", FX_PARAM, FX_STR, NULL,
   "  Output file for the data sequences.\n"},
  {"statefile", FX_PARAM, FX_STR, NULL,
   "  Output file for the most probable state sequences.\n"},
  FX_ENTRY_DOC_DONE
}; */

PARAM_STRING_REQ("type", "HMM type : discrete | gaussian | mistruxe.", "hmm");
PARAM_STRING_REQ("profile", "A file containing HMM profile", "hmm");
PARAM_STRING_REQ("seqfile", "Output file for the data sequences.", "hmm");
PARAM_STRING_REQ("statefile", "Output file for the most probable state sequences.", "hmm");

PARAM_MODULE("hmm", "This is a program computing th emost probable state\n sequences of data sequences from HMM models.\n");
/* const fx_submodule_doc hmm_viterbi_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
}; */

/* const fx_module_doc hmm_viterbi_main_doc = {
  hmm_viterbi_main_entries, hmm_viterbi_main_submodules,
  "This is a program computing the most probable state sequences \n"
  "of data sequences from HMM models.\n"
}; */

int main(int argc, char* argv[]) {
  CLI::ParseCommandLine(argc, argv);

  bool s = true;
  if (CLI::HasParam("hmm/type")) {
    const char* type = CLI::GetParam<std::string>("hmm/type").c_str();
    if (strcmp(type, "discrete") == 0)
      s = viterbi_discrete();
    else if (strcmp(type, "gaussian") == 0)
      s = viterbi_gaussian();
    else if (strcmp(type, "mixture") == 0)
      s = viterbi_mixture();
    else {
      Log::Warn << "Unrecognized type: must be: discrete | gaussian | mixture!" << std::endl;
      s = false;
    }
  }
  else {
    Log::Warn << "Unrecognized type: must be: discrete | gaussian | mixture!";
    s = false;
  }
  if (!(s)) usage();
}

void usage() {
  Log::Warn << "Usage:" << std::endl;
  Log::Warn << "  viterbi --type=={discrete|gaussian|mixture} OPTCLINS" << std::endl;
  Log::Warn << "[OPTCLINS]" << std::endl;
  Log::Warn << "  --profile=file   : file contains HMM profile" << std::endl;
  Log::Warn << "  --seqfile=file   : file contains input sequences" << std::endl;
  Log::Warn << "  --statefile=file : output file for state sequences" << std::endl;
}

bool viterbi_mixture() {
  if (!CLI::HasParam("hmm/profile")) {
    Log::Fatal << "--profile must be defined." << std::endl;
    return false;
  }
  const char* profile = CLI::GetParam<std::string>("hmm/profile").c_str();
  const char* seqin = CLI::GetParam<std::string>("hmm/seqfile").c_str(); //"seq.mix.out");
  const char* stateout = CLI::GetParam<std::string>("hmm/statefile").c_str(); //"state.viterbi.mix.out");

  MixtureofGaussianHMM hmm;
  hmm.InitFromFile(profile);

  std::vector<arma::mat> seqs;
  load_matrix_list(seqin, seqs);

  TextWriter w_state;
  if (!(w_state.Open(stateout))) {
    Log::Warn << "Couldn't open '" << stateout << "' for writing." << std::endl;
    return false;
  }

  for (size_t i = 0; i < seqs.size(); i++) {
    arma::vec states;
    char s[100];

    hmm.ComputeViterbiStateSequence(seqs[i], states);

    sprintf(s, "%% viterbi state sequence %zu", i);
    print_vector(w_state, states, s, "%.0f,");
  }

  return true;
}

bool viterbi_gaussian() {
  if (!CLI::HasParam("hmm/profile")) {
    Log::Fatal << "--profile must be defined." << std::endl;
    return false;
  }
  const char* profile = CLI::GetParam<std::string>("hmm/profile").c_str();
  const char* seqin = CLI::GetParam<std::string>("hmm/seqfile").c_str(); //"seq.gauss.out");
  const char* stateout = CLI::GetParam<std::string>("hmm/statefile").c_str(); //"state.viterbi.gauss.out");

  GaussianHMM hmm;
  hmm.InitFromFile(profile);

  std::vector<arma::mat> seqs;
  load_matrix_list(seqin, seqs);

  TextWriter w_state;
  if (!(w_state.Open(stateout))) {
    Log::Warn << "Couldn't open '" << stateout << "' for writing." << std::endl;
    return false;
  }

  for (size_t i = 0; i < seqs.size(); i++) {
    arma::vec states;
    char s[100];
    hmm.ComputeViterbiStateSequence(seqs[i], states);

    sprintf(s, "%% viterbi state sequence %zu", i);
    print_vector(w_state, states, s, "%.0f,");
  }

  return true;
}

bool viterbi_discrete() {
  if (!CLI::HasParam("hmm/profile")) {
    Log::Fatal << "--profile must be defined." << std::endl;
    return false;
  }
  const char* profile = CLI::GetParam<std::string>("hmm/profile").c_str();
  const char* seqin = CLI::GetParam<std::string>("hmm/seqfile").c_str(); //"seq.out");
  const char* stateout = CLI::GetParam<std::string>("hmm/statefile").c_str(); //"state.viterbi.out");

  DiscreteHMM hmm;

  hmm.InitFromFile(profile);

  std::vector<arma::vec> seqs;
  load_vector_list(seqin, seqs);

  TextWriter w_state;
  if (!(w_state.Open(stateout))) {
    Log::Warn << "Couldn't open '" << stateout << "' for writing." << std::endl;
    return false;
  }

  for (size_t i = 0; i < seqs.size(); i++) {
    arma::vec states;
    char s[100];

    hmm.ComputeViterbiStateSequence(seqs[i], states);

    sprintf(s, "%% viterbi state sequence %zu", i);
    print_vector(w_state, states, s, "%.0f,");
  }

  return true;
}
