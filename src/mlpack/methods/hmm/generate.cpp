/**
 * @file generate.cc
 *
 * This file contains the program to generate sequences from a Hidden Markov
 * Model.
 *
 * Usage:
 *   generate --type=TYPE --profile=PROFILE [OPTCLINS]
 * See the usage() function for complete option list
 */
#include <mlpack/core.hpp>

#include "support.hpp"
#include "discreteHMM.hpp"
#include "gaussianHMM.hpp"
#include "mixgaussHMM.hpp"
#include "mixtureDST.hpp"

using namespace mlpack;
using namespace hmm;
using namespace hmm_support;

bool generate_discrete();
bool generate_gaussian();
bool generate_mixture();
void usage();

PARAM_STRING_REQ("type", "HMM type : discrete | gaussian | mixture.", "");
PARAM_STRING_REQ("profile", "A file containing HMM profile.", "");
PARAM_STRING_REQ("seqfile", "Output file for the generated sequences.", "");
PARAM_STRING_REQ("statefile", "Output file for the generated state sequences.",
		"");

PARAM_INT("length", "Sequence length, default = 10.", "", 10);
PARAM_INT("lenmax", "Maximum sequence length, default = 10", "", 10);
PARAM_INT("numseq", "Number of sequance, default = 10.\n", "", 10);

PARAM_MODULE("hmm", "This is a program generating sequences from HMM models.");

int main(int argc, char* argv[]) {

  CLI::ParseCommandLine(argc, argv);
  bool s = true;
  if (CLI::HasParam("type")) {
    const char* type = CLI::GetParam<std::string>("type").c_str();
    if (strcmp(type, "discrete")==0)
      s = generate_discrete();
    else if (strcmp(type, "gaussian")==0)
      s = generate_gaussian();
    else if (strcmp(type, "mixture")==0)
      s = generate_mixture();
    else {
      Log::Fatal << "Unrecognized type: must be: " <<
		"discrete | gaussian | mixture !!!" << std::endl;
      return true;
    }
  }
  else {
    Log::Fatal << "Unrecognized type: must be: " <<
		"discrete | gaussian | mixture  !!!" << std::endl;
    s = false;
  }
  if (!(s)) usage();

}

void usage() {
  Log::Info << std::endl << "Usage:" << std::endl;
  Log::Info << "  generate --type=={discrete|gaussian|mixture} OPTCLINS" << std::endl;
  Log::Info << "[OPTCLINS]" << std::endl;
  Log::Info << "  --profile=file   : file contains HMM profile" << std::endl;
  Log::Info << "  --length=NUM     : sequence length" << std::endl;
  Log::Info << "  --lenmax=NUM     : maximum sequence length, default = length" << std::endl;
  Log::Info << "  --numseq=NUM     : number of sequence" << std::endl;
  Log::Info << "  --seqfile=file   : output file for generated sequences" << std::endl;
  Log::Info << "  --statefile=file : output file for generated state sequences" << std::endl;
}

bool generate_mixture() {
  if (!CLI::HasParam("profile")) {
    Log::Fatal << "--profile must be defined." << std::endl;
    return false;
  }
  const char* profile = CLI::GetParam<std::string>("profile").c_str();
  const int seqlen = CLI::GetParam<int>("length");
  const int seqlmax = CLI::GetParam<int>("lenmax");
  const int numseq = CLI::GetParam<int>("numseq");
  //const char* seqout = CLI::GetParam<std::string>("hmm/seqfile").c_str();
  //const char* stateout = CLI::GetParam<std::string>("hmm/statefile").c_str();

  Log::Assert(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  Log::Assert(numseq > 0, "NUMSEQ must be positive");

  //double step = (double) (seqlmax - seqlen) / numseq;

  MixtureofGaussianHMM hmm;
  hmm.InitFromFile(profile);

  /** need something better
  TextWriter w_seq, w_state;
  if (!(w_seq.Open(seqout))) {
    Log::Warn << "Couldn't open '" << seqout << "' for writing." << std::endl;
    return false;
  }

  if (!(w_state.Open(stateout))) {
    Log::Warn << "Couldn't open '" << stateout << "' for writing." << std::endl;
    return false;
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
  */

  //printf("---END---");
  return true;
}

bool generate_gaussian() {
  if (!CLI::HasParam("profile")) {
    Log::Fatal << "--profile must be defined." << std::endl;
    return false;
  }
  const char* profile = CLI::GetParam<std::string>("profile").c_str();
  const int seqlen = CLI::GetParam<int>("length");
  const int seqlmax = CLI::GetParam<int>("lenmax");
  const int numseq = CLI::GetParam<int>("numseq");
  //const char* seqout = CLI::GetParam<std::string>("hmm/seqfile").c_str();
  //const char* stateout = CLI::GetParam<std::string>("hmm/statefile").c_str();

  Log::Assert(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  Log::Assert(numseq > 0, "NUMSEQ must be positive");

  //double step = (double) (seqlmax - seqlen) / numseq;

  GaussianHMM hmm;
  hmm.InitFromFile(profile);

  /** need something better
  TextWriter w_seq, w_state;
  if (!(w_seq.Open(seqout))) {
    Log::Warn << "Couldn't open '" << seqout << "' for writing." << std::endl;
    return false;
  }

  if (!(w_state.Open(stateout))) {
    Log::Warn << "Couldn't open '" << stateout << "' for writing." << std::endl;
    return false;
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
  */

  return true;
}

bool generate_discrete() {
  if (!CLI::HasParam("profile")) {
    Log::Fatal << "--profile must be defined." << std::endl;
    return false;
  }
  const char* profile = CLI::GetParam<std::string>("profile").c_str();
  const int seqlen = CLI::GetParam<int>("length");
  const int seqlmax = CLI::GetParam<int>("lenmax");
  const int numseq = CLI::GetParam<int>("numseq");
  //const char* seqout = CLI::GetParam<std::string>("hmm/seqfile").c_str();
  //const char* stateout = CLI::GetParam<std::string>("hmm/statefile").c_str();

  Log::Assert(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  Log::Assert(numseq > 0, "NUMSEQ must be positive");

  //double step = (double) (seqlmax - seqlen) / numseq;

  DiscreteHMM hmm;
  hmm.InitFromFile(profile);

  /** need something better
  TextWriter w_seq, w_state;
  if (!(w_seq.Open(seqout))) {
    Log::Warn << "Couldn't open '" << seqout << "' for writing." << std::endl;
    return false;
  }

  if (!(w_state.Open(stateout))) {
    Log::Warn << "Couldn't open '" << stateout << "' for writing." << std::endl;
    return false;
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
  */
  return true;
}
