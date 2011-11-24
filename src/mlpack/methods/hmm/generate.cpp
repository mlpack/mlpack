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

int main(int argc, char* argv[]) {

  CLI::ParseCommandLine(argc, argv);
  bool s = true;
  if (CLI::HasParam("hmm/type")) {
    const char* type = CLI::GetParam<std::string>("hmm/type").c_str();
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
  Log::Info << "  generate --hmm/type=={discrete|gaussian|mixture} OPTCLINS" << std::endl;
  Log::Info << "[OPTCLINS]" << std::endl;
  Log::Info << "  --hmm/profile=file   : file contains HMM profile" << std::endl;
  Log::Info << "  --hmm/length=NUM     : sequence length" << std::endl;
  Log::Info << "  --hmm/lenmax=NUM     : maximum sequence length, default = length" << std::endl;
  Log::Info << "  --hmm/numseq=NUM     : number of sequence" << std::endl;
  Log::Info << "  --hmm/seqfile=file   : output file for generated sequences" << std::endl;
  Log::Info << "  --hmm/statefile=file : output file for generated state sequences" << std::endl;
}

bool generate_mixture() {
  if (!CLI::HasParam("hmm/profile")) {
    Log::Fatal << "--hmm/profile must be defined." << std::endl;
    return false;
  }
  const char* profile = CLI::GetParam<std::string>("hmm/profile").c_str();
  const int seqlen = CLI::GetParam<int>("hmm/length");
  const int seqlmax = CLI::GetParam<int>("hmm/lenmax");
  const int numseq = CLI::GetParam<int>("hmm/numseq");
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
  if (!CLI::HasParam("hmm/profile")) {
    Log::Fatal << "--hmm/profile must be defined." << std::endl;
    return false;
  }
  const char* profile = CLI::GetParam<std::string>("hmm/profile").c_str();
  const int seqlen = CLI::GetParam<int>("hmm/length");
  const int seqlmax = CLI::GetParam<int>("hmm/lenmax");
  const int numseq = CLI::GetParam<int>("hmm/numseq");
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
  if (!CLI::HasParam("hmm/profile")) {
    Log::Fatal << "--hmm/profile must be defined." << std::endl;
    return false;
  }
  const char* profile = CLI::GetParam<std::string>("hmm/profile").c_str();
  const int seqlen = CLI::GetParam<int>("hmm/length");
  const int seqlmax = CLI::GetParam<int>("hmm/lenmax");
  const int numseq = CLI::GetParam<int>("hmm/numseq");
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
