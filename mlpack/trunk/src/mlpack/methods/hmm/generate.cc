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
#include <mlpack_core.h>

#include "support.h"
#include "discreteHMM.h"
#include "gaussianHMM.h"
#include "mixgaussHMM.h"
#include "mixtureDST.h"

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

using namespace mlpack;

int main(int argc, char* argv[]) {

  IO::ParseCommandLine(argc, argv);
  bool s = true;
  if (IO::HasParam("hmm/type")) {
    const char* type = IO::GetParam<std::string>("hmm/type").c_str();
    if (strcmp(type, "discrete")==0)
      s = generate_discrete();
    else if (strcmp(type, "gaussian")==0)
      s = generate_gaussian();
    else if (strcmp(type, "mixture")==0)
      s = generate_mixture();
    else {
      IO::Fatal << "Unrecognized type: must be: " <<
		"discrete | gaussian | mixture !!!" << std::endl;
      return true;
    }
  }
  else {
    IO::Fatal << "Unrecognized type: must be: " <<
		"discrete | gaussian | mixture  !!!" << std::endl;
    s = false;
  }
  if (!(s)) usage();

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

bool generate_mixture() {
  if (!IO::HasParam("hmm/profile")) {
    IO::Fatal << "--hmm/profile must be defined." << std::endl;
    return false;
  }
  const char* profile = IO::GetParam<std::string>("hmm/profile").c_str();
  const int seqlen = IO::GetParam<int>("hmm/length");
  const int seqlmax = IO::GetParam<int>("hmm/lenmax");
  const int numseq = IO::GetParam<int>("hmm/numseq");
  const char* seqout = IO::GetParam<std::string>("hmm/seqfile").c_str();
  const char* stateout = IO::GetParam<std::string>("hmm/statefile").c_str();

  IO::AssertMessage(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  IO::AssertMessage(numseq > 0, "NUMSEQ must be positive");

  double step = (double) (seqlmax - seqlen) / numseq;

  MixtureofGaussianHMM hmm;
  hmm.InitFromFile(profile);

  TextWriter w_seq, w_state;
  if (!(w_seq.Open(seqout))) {
    IO::Warn << "Couldn't open '" << seqout << "' for writing." << std::endl;
    return false;
  }

  if (!(w_state.Open(stateout))) {
    IO::Warn << "Couldn't open '" << stateout << "' for writing." << std::endl;
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

  //printf("---END---");
  return true;
}

bool generate_gaussian() {
  if (!IO::HasParam("hmm/profile")) {
    IO::Fatal << "--hmm/profile must be defined." << std::endl;
    return false;
  }
  const char* profile = IO::GetParam<std::string>("hmm/profile").c_str();
  const int seqlen = IO::GetParam<int>("hmm/length");
  const int seqlmax = IO::GetParam<int>("hmm/lenmax");
  const int numseq = IO::GetParam<int>("hmm/numseq");
  const char* seqout = IO::GetParam<std::string>("hmm/seqfile").c_str();
  const char* stateout = IO::GetParam<std::string>("hmm/statefile").c_str();

  IO::AssertMessage(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  IO::AssertMessage(numseq > 0, "NUMSEQ must be positive");

  double step = (double) (seqlmax - seqlen) / numseq;

  GaussianHMM hmm;
  hmm.InitFromFile(profile);

  TextWriter w_seq, w_state;
  if (!(w_seq.Open(seqout))) {
    IO::Warn << "Couldn't open '" << seqout << "' for writing." << std::endl;
    return false;
  }

  if (!(w_state.Open(stateout))) {
    IO::Warn << "Couldn't open '" << stateout << "' for writing." << std::endl;
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
  return true;
}

bool generate_discrete() {
  if (!IO::HasParam("hmm/profile")) {
    IO::Fatal << "--hmm/profile must be defined." << std::endl;
    return false;
  }
  const char* profile = IO::GetParam<std::string>("hmm/profile").c_str();
  const int seqlen = IO::GetParam<int>("hmm/length");
  const int seqlmax = IO::GetParam<int>("hmm/lenmax");
  const int numseq = IO::GetParam<int>("hmm/numseq");
  const char* seqout = IO::GetParam<std::string>("hmm/seqfile").c_str();
  const char* stateout = IO::GetParam<std::string>("hmm/statefile").c_str();

  IO::AssertMessage(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  IO::AssertMessage(numseq > 0, "NUMSEQ must be positive");

  double step = (double) (seqlmax - seqlen) / numseq;

  DiscreteHMM hmm;
  hmm.InitFromFile(profile);

  TextWriter w_seq, w_state;
  if (!(w_seq.Open(seqout))) {
    IO::Warn << "Couldn't open '" << seqout << "' for writing." << std::endl;
    return false;
  }

  if (!(w_state.Open(stateout))) {
    IO::Warn << "Couldn't open '" << stateout << "' for writing." << std::endl;
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
  return true;
}
