/**
 * @file train.cc
 *
 * This file contains the program to estimate Hidden Markov Model parameter
 * using training sequences.
 *
 * It use two algorithm: Baum-Welch (EM) and Viterbi
 *
 * Usage:
 *   train --type=TYPE --profile=PROFILE --seqfile=FILE [OPTCLINS]
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

bool train_baumwelch();
bool train_viterbi();
void usage();

/* const fx_entry_doc hmm_train_main_entries[] = {
  {"type", FX_REQUIRED, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"algorithm", FX_PARAM, FX_STR, NULL,
   "  Training algoritm: baumwelch | viterbi.\n"},
  {"seqfile", FX_REQUIRED, FX_STR, NULL,
   "  Output file for the data sequences.\n"},
  {"guess", FX_PARAM, FX_STR, NULL,
   "  File containing guessing HMM model profile.\n"},
  {"numstate", FX_PARAM, FX_INT, NULL,
   "  If no guessing profile specified, at least provide the number of states.\n"},
  {"profile", FX_REQUIRED, FX_STR, NULL,
   "  Output file containing trained HMM profile.\n"},
  {"maxiter", FX_PARAM, FX_INT, NULL,
   "  Maximum number of iterations, default = 500.\n"},
  {"tolerance", FX_PARAM, FX_DOUBLE, NULL,
   "  Error tolerance on log-likelihood as a stopping criteria.\n"},
  FX_ENTRY_DOC_DONE
}; */

PARAM_STRING_REQ("type", "HMM type : discrete | gaussian | mixture.", "hmm");
PARAM_STRING_REQ("profile", "A file containing HMM profile.", "hmm");
PARAM_STRING_REQ("seqfile", "Output file for the generated sequences.", "hmm");
PARAM_STRING("algorithm", "Training algorithm: baumwelch | viterbi.", "hmm",
	"baumwelch");
PARAM_STRING("guess", "File containing guessing HMM model profile.", "hmm",
	"");


PARAM(double, "tolerance",
	"Error tolerance on log-likelihood as a stopping criteria.", "hmm", 1e-3, false);
PARAM_INT("maxiter", "Maximum number of iterations, default = 500.", "hmm", 500);
PARAM_INT("numstate", "If no guessing profile specified, at least provide the number of states.", "hmm", 10);

PARAM_MODULE("hmm", "This is a program generating sequences from HMM models.");

/*
const fx_submodule_doc hmm_train_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
}; */

/*
const fx_module_doc hmm_train_main_doc = {
  hmm_train_main_entries, hmm_train_main_submodules,
  "This is a program training HMM models from data sequences. \n"
};*/

void usage() {
  Log::Warn << "Usage:" << std::endl;
  Log::Warn << "  train --type=={discrete|gaussian|mixture} OPTCLIN" << std::endl;
  Log::Warn << "[OPTCLINS]" << std::endl;
  Log::Warn << "  --algorithm={baumwelch|viterbi} : algorithm used for training, default Baum-Welch" << std::endl;
  Log::Warn << "  --seqfile=file   : file contains input sequences" << std::endl;
  Log::Warn << "  --guess=file     : file contains guess HMM profile" << std::endl;
  Log::Warn << "  --numstate=NUM   : if no guess profile is specified, at least specify the number of state" << std::endl;
  Log::Warn << "  --profile=file   : output file for estimated HMM profile" << std::endl;
  Log::Warn << "  --maxiter=NUM    : maximum number of iteration, default=500" << std::endl;
  Log::Warn << "  --tolerance=NUM  : error tolerance on log-likelihood, default=1e-3" << std::endl;
}

int main(int argc, char* argv[]) {
  CLI::ParseCommandLine(argc, argv);

  bool s = true;
  if (CLI::HasParam("hmm/type")) {
    const char* algorithm = CLI::GetParam<std::string>("hmm/algorithm").c_str();
    if (strcmp(algorithm,"baumwelch") == 0)
      s = train_baumwelch();
    else if (strcmp(algorithm,"viterbi") == 0)
      s = train_viterbi();
    else {
      Log::Fatal << "Unrecognized algorithm: must be baumwelch or viterbi!";
      s = false;
    }
  }
  else {
    Log::Fatal << "Unrecognized type: must be: discrete | gaussian | mixture!";
    s = false;
  }
  if (!(s)) usage();
}

bool train_baumwelch_discrete();
bool train_baumwelch_gaussian();
bool train_baumwelch_mixture();

bool train_baumwelch() {
  const char* type = CLI::GetParam<std::string>("hmm/type").c_str();
  if (strcmp(type, "discrete")==0)
    return train_baumwelch_discrete();
  else if (strcmp(type, "gaussian")==0)
    return train_baumwelch_gaussian();
  else if (strcmp(type, "mixture")==0)
    return train_baumwelch_mixture();
  else {
    printf("Unrecognized type: must be: discrete | gaussian | mixture!\n");
    return false;
  }
}

bool train_viterbi_discrete();
bool train_viterbi_gaussian();
bool train_viterbi_mixture();

bool train_viterbi() {
  const char* type = CLI::GetParam<std::string>("hmm/type").c_str();
  if (strcmp(type, "discrete")==0)
    return train_viterbi_discrete();
  else if (strcmp(type, "gaussian")==0)
    return train_viterbi_gaussian();
  else if (strcmp(type, "mixture")==0)
    return train_viterbi_mixture();
  else {
    printf("Unrecognized type: must be: discrete | gaussian | mixture !!!\n");
    return false;
  }
}

bool train_baumwelch_mixture() {
  if (!CLI::HasParam("hmm/seqfile")) {
    printf("--seqfile must be defined.\n");
    return false;
  }

  MixtureofGaussianHMM hmm;
  std::vector<arma::mat> seqs;

  const char* seqin = CLI::GetParam<std::string>("hmm/seqfile").c_str();
  const char* proout = CLI::GetParam<std::string>("hmm/profile").c_str(); //"pro.mix.out"

  load_matrix_list(seqin, seqs);

  if (CLI::HasParam("hmm/guess")) { // guessed parameters in a file
    const char* guess = CLI::GetParam<std::string>("hmm/guess").c_str();
    Log::Info << "Load parameters from file " << guess << std::endl;
    hmm.InitFromFile(guess);
  }
  else {
    hmm.Init();
    Log::Fatal <<"Automatic initialization not supported !!!";
    return false;
  }

  int maxiter = CLI::GetParam<int>("hmm/maxiter");
  double tol = CLI::GetParam<double>("hmm/tolerance");

  hmm.TrainBaumWelch(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return true;
}

bool train_baumwelch_gaussian() {
  if (!CLI::HasParam("hmm/seqfile")) {
    printf("--seqfile must be defined.\n");
    return false;
  }
  GaussianHMM hmm;
  std::vector<arma::mat> seqs;

  const char* seqin = CLI::GetParam<std::string>("hmm/seqfile").c_str();
  const char* proout = CLI::GetParam<std::string>("hmm/profile").c_str(); //"pro.gauss.out");

  load_matrix_list(seqin, seqs);

  if (CLI::HasParam("hmm/guess")) { // guessed parameters in a file
    const char* guess = CLI::GetParam<std::string>("hmm/guess").c_str();
    Log::Info << "Load parameters from file " << guess << std::endl;
    hmm.InitFromFile(guess);
  }
  else { // otherwise initialized using information from the data
    int numstate = CLI::GetParam<int>("hmm/numstate");
    Log::Info << "Generate HMM parameters: NUMSTATE = " << numstate << std::endl;
    hmm.InitFromData(seqs, numstate);
    Log::Info << "Done." << std::endl;
  }

  int maxiter = CLI::GetParam<int>("hmm/maxiter");
  double tol = CLI::GetParam<double>("hmm/tolerance");

  printf("Training ...\n");
  hmm.TrainBaumWelch(seqs, maxiter, tol);
  printf("Done.\n");

  hmm.SaveProfile(proout);

  return true;
}

bool train_baumwelch_discrete() {
  if (!CLI::HasParam("hmm/seqfile")) {
    printf("--seqfile must be defined.\n");
    return false;
  }

  const char* seqin = CLI::GetParam<std::string>("hmm/seqfile").c_str();
  const char* proout = CLI::GetParam<std::string>("hmm/profile").c_str(); //"pro.dis.out");

  std::vector<arma::vec> seqs;
  load_vector_list(seqin, seqs);

  DiscreteHMM hmm;

  if (CLI::HasParam("hmm/guess")) { // guessed parameters in a file
    const char* guess = CLI::GetParam<std::string>("hmm/guess").c_str();
    Log::Info << "Load HMM parameters from file " << guess << std::endl;
    hmm.InitFromFile(guess);
  }
  else { // otherwise randomly initialized using information from the data
    int numstate = CLI::GetParam<int>("hmm/numstate");
    Log::Info << "Randomly generate parameters: NUMSTATE = " << numstate << std::endl;
    hmm.InitFromData(seqs, numstate);
  }

  int maxiter = CLI::GetParam<int>("hmm/maxiter");
  double tol = CLI::GetParam<double>("hmm/tolerance");

  hmm.TrainBaumWelch(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return true;
}

bool train_viterbi_mixture() {
  if (!CLI::HasParam("hmm/seqfile")) {
    printf("--seqfile must be defined.\n");
    return false;
  }

  MixtureofGaussianHMM hmm;
  std::vector<arma::mat> seqs;

  const char* seqin = CLI::GetParam<std::string>("hmm/seqfile").c_str();
  const char* proout = CLI::GetParam<std::string>("hmm/profile").c_str(); //"pro.mix.out");

  load_matrix_list(seqin, seqs);

  if (CLI::HasParam("hmm/guess")) { // guessed parameters in a file
    const char* guess = CLI::GetParam<std::string>("hmm/guess").c_str();
    Log::Info << "Load parameters from file " << guess << std::endl;
    hmm.InitFromFile(guess);
  } else {
    hmm.Init();
    Log::Info << "Automatic initialization not supported !!!" << std::endl;
    return false;
  }

  int maxiter = CLI::GetParam<int>("hmm/maxiter");
  double tol = CLI::GetParam<double>("hmm/tolerance");

  hmm.TrainViterbi(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return true;
}

bool train_viterbi_gaussian() {
  if (!CLI::HasParam("hmm/seqfile")) {
    Log::Fatal << "--seqfile must be defined." << std::endl;
    return false;
  }

  GaussianHMM hmm;
  std::vector<arma::mat> seqs;

  const char* seqin = CLI::GetParam<std::string>("hmm/seqfile").c_str();
  const char* proout = CLI::GetParam<std::string>("hmm/profile").c_str(); //"pro.gauss.viterbi.out");

  load_matrix_list(seqin, seqs);

  if (CLI::HasParam("hmm/guess")) { // guessed parameters in a file
    const char* guess = CLI::GetParam<std::string>("hmm/guess").c_str();
    Log::Info << "Load parameters from file " << guess << std::endl;
    hmm.InitFromFile(guess);
  }
  else { // otherwise initialized using information from the data
    int numstate = CLI::GetParam<int>("hmm/numstate");
    Log::Info << "Generate parameters: NUMSTATE = " << numstate << std::endl;
    hmm.InitFromData(seqs, numstate);
  }

  int maxiter = CLI::GetParam<int>("hmm/maxiter");
  double tol = CLI::GetParam<double>("hmm/tolerance");

  hmm.TrainViterbi(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return true;
}

bool train_viterbi_discrete() {
  if (!CLI::HasParam("hmm/seqfile")) {
    printf("--seqfile must be defined.\n");
    return false;
  }

  DiscreteHMM hmm;
  std::vector<arma::vec> seqs;

  const char* seqin = CLI::GetParam<std::string>("hmm/seqfile").c_str();
  const char* proout = CLI::GetParam<std::string>("hmm/profile").c_str(); //"pro.dis.viterbi.out");

  load_vector_list(seqin, seqs);

  if (CLI::HasParam("hmm/guess")) { // guessed parameters in a file
    std::vector<arma::mat> matlst;
    const char* guess = CLI::GetParam<std::string>("hmm/guess").c_str();
    Log::Info << "Load parameters from file " << guess << std::endl;
    hmm.InitFromFile(guess);
  }
  else { // otherwise randomly initialized using information from the data
    int numstate = CLI::GetParam<int>("hmm/numstate");
    printf("Generate parameters with NUMSTATE = %d\n", numstate);
    hmm.InitFromData(seqs, numstate);
  }

  int maxiter = CLI::GetParam<int>("hmm/maxiter");
  double tol = CLI::GetParam<double>("hmm/tolerance");

  hmm.TrainViterbi(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return true;
}
