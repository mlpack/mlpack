/**
 * @file loglik.cc
 *
 * This file contains the program to compute log-likelihood of sequences
 * according to a Hidden Markov  Model.
 *
 * Usage:
 *   loglik --type=TYPE --profile=PROFILE [OPTCLINS]
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

bool loglik_discrete();
bool loglik_gaussian();
bool loglik_mixture();
void usage();

/*const fx_entry_doc hmm_loglik_main_entries[] = {
  {"type", FX_REQUIRED, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"profile", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM profile.\n"},
  {"seqfile", FX_PARAM, FX_STR, NULL,
   "  Output file for the data sequences.\n"},
  {"logfile", FX_PARAM, FX_STR, NULL,
   "  Output file for the computed log-likelihood of the sequences.\n"},
  FX_ENTRY_DOC_DONE
};*/

PARAM_STRING_REQ("type", "HMM type : discrete | gaussian | mixture.", "hmm");
PARAM_STRING_REQ("profile", "A file containing HMM profile.", "hmm");
PARAM_STRING("seqfile", "Outputfile for the datasequences.",
	"hmm", "seq.mix.out");
PARAM_STRING("logfile",
	"Output file for the computed log-likelihood of the sequences.",
	"hmm", "log.mix.out");

PARAM_MODULE("hmm", "This is a program computing log-likelihood of data \nsequences from HMM models.");
/* const fx_submodule_doc hmm_loglik_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
}; */

/* const fx_module_doc hmm_loglik_main_doc = {
  hmm_loglik_main_entries, hmm_loglik_main_submodules,
  "This is a program computing log-likelihood of data sequences \n"
  "from HMM models.\n"
}; */

int main(int argc, char* argv[]) {
  CLI::ParseCommandLine(argc, argv);

  bool s = true;
  if (CLI::HasParam("hmm/type")) {
    const char* type = CLI::GetParam<std::string>("hmm/type").c_str();
    if (strcmp(type, "discrete")==0)
      s = loglik_discrete();
    else if (strcmp(type, "gaussian")==0)
      s = loglik_gaussian();
    else if (strcmp(type, "mixture")==0)
      s = loglik_mixture();
    else {
      Log::Info << "Unrecognized type: must be: discrete | gaussian | mixture !!!";
      s = false;
    }
  }
  else {
    Log::Info << "Unrecognized type: must be: discrete | gaussian | mixture  !!!";
    s = false;
  }
  if (!(s)) usage();
}

void usage() {
  Log::Warn << "\n" << std::endl;
  Log::Warn << "Usage:\n" << std::endl;
  Log::Warn << "  loglik --type=={discrete|gaussian|mixture} OPTCLINS" << std::endl;
  Log::Warn << "[OPTCLINS]" << std::endl;
  Log::Warn << "  --profile==file   : file contains HMM profile" << std::endl;
  Log::Warn << "  --seqfile==file   : file contains input sequences" << std::endl;
  Log::Warn << "  --logfile==file   : output file for log-likelihood of the sequences" << std::endl;
}

bool loglik_mixture() {
  if (!CLI::HasParam("hmm/profile")) {
    Log::Warn << "--profile must be defined." << std::endl;
    return false;
  }
  const char* profile = CLI::GetParam<std::string>("hmm/profile").c_str();
  const char* seqin = CLI::GetParam<std::string>("hmm/seqfile").c_str();
  //const char* logout = CLI::GetParam<std::string>("hmm/logfile").c_str();

  MixtureofGaussianHMM hmm;
  hmm.InitFromFile(profile);

  std::vector<arma::mat> seqs;
  load_matrix_list(seqin, seqs);

  /** need something better
  TextWriter w_log;
  if (!(w_log.Open(logout))) {
    Log::Warn << "Couldn't open '" << logout << "' for writing." << std::endl;
    return false;
  }

  std::vector<double> list_loglik;
  hmm.ComputeLogLikelihood(seqs, list_loglik);

  for (size_t i = 0; i < seqs.size(); i++)
    w_log.Printf("%f\n", list_loglik[i]);
  */

  return true;
}

bool loglik_gaussian() {
  if (!CLI::HasParam("hmm/profile")) {
    Log::Warn << "--profile must be defined." << std::endl;
    return false;
  }
  const char* profile = CLI::GetParam<std::string>("hmm/profile").c_str();
  const char* seqin = CLI::GetParam<std::string>("hmm/seqfile").c_str();
  //const char* logout = CLI::GetParam<std::string>("hmm/logfile").c_str();

  GaussianHMM hmm;
  hmm.InitFromFile(profile);

  std::vector<arma::mat> seqs;
  load_matrix_list(seqin, seqs);

  /** need something better
  TextWriter w_log;
  if (!(w_log.Open(logout))) {
    Log::Warn << "Couldn't open '"<< logout <<"' for writing." << std::endl;
    return false;
  }

  std::vector<double> list_loglik;
  hmm.ComputeLogLikelihood(seqs, list_loglik);

  for (size_t i = 0; i < seqs.size(); i++)
    w_log.Printf("%f\n", list_loglik[i]);
  */

  return true;
}

bool loglik_discrete() {
  if (!CLI::HasParam("hmm/profile")) {
    Log::Warn << "--profile must be defined." << std::endl;
    return false;
  }
  const char* profile = CLI::GetParam<std::string>("hmm/profile").c_str();
  const char* seqin = CLI::GetParam<std::string>("hmm/seqfile").c_str();
  //const char* logout = CLI::GetParam<std::string>("hmm/logfile").c_str();

  DiscreteHMM hmm;
  hmm.InitFromFile(profile);

  std::vector<arma::vec> seqs;
  load_vector_list(seqin, seqs);

  /** need something better
  TextWriter w_log;
  if (!(w_log.Open(logout))) {
    Log::Warn << "Couldn't open '"<< logout <<"' for writing." << std::endl;
    return false;
  }

  std::vector<double> list_loglik;
  hmm.ComputeLogLikelihood(seqs, list_loglik);

  for (size_t i = 0; i < seqs.size(); i++)
    w_log.Printf("%f\n", list_loglik[i]);
  */

  return true;
}
