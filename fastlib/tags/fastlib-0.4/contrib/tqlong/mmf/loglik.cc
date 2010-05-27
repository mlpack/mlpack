/**
 * @file loglik.cc
 *
 * This file contains the program to compute log-likelihood of sequences
 * according to a Hidden Markov  Model.
 *
 * Usage:
 *   loglik --type=TYPE --profile=PROFILE [OPTIONS]
 * See the usage() function for complete option list
 */

#include "fastlib/fastlib.h"
#include "support.h"
#include "discreteHMM.h"
#include "gaussianHMM.h"
#include "mixgaussHMM.h"
#include "mixtureDST.h"

using namespace hmm_support;

success_t loglik_discrete();
success_t loglik_gaussian();
success_t loglik_mixture();
void usage();

const fx_entry_doc hmm_loglik_main_entries[] = {
  {"type", FX_REQUIRED, FX_STR, NULL,
   "  HMM type : discrete | gaussian | mixture.\n"},
  {"profile", FX_REQUIRED, FX_STR, NULL,
   "  A file containing HMM profile.\n"},
  {"seqfile", FX_PARAM, FX_STR, NULL,
   "  Output file for the data sequences.\n"},
  {"logfile", FX_PARAM, FX_STR, NULL,
   "  Output file for the computed log-likelihood of the sequences.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hmm_loglik_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hmm_loglik_main_doc = {
  hmm_loglik_main_entries, hmm_loglik_main_submodules,
  "This is a program computing log-likelihood of data sequences \n"
  "from HMM models.\n"
};

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &hmm_loglik_main_doc);
  success_t s = SUCCESS_PASS;
  if (fx_param_exists(NULL,"type")) {
    const char* type = fx_param_str_req(NULL, "type");
    if (strcmp(type, "discrete")==0)
      s = loglik_discrete();
    else if (strcmp(type, "gaussian")==0) 
      s = loglik_gaussian();
    else if (strcmp(type, "mixture")==0) 
      s = loglik_mixture();
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
  fx_done(NULL);
}

void usage() {
  printf("\n"
	 "Usage:\n"
	 "  loglik --type=={discrete|gaussian|mixture} OPTIONS\n"
	 "[OPTIONS]\n"
	 "  --profile==file   : file contains HMM profile\n"
	 "  --seqfile==file   : file contains input sequences\n"
	 "  --logfile==file   : output file for log-likelihood of the sequences\n"
	 );
}

success_t loglik_mixture() {
  if (!fx_param_exists(NULL, "profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
  const char* profile = fx_param_str_req(NULL, "profile");
  const char* seqin = fx_param_str(NULL, "seqfile", "seq.mix.out");
  const char* logout = fx_param_str(NULL, "logfile", "log.mix.out");

  MixtureofGaussianHMM hmm;
  hmm.InitFromFile(profile);

  ArrayList<Matrix> seqs;
  load_matrix_list(seqin, &seqs);

  TextWriter w_log;
  if (!PASSED(w_log.Open(logout))) {
    NONFATAL("Couldn't open '%s' for writing.", logout);
    return SUCCESS_FAIL;
  }

  ArrayList<double> list_loglik;
  hmm.ComputeLogLikelihood(seqs, &list_loglik);

  for (int i = 0; i < seqs.size(); i++)
    w_log.Printf("%f\n", list_loglik[i]);
  
  return SUCCESS_PASS;
}

success_t loglik_gaussian() {
  if (!fx_param_exists(NULL, "profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
  const char* profile = fx_param_str_req(NULL, "profile");
  const char* seqin = fx_param_str(NULL, "seqfile", "seq.gauss.out");
  const char* logout = fx_param_str(NULL, "logfile", "log.gauss.out");

  GaussianHMM hmm;
  hmm.InitFromFile(profile);

  ArrayList<Matrix> seqs;
  load_matrix_list(seqin, &seqs);

  TextWriter w_log;
  if (!PASSED(w_log.Open(logout))) {
    NONFATAL("Couldn't open '%s' for writing.", logout);
    return SUCCESS_FAIL;
  }

  ArrayList<double> list_loglik;
  hmm.ComputeLogLikelihood(seqs, &list_loglik);

  for (int i = 0; i < seqs.size(); i++)
    w_log.Printf("%f\n", list_loglik[i]);
  
  return SUCCESS_PASS;
}

success_t loglik_discrete() {
  if (!fx_param_exists(NULL, "profile")) {
    printf("--profile must be defined.\n");
    return SUCCESS_FAIL;
  }
  const char* profile = fx_param_str_req(NULL, "profile");
  const char* seqin = fx_param_str(NULL, "seqfile", "seq.out");
  const char* logout = fx_param_str(NULL, "logfile", "log.out");

  DiscreteHMM hmm;
  hmm.InitFromFile(profile);

  ArrayList<Vector> seqs;
  load_vector_list(seqin, &seqs);

  TextWriter w_log;
  if (!PASSED(w_log.Open(logout))) {
    NONFATAL("Couldn't open '%s' for writing.", logout);
    return SUCCESS_FAIL;
  }

  ArrayList<double> list_loglik;
  hmm.ComputeLogLikelihood(seqs, &list_loglik);

  for (int i = 0; i < seqs.size(); i++)
    w_log.Printf("%f\n", list_loglik[i]);
  return SUCCESS_PASS;
}

