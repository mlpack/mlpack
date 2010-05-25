/**
 * @file loglik.cc
 *
 * This file contains the program to compute log-likelihood of sequences
 * according to a Hidden Markov  Model.
 */

#include <fastlib/fastlib.h>
#include "support.h"
#include "discreteHMM.h"
#include "gaussianHMM.h"
#include "mixgaussHMM.h"
#include "mixtureDST.h"
#include "hmm_documentation.h"

using namespace hmm_support;

success_t loglik_discrete();
success_t loglik_gaussian();
success_t loglik_mixture();
void usage();

const fx_entry_doc hmm_loglik_main_entries[] = {
  {"input_model", FX_REQUIRED, FX_STR, NULL,
   "Input file containing the trained HMM profile.\n"},
  {"input_sequence_file", FX_REQUIRED, FX_STR, NULL,
   "Input file of data sequences whose log-likelihood is to be evaluated.\n"},
  {"output_file", FX_PARAM, FX_STR, NULL,
   "Output file for the computed log-likelihood of each input sequence.\n"
   "     (default output.<type>.lik)\n"},
  {"type", FX_REQUIRED, FX_STR, NULL,
   "HMM type: discrete | gaussian | mixture\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hmm_loglik_main_submodules[] = {
  {"formats", &hmm_format_doc,
   "Documentation for file formats used by this program and other MLPACK HMM tools\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hmm_loglik_main_doc = {
  hmm_loglik_main_entries, hmm_loglik_main_submodules,
  "The hmm_loglik utility calculates the log-likelihood of a set of given\n"
  "observation sequences given a trained HMM profile.  It stores the output of each\n" 
  "calculation in the given output file, where line N in that file corresponds\n"
  "to the Nth input sequence.\n"
  "\n"
  "For more information on the file formats used by the MLPACK HMM utilities, see\n"
  "the documentation in the 'formats' submodule (--help=formats).\n"
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
      FATAL("Unrecognized HMM type; must be 'discrete', 'gaussian', or 'mixture'.\n");
      s = SUCCESS_FAIL;
    }
  }
  else {
    FATAL("Unrecognized HMM type; must be 'discrete', 'gaussian', or 'mixture'.\n");
    s = SUCCESS_FAIL;
  }
  
  fx_done(NULL);
}

success_t loglik_mixture() {
  const char* profile = fx_param_str_req(NULL, "input_model");
  const char* seqin = fx_param_str_req(NULL, "input_sequence_file");
  const char* logout = fx_param_str(NULL, "output_file", "output.mix.lik");

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
  const char* profile = fx_param_str_req(NULL, "input_model");
  const char* seqin = fx_param_str_req(NULL, "input_sequence_file");
  const char* logout = fx_param_str(NULL, "output_file", "output.gauss.lik");

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
  const char* profile = fx_param_str_req(NULL, "input_model");
  const char* seqin = fx_param_str_req(NULL, "input_sequence_file");
  const char* logout = fx_param_str(NULL, "output_file", "output.dis.lik");

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

