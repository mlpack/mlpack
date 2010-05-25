/**
 * @file train.cc
 *
 * This file contains the program to estimate Hidden Markov Model parameter
 * using training sequences.
 *
 * It use two algorithm: Baum-Welch (EM) and Viterbi
 *
 * Usage:
 *   train --type=TYPE --profile=PROFILE --seqfile=FILE [OPTIONS]
 * See the usage() function for complete option list
 */

#include <fastlib/fastlib.h>
#include "support.h"
#include "discreteHMM.h"
#include "gaussianHMM.h"
#include "mixgaussHMM.h"
#include "mixtureDST.h"
#include "hmm_documentation.h"

using namespace hmm_support;

success_t train_baumwelch();
success_t train_viterbi();
void usage();

const fx_entry_doc hmm_train_main_entries[] = {
  {"input_file", FX_REQUIRED, FX_STR, NULL,
   "Input file containing data sequences to train on.\n"},
  {"guess_file", FX_PARAM, FX_STR, NULL,
   "File containing guess of HMM model profile (.hmm)\n"},
  {"output_file", FX_PARAM, FX_STR, NULL,
   "Output file containing trained HMM profile (default output.<type>.hmm)\n"},
  {"type", FX_REQUIRED, FX_STR, NULL,
   "HMM type: discrete | gaussian | mixture\n"},
  {"algorithm", FX_PARAM, FX_STR, NULL,
   "Training algorithm: baumwelch | viterbi (default baumwelch)\n"},
  {"num_states", FX_PARAM, FX_INT, NULL,
   "If no guess profile is specified, at least provide the number of states\n"},
  {"max_iter", FX_PARAM, FX_INT, NULL,
   "Maximum number of iterations (default 500)\n"},
  {"tolerance", FX_PARAM, FX_DOUBLE, NULL,
   "Error tolerance on log-likelihood as a stopping criteria (default 0.001)\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hmm_train_submod_entries[] = {
  {"formats", &hmm_format_doc,
  "Documentation for file formats used by this program and other MLPACK HMM tools\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hmm_train_main_doc = {
  hmm_train_main_entries, hmm_train_submod_entries,
  "The hmm_train utility is for training HMMs from data sequences. Given an\n"
  "input file of data sequences, and optionally a file containing a guess of\n"
  "the HMM profile (or a parameter specifying the number of states), the program\n"
  "will use the specified training algorithm (Viterbi or Baum-Welch, default\n"
  "Baum-Welch) to train the HMM until the log-likelihood error reaches the given\n"
  "value (default 0.001).\n"
  "\n"
  "It is important to note that if the guess file is not specified, the number\n"
  "of states must be specified.  Additionally, a guess file must be specified\n"
  "if the HMM type is 'mixture'.\n"
  "\n"
  "See the documentation in the 'formats' submodule (--help=formats) for more\n"
  "information on the file formats that this program uses.\n"
};

int main(int argc, char* argv[]) {
  fx_init(argc, argv, &hmm_train_main_doc);
  success_t s = SUCCESS_PASS;
  if (fx_param_exists(NULL, "type")) {
    const char* algorithm = fx_param_str(NULL, "algorithm", "baumwelch");
    if (strcmp(algorithm, "baumwelch") == 0)
      s = train_baumwelch();
    else if (strcmp(algorithm, "viterbi") == 0)
      s = train_viterbi();
    else {
      FATAL("Unrecognized training algorithm: must be 'baumwelch' or 'viterbi'.\n");
      s = SUCCESS_FAIL;
    }
  }
  else {
    FATAL("Unrecognized HMM type; must be 'discrete', 'gaussian', or 'mixture'.\n");
    s = SUCCESS_FAIL;
  }
  
  fx_done(NULL);
}

success_t train_baumwelch_discrete();
success_t train_baumwelch_gaussian();
success_t train_baumwelch_mixture();

success_t train_baumwelch() {
  const char* type = fx_param_str_req(NULL, "type");
  if (strcmp(type, "discrete")==0)
    return train_baumwelch_discrete();
  else if (strcmp(type, "gaussian")==0)
    return train_baumwelch_gaussian();
  else if (strcmp(type, "mixture")==0)
    return train_baumwelch_mixture();

  return SUCCESS_FAIL;
}

success_t train_viterbi_discrete();
success_t train_viterbi_gaussian();
success_t train_viterbi_mixture();

success_t train_viterbi() {
  const char* type = fx_param_str_req(NULL, "type");
  if (strcmp(type, "discrete")==0)
    return train_viterbi_discrete();
  else if (strcmp(type, "gaussian")==0)
    return train_viterbi_gaussian();
  else if (strcmp(type, "mixture")==0)
    return train_viterbi_mixture();
  
  return SUCCESS_FAIL;
}

success_t train_baumwelch_mixture() {
  MixtureofGaussianHMM hmm;
  ArrayList<Matrix> seqs;

  const char* seqin = fx_param_str_req(NULL, "input_file");
  const char* proout = fx_param_str(NULL, "output_file", "output.mix.hmm");

  load_matrix_list(seqin, &seqs);

  if (fx_param_exists(NULL, "guess_file")) { // guessed parameters in a file
    const char* guess = fx_param_str_req(NULL, "guess_file");
    NOTIFY("Loading guess parameters from file %s...", guess);
    hmm.InitFromFile(guess);
  }
  else {
    hmm.Init();
    FATAL("Automatic initialization not supported for mixture training.");
    return SUCCESS_FAIL;
  }

  int maxiter = fx_param_int(NULL, "max_iter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  NOTIFY("Training using Baum-Welch algorithm...");
  hmm.TrainBaumWelch(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return SUCCESS_PASS;
}

success_t train_baumwelch_gaussian() {
  GaussianHMM hmm;
  ArrayList<Matrix> seqs;

  const char* seqin = fx_param_str_req(NULL, "input_file");
  const char* proout = fx_param_str(NULL, "output_file", "output.gauss.hmm");

  load_matrix_list(seqin, &seqs);

  if (fx_param_exists(NULL, "guess_file")) { // guessed parameters in a file
    const char* guess = fx_param_str_req(NULL, "guess_file");
    NOTIFY("Loading guess parameters from file %s...", guess);
    hmm.InitFromFile(guess);
  }
  else { // otherwise initialized using information from the data
    int numstate = fx_param_int_req(NULL, "num_states");
    NOTIFY("Generating HMM parameters: %d states...", numstate);
    hmm.InitFromData(seqs, numstate);
  }

  int maxiter = fx_param_int(NULL, "max_iter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  NOTIFY("Training using Baum-Welch algorithm...");
  hmm.TrainBaumWelch(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return SUCCESS_PASS;
}

success_t train_baumwelch_discrete() {
  const char* seqin = fx_param_str_req(NULL, "input_file");
  const char* proout = fx_param_str(NULL, "output_file", "output.dis.hmm");

  ArrayList<Vector> seqs;
  load_vector_list(seqin, &seqs);

  DiscreteHMM hmm;

  if (fx_param_exists(NULL, "guess_file")) { // guessed parameters in a file
    const char* guess = fx_param_str_req(NULL, "guess_file");
    NOTIFY("Loading HMM parameters from file %s...", guess);
    hmm.InitFromFile(guess);
  }
  else { // otherwise randomly initialized using information from the data
    int numstate = fx_param_int_req(NULL, "num_states");
    NOTIFY("Randomly generating parameters: %d states...", numstate);
    hmm.InitFromData(seqs, numstate);
  }

  int maxiter = fx_param_int(NULL, "max_iter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  NOTIFY("Training using Baum-Welch algorithm...");
  hmm.TrainBaumWelch(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return SUCCESS_PASS;
}

success_t train_viterbi_mixture() {
  MixtureofGaussianHMM hmm;
  ArrayList<Matrix> seqs;

  const char* seqin = fx_param_str_req(NULL, "input_file");
  const char* proout = fx_param_str(NULL, "output_file", "output.mix.hmm");

  load_matrix_list(seqin, &seqs);

  if (fx_param_exists(NULL, "guess_file")) { // guessed parameters in a file
    const char* guess = fx_param_str_req(NULL, "guess_file");
    NOTIFY("Loading parameters from guess file %s...", guess);
    hmm.InitFromFile(guess);
  }
  else {
    hmm.Init();
    FATAL("Automatic initialization not supported for mixture training.");
    return SUCCESS_FAIL;
  }

  int maxiter = fx_param_int(NULL, "max_iter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  NOTIFY("Training using Viterbi algorithm...");
  hmm.TrainViterbi(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return SUCCESS_PASS;
}

success_t train_viterbi_gaussian() {
  GaussianHMM hmm;
  ArrayList<Matrix> seqs;

  const char* seqin = fx_param_str_req(NULL, "input_file");
  const char* proout = fx_param_str(NULL, "output_file", "output.gauss.hmm");

  load_matrix_list(seqin, &seqs);

  if (fx_param_exists(NULL, "guess_file")) { // guessed parameters in a file
    const char* guess = fx_param_str_req(NULL, "guess_file");
    NOTIFY("Loading parameters from file %s...", guess);
    hmm.InitFromFile(guess);
  }
  else { // otherwise initialized using information from the data
    int numstate = fx_param_int_req(NULL, "num_states");
    NOTIFY("Generating parameters with %d states\n", numstate);
    hmm.InitFromData(seqs, numstate);
  }

  int maxiter = fx_param_int(NULL, "max_iter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  NOTIFY("Training using Viterbi algorithm...");
  hmm.TrainViterbi(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return SUCCESS_PASS;
}

success_t train_viterbi_discrete() {
  DiscreteHMM hmm;
  ArrayList<Vector> seqs;

  const char* seqin = fx_param_str_req(NULL, "input_file");
  const char* proout = fx_param_str(NULL, "output_file", "output.dis.hmm");

  load_vector_list(seqin, &seqs);

  if (fx_param_exists(NULL, "guess_file")) { // guessed parameters in a file
    ArrayList<Matrix> matlst;
    const char* guess = fx_param_str_req(NULL, "guess_file");
    NOTIFY("Loading guess parameters from file %s...", guess);
    hmm.InitFromFile(guess);
  }
  else { // otherwise randomly initialized using information from the data
    int numstate = fx_param_int_req(NULL, "num_states");
    NOTIFY("Generate parameters with %d states\n", numstate);
    hmm.InitFromData(seqs, numstate);
  }

  int maxiter = fx_param_int(NULL, "max_iter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  NOTIFY("Training using Viterbi algorithm...");
  hmm.TrainViterbi(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return SUCCESS_PASS;
}
