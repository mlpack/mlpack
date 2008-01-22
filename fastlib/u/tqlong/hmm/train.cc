#include "fastlib/fastlib.h"
#include "support.h"
#include "discreteHMM.h"
#include "gaussianHMM.h"
#include "mixgaussHMM.h"
#include "mixtureDST.h"

success_t train_baumwelch();
success_t train_viterbi();
void usage();

int main(int argc, char* argv[]) {
  fx_init(argc, argv);
  success_t s = SUCCESS_PASS;
  if (fx_param_exists(NULL,"type")) {
    const char* algorithm = fx_param_str(NULL, "algorithm", "baumwelch");
    if (strcmp(algorithm,"baumwelch")==0)
      s = train_baumwelch();
    else if (strcmp(algorithm,"viterbi")==0)
      s = train_viterbi();
    else {
      printf("Unrecognized algorithm: must be baumwelch or viterbi !!!\n");
      s = SUCCESS_FAIL;
    }
  }
  else {
    printf("Unrecognized type: must be: discrete | gaussian | mixture  !!!\n");
    s = SUCCESS_FAIL;
  }
  if (!PASSED(s)) usage();
  fx_done();
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
  else {
    printf("Unrecognized type: must be: discrete | gaussian | mixture !!!\n");
    return SUCCESS_FAIL;
  }
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
  else {
    printf("Unrecognized type: must be: discrete | gaussian | mixture !!!\n");
    return SUCCESS_FAIL;
  }
}

void usage() {
  printf("\nUsage:\n"
	 "  train --type=={discrete|gaussian|mixture} OPTION\n"
	 "[OPTIONS]\n"
	 "  --algorithm={baumwelch|viterbi} : algorithm used for training, default Baum-Welch\n"
	 "  --seqfile=file   : file contains input sequences\n"
	 "  --guess=file     : file contains guess HMM profile\n"
	 "  --numstate=NUM   : if no guess profile is specified, at least specify the number of state\n"
	 "  --profile=file   : output file for estimated HMM profile\n"
	 "  --maxiter=NUM    : maximum number of iteration, default=500\n"
	 "  --tolerance=NUM  : error tolerance on log-likelihood, default=1e-3\n"
	 );
}

success_t train_baumwelch_mixture() {
  if (!fx_param_exists(NULL, "seqfile")) {
    printf("--seqfile must be defined.\n");
    return SUCCESS_FAIL;
  }

  MixtureofGaussianHMM hmm;
  ArrayList<Matrix> seqs;

  const char* seqin = fx_param_str_req(NULL, "seqfile");
  const char* proout = fx_param_str(NULL, "profile", "pro.mix.out");

  load_matrix_list(seqin, &seqs);

  if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
    const char* guess = fx_param_str_req(NULL, "guess");
    printf("Load parameters from file %s\n", guess);
    hmm.InitFromFile(guess);
  }
  else {
    hmm.Init();
    printf("Automatic initialization not supported !!!");
    return SUCCESS_FAIL;
  }

  int maxiter = fx_param_int(NULL, "maxiter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  hmm.TrainBaumWelch(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return SUCCESS_PASS;
}

success_t train_baumwelch_gaussian() {
  if (!fx_param_exists(NULL, "seqfile")) {
    printf("--seqfile must be defined.\n");
    return SUCCESS_FAIL;
  }
  GaussianHMM hmm;
  ArrayList<Matrix> seqs;

  const char* seqin = fx_param_str_req(NULL, "seqfile");
  const char* proout = fx_param_str(NULL, "profile", "pro.gauss.out");

  load_matrix_list(seqin, &seqs);

  if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
    const char* guess = fx_param_str_req(NULL, "guess");
    printf("Load parameters from file %s\n", guess);
    hmm.InitFromFile(guess);
  }
  else { // otherwise initialized using information from the data
    int numstate = fx_param_int_req(NULL, "numstate");
    printf("Generate HMM parameters: NUMSTATE = %d\n", numstate);
    hmm.InitFromData(seqs, numstate);
    printf("Done.\n");
  }

  int maxiter = fx_param_int(NULL, "maxiter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  printf("Training ...\n");
  hmm.TrainBaumWelch(seqs, maxiter, tol);
  printf("Done.\n");

  hmm.SaveProfile(proout);

  return SUCCESS_PASS;
}

success_t train_baumwelch_discrete() {
  if (!fx_param_exists(NULL, "seqfile")) {
    printf("--seqfile must be defined.\n");
    return SUCCESS_FAIL;
  }

  const char* seqin = fx_param_str_req(NULL, "seqfile");
  const char* proout = fx_param_str(NULL, "profile", "pro.dis.out");

  ArrayList<Vector> seqs;
  load_vector_list(seqin, &seqs);

  DiscreteHMM hmm;

  if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
    const char* guess = fx_param_str_req(NULL, "guess");
    printf("Load HMM parameters from file %s\n", guess);
    hmm.InitFromFile(guess);
  }
  else { // otherwise randomly initialized using information from the data
    int numstate = fx_param_int_req(NULL, "numstate");
    printf("Randomly generate parameters: NUMSTATE = %d\n", numstate);
    hmm.InitFromData(seqs, numstate);
  }

  int maxiter = fx_param_int(NULL, "maxiter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  hmm.TrainBaumWelch(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return SUCCESS_PASS;
}

success_t train_viterbi_mixture() {
  if (!fx_param_exists(NULL, "seqfile")) {
    printf("--seqfile must be defined.\n");
    return SUCCESS_FAIL;
  }
  
  MixtureofGaussianHMM hmm;
  ArrayList<Matrix> seqs;

  const char* seqin = fx_param_str_req(NULL, "seqfile");
  const char* proout = fx_param_str(NULL, "profile", "pro.mix.out");

  load_matrix_list(seqin, &seqs);

  if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
    const char* guess = fx_param_str_req(NULL, "guess");
    printf("Load parameters from file %s\n", guess);
    hmm.InitFromFile(guess);
  }
  else {
    hmm.Init();
    printf("Automatic initialization not supported !!!");
    return SUCCESS_FAIL;
  }

  int maxiter = fx_param_int(NULL, "maxiter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  hmm.TrainViterbi(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return SUCCESS_PASS;
}

success_t train_viterbi_gaussian() {
  if (!fx_param_exists(NULL, "seqfile")) {
    printf("--seqfile must be defined.\n");
    return SUCCESS_FAIL;
  }
  
  GaussianHMM hmm;
  ArrayList<Matrix> seqs;

  const char* seqin = fx_param_str_req(NULL, "seqfile");
  const char* proout = fx_param_str(NULL, "profile", "pro.gauss.viterbi.out");

  load_matrix_list(seqin, &seqs);

  if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
    const char* guess = fx_param_str_req(NULL, "guess");
    printf("Load parameters from file %s\n", guess);
    hmm.InitFromFile(guess);
  }
  else { // otherwise initialized using information from the data
    int numstate = fx_param_int_req(NULL, "numstate");
    printf("Generate parameters: NUMSTATE = %d\n", numstate);
    hmm.InitFromData(seqs, numstate);
  }

  int maxiter = fx_param_int(NULL, "maxiter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  hmm.TrainViterbi(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return SUCCESS_PASS;
}

success_t train_viterbi_discrete() {
  if (!fx_param_exists(NULL, "seqfile")) {
    printf("--seqfile must be defined.\n");
    return SUCCESS_FAIL;
  }

  DiscreteHMM hmm;
  ArrayList<Vector> seqs;

  const char* seqin = fx_param_str_req(NULL, "seqfile");
  const char* proout = fx_param_str(NULL, "profile", "pro.dis.viterbi.out");

  load_vector_list(seqin, &seqs);

  if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
    ArrayList<Matrix> matlst;
    const char* guess = fx_param_str_req(NULL, "guess");
    printf("Load parameters from file %s\n", guess);
    hmm.InitFromFile(guess);
  }
  else { // otherwise randomly initialized using information from the data
    int numstate = fx_param_int_req(NULL, "numstate");
    printf("Generate parameters with NUMSTATE = %d\n", numstate);
    hmm.InitFromData(seqs, numstate);
  }

  int maxiter = fx_param_int(NULL, "maxiter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  hmm.TrainViterbi(seqs, maxiter, tol);

  hmm.SaveProfile(proout);

  return SUCCESS_PASS;
}
