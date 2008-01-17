#include "fastlib/fastlib.h"
#include "support.h"
#include "discreteHMM.h"
#include "gaussianHMM.h"
#include "mixgaussHMM.h"
#include "mixtureDST.h"

success_t generate_mode();
success_t loglik_mode();
success_t viterbi_mode();
success_t train_mode();

int main(int argc, char* argv[]) {
  fx_init(argc, argv);
  const char* mode = fx_param_str_req(NULL, "mode");
  if (strcmp(mode, "generate") == 0) 
    generate_mode();
  else if (strcmp(mode, "loglik") == 0)
    loglik_mode();
  else if (strcmp(mode, "viterbi") == 0) 
    viterbi_mode();
  else if (strcmp(mode, "train") == 0) 
    train_mode();
  else     
    printf("Unrecognized mode: must be: generate | loglik | viterbi | train !!!\n");
  fx_done();
}

success_t train_baumwelch();
success_t train_viterbi();

success_t train_mode() {
  const char* algorithm = fx_param_str(NULL, "algorithm", "baumwelch");
  if (strcmp(algorithm,"baumwelch")==0)
    return train_baumwelch();
  else if (strcmp(algorithm,"viterbi")==0)
    return train_viterbi();
  else {
    printf("Unrecognized algorithm: must be baumwelch or viterbi !!!\n");
    return SUCCESS_PASS;
  }
}

success_t viterbi_discrete();
success_t viterbi_gaussian();
success_t viterbi_mixture();

success_t viterbi_mode() {
  const char* type = fx_param_str_req(NULL, "type");
  if (strcmp(type, "discrete")==0)
    return viterbi_discrete();
  else if (strcmp(type, "gaussian")==0)
    return viterbi_gaussian();
  else if (strcmp(type, "mixture")==0)
    return viterbi_mixture();
  else {
    printf("Unrecognized type: must be: discrete | gaussian | mixture !!!\n");
    return SUCCESS_PASS;
  }
}

success_t viterbi_mixture() {
  const char* profile = fx_param_str_req(NULL, "profile");
  const char* seqin = fx_param_str(NULL, "seqfile", "seq.mix.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.viterbi.mix.out");

  Matrix trans;
  ArrayList<MixtureGauss> mixs;
  load_profileM(profile, &trans, &mixs);

  ArrayList<Matrix> seqs;
  load_matrix_list(seqin, &seqs);

  TextWriter w_state;
  if (!PASSED(w_state.Open(stateout))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout);
    return SUCCESS_FAIL;
  }

  int M = trans.n_rows();

  for (int i = 0; i < seqs.size(); i++) {
    Vector states;
    char s[100];
    int L = seqs[i].n_cols();
    Matrix emis_prob;
    emis_prob.Init(M, L);
    hmm_cal_emis_probM(seqs[i], mixs, &emis_prob);
    hmm_viterbiG_init(trans, emis_prob, &states);
    
    sprintf(s, "%% viterbi state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");
  }
  return SUCCESS_PASS;
}

success_t viterbi_gaussian() {
  const char* profile = fx_param_str_req(NULL, "profile");
  const char* seqin = fx_param_str(NULL, "seqfile", "seq.gauss.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.viterbi.gauss.out");

  Matrix trans;
  ArrayList<Vector> means;
  ArrayList<Matrix> covs, inv_covs;
  Vector det;
  load_profileG(profile, &trans, &means, &covs);

  ArrayList<Matrix> seqs;
  load_matrix_list(seqin, &seqs);

  TextWriter w_state;
  if (!PASSED(w_state.Open(stateout))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout);
    return SUCCESS_FAIL;
  }

  int M = trans.n_rows();
  int N = means[0].length();
  inv_covs.Init();
  det.Init(M);
  for (int i = 0; i < M; i++) {
    Matrix inv;
    la::InverseInit(covs[i], &inv);
    inv_covs.AddBackItem(inv);
    det[i] = pow(2.0*math::PI, -N/2.0) * pow(la::Determinant(covs[i]), -0.5);
  }

  for (int i = 0; i < seqs.size(); i++) {
    Vector states;
    char s[100];
    int L = seqs[i].n_cols();
    Matrix emis_prob;
    emis_prob.Init(M, L);
    hmm_cal_emis_prob(seqs[i], means, inv_covs, det, &emis_prob);
    hmm_viterbiG_init(trans, emis_prob, &states);
    
    sprintf(s, "%% viterbi state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");
  }
  return SUCCESS_PASS;
}

success_t viterbi_discrete() {
  const char* profile = fx_param_str_req(NULL, "profile");
  const char* seqin = fx_param_str(NULL, "seqfile", "seq.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state_viterbi.out");

  ArrayList<Matrix> matlst;
  load_matrix_list(profile, &matlst);

  ArrayList<Vector> seqs;
  load_vector_list(seqin, &seqs);

  TextWriter w_state;
  if (!PASSED(w_state.Open(stateout))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout);
    return SUCCESS_FAIL;
  }

  Matrix trans, emis;
  trans.Alias(matlst[0]);
  emis.Alias(matlst[1]);
  for (int i = 0; i < seqs.size(); i++) {
    Vector states;
    char s[100];
    
    hmm_viterbiD_init(seqs[i], trans, emis, &states);
    
    sprintf(s, "%% viterbi state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");
  }
  return SUCCESS_PASS;
}

success_t loglik_discrete();
success_t loglik_gaussian();
success_t loglik_mixture();

success_t loglik_mode() {
  const char* type = fx_param_str_req(NULL, "type");
  if (strcmp(type, "discrete")==0)
    return loglik_discrete();
  else if (strcmp(type, "gaussian")==0) 
    return loglik_gaussian();
  else if (strcmp(type, "mixture")==0) 
    return loglik_mixture();
  else {
    printf("Unrecognized type: must be: discrete | gaussian | mixture !!!\n");
    return SUCCESS_PASS;
  }
}

success_t loglik_mixture() {
  const char* profile = fx_param_str_req(NULL, "profile");
  const char* seqin = fx_param_str(NULL, "seqfile", "seq.mix.out");
  const char* logout = fx_param_str(NULL, "logfile", "log.mix.out");

  Matrix trans;
  ArrayList<MixtureGauss> mixs;
  load_profileM(profile, &trans, &mixs);

  ArrayList<Matrix> seqs;
  load_matrix_list(seqin, &seqs);

  TextWriter w_log;
  if (!PASSED(w_log.Open(logout))) {
    NONFATAL("Couldn't open '%s' for writing.", logout);
    return SUCCESS_FAIL;
  }

  int M = trans.n_rows();
  for (int i = 0; i < seqs.size(); i++) {
    int L = seqs[i].n_cols();
    Matrix ps, fs, bs, emis_prob;
    Vector sc;
    emis_prob.Init(M, L);
    ps.Init(M, L);
    fs.Init(M, L);
    bs.Init(M, L);
    sc.Init(L);
    hmm_cal_emis_probM(seqs[i], mixs, &emis_prob);
    double loglik = hmm_decodeG(trans, emis_prob, &ps, &fs, &bs, &sc);
    w_log.Printf("%% Loglik sequence %d\n %f\n", i, loglik);
  }

  return SUCCESS_PASS;
}

success_t loglik_gaussian() {
  const char* profile = fx_param_str_req(NULL, "profile");
  const char* seqin = fx_param_str(NULL, "seqfile", "seq.gauss.out");
  const char* logout = fx_param_str(NULL, "logfile", "log.gauss.out");

  Matrix trans;
  ArrayList<Vector> means;
  ArrayList<Matrix> covs, inv_covs;
  Vector det;
  load_profileG(profile, &trans, &means, &covs);

  ArrayList<Matrix> seqs;
  load_matrix_list(seqin, &seqs);

  TextWriter w_log;
  if (!PASSED(w_log.Open(logout))) {
    NONFATAL("Couldn't open '%s' for writing.", logout);
    return SUCCESS_FAIL;
  }

  int M = trans.n_rows();
  int N = means[0].length();
  inv_covs.Init();
  det.Init(M);
  for (int i = 0; i < M; i++) {
    Matrix inv;
    la::InverseInit(covs[i], &inv);
    inv_covs.AddBackItem(inv);
    det[i] = pow(2.0*math::PI, -N/2.0) * pow(la::Determinant(covs[i]), -0.5);
  }

  for (int i = 0; i < seqs.size(); i++) {
    int L = seqs[i].n_cols();
    Matrix ps, fs, bs, emis_prob;
    Vector sc;
    emis_prob.Init(M, L);
    ps.Init(M, L);
    fs.Init(M, L);
    bs.Init(M, L);
    sc.Init(L);
    hmm_cal_emis_prob(seqs[i], means, inv_covs, det, &emis_prob);
    double loglik = hmm_decodeG(trans, emis_prob, &ps, &fs, &bs, &sc);
    w_log.Printf("%% Loglik sequence %d\n %f\n", i, loglik);
  }

  return SUCCESS_PASS;
}

success_t loglik_discrete() {
  const char* profile = fx_param_str_req(NULL, "profile");
  const char* seqin = fx_param_str(NULL, "seqfile", "seq.out");
  const char* logout = fx_param_str(NULL, "logfile", "log.out");

  ArrayList<Matrix> matlst;
  load_matrix_list(profile, &matlst);

  ArrayList<Vector> seqs;
  load_vector_list(seqin, &seqs);

  TextWriter w_log;
  if (!PASSED(w_log.Open(logout))) {
    NONFATAL("Couldn't open '%s' for writing.", logout);
    return SUCCESS_FAIL;
  }

  Matrix trans, emis;
  trans.Alias(matlst[0]);
  emis.Alias(matlst[1]);
  for (int i = 0; i < seqs.size(); i++) {
    int L = seqs[i].length();
    int M = trans.n_rows();
    Matrix ps, fs, bs;
    Vector sc;
    ps.Init(M, L);
    fs.Init(M, L);
    bs.Init(M, L);
    sc.Init(L);
    double loglik = hmm_decodeD(seqs[i], trans, emis, &ps, &fs, &bs, &sc);
    w_log.Printf("%% Loglik sequence %d\n %f\n", i, loglik);
  }
  return SUCCESS_PASS;
}

success_t generate_discrete();
success_t generate_gaussian();
success_t generate_mixture();

success_t generate_mode() {
  const char* type = fx_param_str_req(NULL, "type");
  if (strcmp(type, "discrete")==0)
    return generate_discrete();
  else if (strcmp(type, "gaussian")==0) 
    return generate_gaussian();
  else if (strcmp(type, "mixture")==0) 
    return generate_mixture();
  else {
    printf("Unrecognized type: must be: discrete | gaussian | mixture !!!\n");
    return SUCCESS_PASS;
  }
}

success_t generate_mixture() {
  const char* profile = fx_param_str_req(NULL, "profile");
  const int seqlen = fx_param_int(NULL, "length", 10);
  const int seqlmax = fx_param_int(NULL, "lenmax", seqlen);
  const int numseq = fx_param_int(NULL, "numseq", 10);
  const char* seqout = fx_param_str(NULL, "seqfile", "seq.mix.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.mix.out");

  DEBUG_ASSERT_MSG(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  DEBUG_ASSERT_MSG(numseq > 0, "NUMSEQ must be positive");

  double step = (double) (seqlmax-seqlen) / numseq;

  Matrix trans;
  ArrayList<MixtureGauss> mixs;
  load_profileM(profile, &trans, &mixs);
  
  TextWriter w_seq, w_state;
  if (!PASSED(w_seq.Open(seqout))) {
    NONFATAL("Couldn't open '%s' for writing.", seqout);
    return SUCCESS_FAIL;
  }

  if (!PASSED(w_state.Open(stateout))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout);
    return SUCCESS_FAIL;
  }

  double L = seqlen;
  for (int i = 0; i < numseq; i++, L+=step) {
    Matrix seq;
    Vector states;
    char s[100];

    hmm_generateM_init((int)L, trans, mixs, &seq, &states);
    
    sprintf(s, "%% sequence %d", i);
    print_matrix(w_seq, seq, s, "%E,");    
    sprintf(s, "%% state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");    
  }
  //printf("---END---");
  return SUCCESS_PASS;
}

success_t generate_gaussian() {
  const char* profile = fx_param_str_req(NULL, "profile");
  const int seqlen = fx_param_int(NULL, "length", 10);
  const int seqlmax = fx_param_int(NULL, "lenmax", seqlen);
  const int numseq = fx_param_int(NULL, "numseq", 10);
  const char* seqout = fx_param_str(NULL, "seqfile", "seq.gauss.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.gauss.out");

  DEBUG_ASSERT_MSG(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  DEBUG_ASSERT_MSG(numseq > 0, "NUMSEQ must be positive");

  double step = (double) (seqlmax-seqlen) / numseq;

  Matrix trans;
  ArrayList<Vector> means;
  ArrayList<Matrix> covs;
  load_profileG(profile, &trans, &means, &covs);
  
  TextWriter w_seq, w_state;
  if (!PASSED(w_seq.Open(seqout))) {
    NONFATAL("Couldn't open '%s' for writing.", seqout);
    return SUCCESS_FAIL;
  }

  if (!PASSED(w_state.Open(stateout))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout);
    return SUCCESS_FAIL;
  }

  double L = seqlen;
  for (int i = 0; i < numseq; i++, L+=step) {
    Matrix seq;
    Vector states;
    char s[100];

    hmm_generateG_init((int)L, trans, means, covs, &seq, &states);
    
    sprintf(s, "%% sequence %d", i);
    print_matrix(w_seq, seq, s, "%E,");    
    sprintf(s, "%% state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");    
  }
  return SUCCESS_PASS;
}

success_t generate_discrete() {
  const char* profile = fx_param_str_req(NULL, "profile");
  const int seqlen = fx_param_int(NULL, "length", 10);
  const int seqlmax = fx_param_int(NULL, "lenmax", seqlen);
  const int numseq = fx_param_int(NULL, "numseq", 10);
  const char* seqout = fx_param_str(NULL, "seqfile", "seq.out");
  const char* stateout = fx_param_str(NULL, "statefile", "state.out");

  DEBUG_ASSERT_MSG(seqlen <= seqlmax, "LENMAX must bigger than LENGTH");
  DEBUG_ASSERT_MSG(numseq > 0, "NUMSEQ must be positive");

  double step = (double) (seqlmax-seqlen) / numseq;

  ArrayList<Matrix> matlst;
  load_matrix_list(profile, &matlst);
  Matrix trans, emis;
  trans.Alias(matlst[0]);
  emis.Alias(matlst[1]);

  TextWriter w_seq, w_state;
  if (!PASSED(w_seq.Open(seqout))) {
    NONFATAL("Couldn't open '%s' for writing.", seqout);
    return SUCCESS_FAIL;
  }

  if (!PASSED(w_state.Open(stateout))) {
    NONFATAL("Couldn't open '%s' for writing.", stateout);
    return SUCCESS_FAIL;
  }

  double L = seqlen;
  for (int i = 0; i < numseq; i++, L+=step) {
    Vector seq, states;
    char s[100];

    hmm_generateD_init((int)L, trans, emis, &seq, &states);
    
    sprintf(s, "%% sequence %d", i);
    print_vector(w_seq, seq, s, "%.0f,");    
    sprintf(s, "%% state sequence %d", i);
    print_vector(w_state, states, s, "%.0f,");    
  }
  return SUCCESS_PASS;
}

success_t train_baumwelch_mixture() {
  Matrix gTR;
  ArrayList<MixtureGauss> gMIX;
  ArrayList<Matrix> seqs;

  const char* seqin = fx_param_str_req(NULL, "seqfile");
  const char* proout = fx_param_str(NULL, "profile", "pro.mix.out");

  load_matrix_list(seqin, &seqs);

  if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
    const char* guess = fx_param_str_req(NULL, "guess");
    printf("Load parameters from file %s\n", guess);
    load_profileM(guess, &gTR, &gMIX);
  }
  else {
    printf("Automatic initialization not supported !!!");
    return SUCCESS_FAIL;
  }
  /*
  else { // otherwise initialized using information from the data
    int numstate = fx_param_int_req(NULL, "numstate");
    printf("Generate parameters: NUMSTATE = %d\n", numstate);
    init_gauss_param(numstate, seqs, &gTR, &gME, &gCO);
  }
  */
  int maxiter = fx_param_int(NULL, "maxiter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  hmm_trainM(seqs, &gTR, &gMIX, maxiter, tol);

  save_profileM(proout, gTR, gMIX);

  return SUCCESS_PASS;
}

success_t train_baumwelch_gaussian() {
  Matrix gTR;
  ArrayList<Vector> gME;
  ArrayList<Matrix> gCO;
  ArrayList<Matrix> seqs;

  const char* seqin = fx_param_str_req(NULL, "seqfile");
  const char* proout = fx_param_str(NULL, "profile", "pro.gauss.out");

  load_matrix_list(seqin, &seqs);

  if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
    const char* guess = fx_param_str_req(NULL, "guess");
    printf("Load parameters from file %s\n", guess);
    load_profileG(guess, &gTR, &gME, &gCO);
  }
  else { // otherwise initialized using information from the data
    int numstate = fx_param_int_req(NULL, "numstate");
    printf("Generate parameters: NUMSTATE = %d\n", numstate);
    init_gauss_param(numstate, seqs, &gTR, &gME, &gCO);
  }

  int maxiter = fx_param_int(NULL, "maxiter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  hmm_trainG(seqs, &gTR, &gME, &gCO, maxiter, tol);

  save_profileG(proout, gTR, gME, gCO);

  return SUCCESS_PASS;
}

success_t train_baumwelch_discrete() {
  //ArrayList<Matrix> matlst;
  Matrix gTR, gEM;
  ArrayList<Vector> seqs;

  const char* seqin = fx_param_str_req(NULL, "seqfile");
  const char* proout = fx_param_str(NULL, "profile", "pro.dis.out");

  load_vector_list(seqin, &seqs);

  TextWriter w_pro;
  if (!PASSED(w_pro.Open(proout))) {
    NONFATAL("Couldn't open '%s' for writing.", proout);
    return SUCCESS_FAIL;
  }
    
  if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
    ArrayList<Matrix> matlst;
    const char* guess = fx_param_str_req(NULL, "guess");
    printf("Load parameters from file %s\n", guess);
    load_matrix_list(guess, &matlst);
    gTR.Own(&matlst[0]);
    gEM.Own(&matlst[1]);
  }
  else { // otherwise randomly initialized using information from the data
    int numstate = fx_param_int_req(NULL, "numstate");
    int numsymbol = 0;
    int maxseq = 0;
    for (int i = 0; i < seqs.size(); i++) 
      if (seqs[i].length() > seqs[maxseq].length()) maxseq = i;
    for (int i = 0; i < seqs[maxseq].length(); i++)
      if (seqs[maxseq][i] > numsymbol) numsymbol = (int) seqs[maxseq][i];
    numsymbol++;
    printf("Randomly generate parameters: NUMSTATE = %d NUMSYMBOL = %d\n", numstate, numsymbol);
    Vector states;
    int L = seqs[maxseq].length();
    states.Init(L);
    for (int i = 0; i < L; i++) states[i] = rand() % numstate;
    hmm_estimateD_init(numsymbol, numstate, seqs[maxseq], states, &gTR, &gEM);
  }

  int maxiter = fx_param_int(NULL, "maxiter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  hmm_trainD(seqs, &gTR, &gEM, maxiter, tol);

  print_matrix(w_pro, gTR, "%% transmision");
  print_matrix(w_pro, gEM, "%% emission");

  return SUCCESS_PASS;
}

success_t train_viterbi_mixture() {
  Matrix gTR;
  ArrayList<MixtureGauss> gMIX;
  ArrayList<Matrix> seqs;

  const char* seqin = fx_param_str_req(NULL, "seqfile");
  const char* proout = fx_param_str(NULL, "profile", "pro.mix.out");

  load_matrix_list(seqin, &seqs);

  if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
    const char* guess = fx_param_str_req(NULL, "guess");
    printf("Load parameters from file %s\n", guess);
    load_profileM(guess, &gTR, &gMIX);
  }
  else {
    printf("Automatic initialization not supported !!!");
    return SUCCESS_FAIL;
  }
  /*
  else { // otherwise initialized using information from the data
    int numstate = fx_param_int_req(NULL, "numstate");
    printf("Generate parameters: NUMSTATE = %d\n", numstate);
    init_gauss_param(numstate, seqs, &gTR, &gME, &gCO);
  }
  */
  int maxiter = fx_param_int(NULL, "maxiter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  hmm_train_viterbiM(seqs, &gTR, &gMIX, maxiter, tol);

  save_profileM(proout, gTR, gMIX);

  return SUCCESS_PASS;
}

success_t train_viterbi_gaussian() {
  Matrix gTR;
  ArrayList<Vector> gME;
  ArrayList<Matrix> gCO;
  ArrayList<Matrix> seqs;

  const char* seqin = fx_param_str_req(NULL, "seqfile");
  const char* proout = fx_param_str(NULL, "profile", "pro.gauss.viterbi.out");

  load_matrix_list(seqin, &seqs);

  if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
    const char* guess = fx_param_str_req(NULL, "guess");
    printf("Load parameters from file %s\n", guess);
    load_profileG(guess, &gTR, &gME, &gCO);
  }
  else { // otherwise initialized using information from the data
    int numstate = fx_param_int_req(NULL, "numstate");
    printf("Generate parameters: NUMSTATE = %d\n", numstate);
    init_gauss_param(numstate, seqs, &gTR, &gME, &gCO);
  }

  int maxiter = fx_param_int(NULL, "maxiter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  hmm_train_viterbiG(seqs, &gTR, &gME, &gCO, maxiter, tol);

  save_profileG(proout, gTR, gME, gCO);

  return SUCCESS_PASS;
}

success_t train_viterbi_discrete() {
  //ArrayList<Matrix> matlst;
  Matrix gTR, gEM;
  ArrayList<Vector> seqs;

  const char* seqin = fx_param_str_req(NULL, "seqfile");
  const char* proout = fx_param_str(NULL, "profile", "pro.dis.viterbi.out");

  load_vector_list(seqin, &seqs);

  TextWriter w_pro;
  if (!PASSED(w_pro.Open(proout))) {
    NONFATAL("Couldn't open '%s' for writing.", proout);
    return SUCCESS_FAIL;
  }
    
  if (fx_param_exists(NULL, "guess")) { // guessed parameters in a file
    ArrayList<Matrix> matlst;
    const char* guess = fx_param_str_req(NULL, "guess");
    printf("Load parameters from file %s\n", guess);
    load_matrix_list(guess, &matlst);
    gTR.Own(&matlst[0]);
    gEM.Own(&matlst[1]);
  }
  else { // otherwise randomly initialized using information from the data
    int numstate = fx_param_int_req(NULL, "numstate");
    int numsymbol = 0;
    int maxseq = 0;
    for (int i = 0; i < seqs.size(); i++) 
      if (seqs[i].length() > seqs[maxseq].length()) maxseq = i;
    for (int i = 0; i < seqs[maxseq].length(); i++)
      if (seqs[maxseq][i] > numsymbol) numsymbol = (int) seqs[maxseq][i];
    numsymbol++;
    printf("Randomly generate parameters: NUMSTATE = %d NUMSYMBOL = %d\n", numstate, numsymbol);
    Vector states;
    int L = seqs[maxseq].length();
    states.Init(L);
    for (int i = 0; i < L; i++) states[i] = rand() % numstate;
    hmm_estimateD_init(numsymbol, numstate, seqs[maxseq], states, &gTR, &gEM);
  }

  int maxiter = fx_param_int(NULL, "maxiter", 500);
  double tol = fx_param_double(NULL, "tolerance", 1e-3);

  hmm_train_viterbiD(seqs, &gTR, &gEM, maxiter, tol);

  print_matrix(w_pro, gTR, "%% transmision");
  print_matrix(w_pro, gEM, "%% emission");

  return SUCCESS_PASS;
}

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
    return SUCCESS_PASS;
  }
}

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
    return SUCCESS_PASS;
  }
}
