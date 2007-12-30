#include "fastlib/fastlib.h"
#include "support.h"
//#include "discreteHMM.h"
#include "gaussianHMM.h"
#include "mixgaussHMM.h"
#include "mixtureDST.h"
/*
int main(int argc, char* argv[]) {
  printf("TEST PROGRAM FOR DISCRETE HMM GENERATE\n");

  fx_init(argc, argv);
  const char* trans_fn = fx_param_str(NULL, "trans", "trans.txt");
  const char* emis_fn = fx_param_str(NULL, "emis", "emis.txt");
  const char* seq_fn = fx_param_str(NULL, "seq", "seq.out");
  int L = fx_param_int(NULL, "length", 10);
  
  Matrix trans, emis;

  data::Load(trans_fn, &trans);
  data::Load(emis_fn, &emis);

  print_matrix(trans, "TRANS");
  print_matrix(emis, "EMIS");

  Vector states, seq;

  hmm_generateD_init(L, trans, emis, &seq, &states);

  Matrix states_, seq_;

  states_.AliasColVector(states);
  seq_.AliasColVector(seq);

  data::Save(seq_fn, seq_);
  data::Save("states.out", states_);

  fx_done();

  return 0;
}
*/
/*
int main(int argc, char* argv[]) {
  printf("TEST PROGRAM FOR DISCRETE HMM ESTIMATE\n");

  fx_init(argc, argv);
  const char* trans_fn = fx_param_str(NULL, "trans", "trans.txt");
  const char* emis_fn = fx_param_str(NULL, "emis", "emis.txt");
  int L = fx_param_int(NULL, "length", 10);
  
  Matrix trans, emis;

  data::Load(trans_fn, &trans);
  data::Load(emis_fn, &emis);

  print_matrix(trans, "TRANS");
  print_matrix(emis, "EMIS");

  Vector states, seq;

  hmm_generateD_init(L, trans, emis, &seq, &states);

  Matrix estimateTR, estimateE;
  hmm_estimateD_init(6, 3, seq, states, &estimateTR, &estimateE);

  print_matrix(estimateTR, "ESTIMATE TRANS");
  print_matrix(estimateE, "ESTIMATE EMIS");


  Matrix states_, seq_;

  states_.AliasColVector(states);
  seq_.AliasColVector(seq);

  data::Save("seq.out", seq_);
  data::Save("states.out", states_);

  fx_done();

  return 0;
}
*/
/*
int main(int argc, char* argv[]) {
  printf("TEST PROGRAM FOR DISCRETE HMM DECODE\n");

  fx_init(argc, argv);
  const char* trans_fn = fx_param_str(NULL, "trans", "trans.txt");
  const char* emis_fn = fx_param_str(NULL, "emis", "emis.txt");
  int L = fx_param_int(NULL, "length", 10);
  
  Matrix trans, emis;

  data::Load(trans_fn, &trans);
  data::Load(emis_fn, &emis);

  print_matrix(trans, "TRANS");
  print_matrix(emis, "EMIS");

  Vector states, seq;

  hmm_generateD_init(L, trans, emis, &seq, &states);

  Matrix ps, fs, bs;
  Vector s;
  int M = trans.n_cols();
  ps.Init(M, L);
  fs.Init(M, L);
  bs.Init(M, L);
  s.Init(L);

  double logpseq = hmm_decodeD(seq, trans, emis, &ps, &fs, &bs, &s);

  print_matrix(ps, "PS");
  printf("LOGPSEQ = %8.4f\n", logpseq);

  Matrix states_, seq_;

  states_.AliasColVector(states);
  seq_.AliasColVector(seq);

  data::Save("seq.out", seq_);
  data::Save("states.out", states_);

  fx_done();

  return 0;
}
*/
/*
int main(int argc, char* argv[]) {
  printf("TEST PROGRAM FOR DISCRETE HMM VITERBI\n");
  srand(time(NULL));

  fx_init(argc, argv);
  const char* trans_fn = fx_param_str(NULL, "trans", "trans.txt");
  const char* emis_fn = fx_param_str(NULL, "emis", "emis.txt");
  int L = fx_param_int(NULL, "length", 10);
  
  Matrix trans, emis;

  data::Load(trans_fn, &trans);
  data::Load(emis_fn, &emis);

  print_matrix(trans, "TRANS");
  print_matrix(emis, "EMIS");

  Vector states, seq;

  hmm_generateD_init(L, trans, emis, &seq, &states);

  Vector s;

  double logpseq = hmm_viterbiD_init(seq, trans, emis, &s);

  int error = 0;
  for (int t = 0; t < L; t++)
    if (s[t] != states[t]) error++;

  printf("Error# = %d\n", error);
  //print_vector(s, "MOST PROBABLE STATES");
  printf("LOGPSEQ = %8.4f\n", logpseq);

  Matrix states_, seq_;

  states_.AliasColVector(states);
  seq_.AliasColVector(seq);

  data::Save("seq.out", seq_);
  data::Save("states.out", states_);

  fx_done();

  return 0;
}
*/

void init_probability(Matrix* prob_matrix, int m, int n) {
  Matrix& a_ = *prob_matrix;
  a_.Init(m, n);
  for (int i = 0; i < m; i++) {
    double s = 1.0;
    for (int j = 0; j < n-1; j++) {
      a_.ref(i, j) = RAND_UNIFORM(s*0.2, s*0.8);
      s -= a_.get(i, j);
    }
    a_.ref(i, n-1) = s;
  }
}


/*
int main(int argc, char* argv[]) {
  printf("TEST PROGRAM FOR DISCRETE HMM TRAIN\n");
  //srand(time(NULL));

  fx_init(argc, argv);
  const char* seqs_fn = fx_param_str(NULL, "seqs", "seq100.out");

  Matrix seqmat;
  data::Load(seqs_fn, &seqmat);

  //print_matrix(seqmat, "SEQMAT");

  ArrayList<Vector> seqs;
  mat2arrlst(seqmat, &seqs);

  printf("SEQ ARRAY LIST (%d) = \n", seqs.size());
  
  //for (int i = 0; i < seqs.size(); i++) {
  //  char s[100];
  //  sprintf(s, "SEQ #%d", i);
  //  print_vector(seqs[i], s);
  //}

  Matrix gTR, gEM;
  init_probability(&gTR, 2, 2);
  init_probability(&gEM, 2, 6);

  print_matrix(gTR, "GUESS TRANS");
  print_matrix(gEM, "GUESS EMIS");
  printf("\nSTART...\n");

  hmm_trainD(seqs, &gTR, &gEM, 500, 1e-2);

  printf("\nFINISHED ...\n");
  print_matrix(gTR, "TRANS");
  print_matrix(gEM, "EMIS");

  data::Save("trans.out", gTR);
  data::Save("emis.out", gEM);

  fx_done();

  return 0;
}
*/

void test_hmm_estimateG() {
  printf("TEST PROGRAM FOR GUASSIAN HMM ESTIMATE\n");
  srand(time(NULL));

  const char* seq_fn = fx_param_str(NULL, "seq", "seqG.out");
  const char* state_fn = fx_param_str(NULL, "state", "states.out");
  
  Matrix seq;
  Matrix statesmat;
  Vector states;

  data::Load(seq_fn, &seq);
  data::Load(state_fn, &statesmat);
  statesmat.MakeColumnVector(0, &states);

  Matrix trans;
  ArrayList<Vector> means;
  ArrayList<Matrix> covs;

  hmm_estimateG_init(seq, states, &trans, &means, &covs);

  for (int i = 0; i < means.size(); i++) {
    printf("STATE #%d\n", i);
    print_vector(means[i], "  MEAN");
    print_matrix(covs[i], "  COV");
  }

  print_matrix(trans, "TRANS");

}

void test_hmm_generateG() {
  printf("TEST PROGRAM FOR GUASSIAN HMM GENERATE\n");
  srand(time(NULL));

  const char* trans_fn = fx_param_str(NULL, "trans", "transG.txt");
  const char* means_fn = fx_param_str(NULL, "means", "meansG.txt");
  const char* covs_fn = fx_param_str(NULL, "covs", "covsG.txt");
  const char* seq_fn = fx_param_str(NULL, "seq", "seqG.out");
  int L = fx_param_int(NULL, "length", 10);
  
  Matrix trans;
  Matrix meansmat, covsmat;

  data::Load(trans_fn, &trans);
  data::Load(means_fn, &meansmat);
  data::Load(covs_fn, &covsmat);

  ArrayList<Vector> means;
  ArrayList<Matrix> covs;

  mat2arrlst(meansmat, &means);
  mat2arrlstmat(meansmat.n_rows(), covsmat, &covs);

  DEBUG_ASSERT_MSG((means.size() == covs.size()), "main: means and covs sizes do not match");

  for (int i = 0; i < means.size(); i++) {
    printf("STATE #%d\n", i);
    print_vector(means[i], "  MEAN");
    print_matrix(covs[i], "  COV");
  }

  print_matrix(trans, "TRANS");

  Vector states;
  Matrix seq;

  hmm_generateG_init(L, trans, means, covs, &seq, &states);

  print_vector(states, "STATES");

  Matrix states_;

  states_.AliasColVector(states);

  data::Save(seq_fn, seq);
  data::Save("states.out", states_);
}

void test_hmm_decodeG() {
  printf("TEST PROGRAM FOR GUASSIAN HMM DECODE\n");
  //srand(time(NULL));

  const char* trans_fn = fx_param_str(NULL, "trans", "transG.txt");
  const char* means_fn = fx_param_str(NULL, "means", "meansG.txt");
  const char* covs_fn = fx_param_str(NULL, "covs", "covsG.txt");
  const char* seq_fn = fx_param_str(NULL, "seq", "seqG.out");
  
  Matrix trans;
  Matrix meansmat, covsmat;
  Matrix seq;

  data::Load(trans_fn, &trans);
  data::Load(means_fn, &meansmat);
  data::Load(covs_fn, &covsmat);
  data::Load(seq_fn, &seq);

  DEBUG_ASSERT_MSG(trans.n_rows() == trans.n_cols(), "CHECK TRANS");

  ArrayList<Vector> means;
  ArrayList<Matrix> covs;

  mat2arrlst(meansmat, &means);
  mat2arrlstmat(meansmat.n_rows(), covsmat, &covs);

  DEBUG_ASSERT_MSG(means.size() == covs.size(), "CHECK MEANS & COVS");

  ArrayList<Matrix> inv_covs;
  Vector det_covs;

  inv_covs.Copy(covs);
  det_covs.Init(covs.size());
  int M = trans.n_rows(); // number of states
  int L = seq.n_cols();   // sequence length
  int N = means[0].length(); // emission vector length
  for (int i = 0; i < covs.size(); i++) {
    double det = la::Determinant(covs[i]);
    la::InverseOverwrite(covs[i], &inv_covs[i]);
    det_covs[i] = pow(2.0*math::PI, -N/2.0) * pow(det, -0.5);
  }

  Matrix emis_prob;
  emis_prob.Init(M, L);
  
  for (int t = 0; t < L; t++) {
    Vector e;
    seq.MakeColumnVector(t, &e);
    for (int i = 0; i < M; i++)
      emis_prob.ref(i, t) = NORMAL_DENSITY(e, means[i], inv_covs[i], det_covs[i]);
  }

  for (int i = 0; i < means.size(); i++) {
    printf("STATE #%d\n", i);
    print_vector(means[i], "  MEAN");
    print_matrix(covs[i], "  COV");
    print_matrix(inv_covs[i], "  INVCOV");
    printf("  DET = %8.4f\n", det_covs[i]);
  }

  print_matrix(trans, "TRANS");
  print_matrix(seq, "SEQ");
  print_matrix(emis_prob, "EMIS_PROB");

  Matrix ps, fs, bs;
  Vector s;

  ps.Init(M, L);
  fs.Init(M, L);
  bs.Init(M, L);
  s.Init(L);

  double logpseq = hmm_decodeG(trans, emis_prob, &ps, &fs, &bs, &s);

  print_matrix(ps, "PS");
  printf("LOGPSEQ = %8.4f\n", logpseq);
}

void test_hmm_viterbiG() {
  printf("TEST PROGRAM FOR GUASSIAN HMM DECODE\n");
  //srand(time(NULL));

  const char* trans_fn = fx_param_str(NULL, "trans", "transG.txt");
  const char* means_fn = fx_param_str(NULL, "means", "meansG.txt");
  const char* covs_fn = fx_param_str(NULL, "covs", "covsG.txt");
  const char* seq_fn = fx_param_str(NULL, "seq", "seqG.out");
  
  Matrix trans;
  Matrix meansmat, covsmat;
  Matrix seq;

  data::Load(trans_fn, &trans);
  data::Load(means_fn, &meansmat);
  data::Load(covs_fn, &covsmat);
  data::Load(seq_fn, &seq);

  DEBUG_ASSERT_MSG(trans.n_rows() == trans.n_cols(), "CHECK TRANS");

  ArrayList<Vector> means;
  ArrayList<Matrix> covs;

  mat2arrlst(meansmat, &means);
  mat2arrlstmat(meansmat.n_rows(), covsmat, &covs);

  DEBUG_ASSERT_MSG(means.size() == covs.size(), "CHECK MEANS & COVS");

  ArrayList<Matrix> inv_covs;
  Vector det_covs;

  inv_covs.Copy(covs);
  det_covs.Init(covs.size());
  int M = trans.n_rows(); // number of states
  int L = seq.n_cols();   // sequence length
  int N = means[0].length(); // emission vector length
  for (int i = 0; i < covs.size(); i++) {
    double det = la::Determinant(covs[i]);
    la::InverseOverwrite(covs[i], &inv_covs[i]);
    det_covs[i] = pow(2.0*math::PI, -N/2.0) * pow(det, -0.5);
  }

  Matrix emis_prob;
  emis_prob.Init(M, L);
  
  for (int t = 0; t < L; t++) {
    Vector e;
    seq.MakeColumnVector(t, &e);
    for (int i = 0; i < M; i++)
      emis_prob.ref(i, t) = NORMAL_DENSITY(e, means[i], inv_covs[i], det_covs[i]);
  }

  for (int i = 0; i < means.size(); i++) {
    printf("STATE #%d\n", i);
    print_vector(means[i], "  MEAN");
    print_matrix(covs[i], "  COV");
    print_matrix(inv_covs[i], "  INVCOV");
    printf("  DET = %8.4f\n", det_covs[i]);
  }

  print_matrix(trans, "TRANS");
  print_matrix(seq, "SEQ");
  print_matrix(emis_prob, "EMIS_PROB");

  Vector s;

  double logpseq = hmm_viterbiG_init(trans, emis_prob, &s);

  print_vector(s, "MOST PROBABLE STATES");
  printf("LOGPSEQ = %8.4f\n", logpseq);

  Matrix s_;
  s_.AliasColVector(s);

  data::Save("states.txt", s_);
}

void init_gauss_param(int M, const ArrayList<Matrix>& seqs, Matrix* guessTR, ArrayList<Vector>* guessME, ArrayList<Matrix>* guessCO) {
  int N = seqs[0].n_rows();
  Matrix& gTR = *guessTR;
  ArrayList<Vector>& gME = *guessME;
  ArrayList<Matrix>& gCO = *guessCO;
  ArrayList<int> labels;
  Vector sumState;

  kmeans(seqs, M, &labels, &gME, 1000, 1e-5);

  for (int i = 0; i < labels.size(); i++) printf("%8d", labels[i]);
  printf("---1---\n");

  gTR.Init(M, M); gTR.SetZero();
  sumState.Init(M); sumState.SetZero();
  gCO.Init();
  for (int i = 0; i < M; i++) {
    Matrix m;
    m.Init(N, N); m.SetZero();
    gCO.AddBackItem(m);
  }
  //printf("---2---\n");

  int t = 0;
  for (int p=0; p < seqs.size(); p++) {
    for (int q=0; q < seqs[p].n_cols(); q++,t++)  {
      if (q == seqs[p].n_cols() -1) continue;
      int i = labels[t];
      int j = labels[t+1];

      gTR.ref(i, j)++;
      sumState[i]++;
      
      Vector data_j_Vec, sub_Vec;
      Matrix tmp_cov;

      seqs[p].MakeColumnVector(q, &data_j_Vec);
      la::SubInit(gME[i], data_j_Vec, &sub_Vec);
      tmp_cov.AliasColVector(sub_Vec);
      //printf("t = %d x = %8.3f\n", t, sub_Vec[0]);
      la::MulExpert(1, false, tmp_cov, true, tmp_cov, 1, &gCO[i]);
    }
  }
  
  for (int i = 0; i < M; i++) 
    if (sumState[i] == 0) {
      for (int j = 0; j < M; j++) gTR.ref(i, j) = 0;
      gTR.ref(i, i) = 1;
      gME[i].SetZero();
      gCO[i].SetZero();
      for (int j = 0; j < N; j++) gCO[i].ref(j, j) = 1;
    }
    else {
      for (int j = 0; j < M; j++) gTR.ref(i, j) /= sumState[i];
      la::Scale(1.0/sumState[i], &gCO[i]);
      for (int j = 0; j < N; j++) gCO[i].ref(j, j) += 1e-3; // make sure the diagonal elements are not too small
    }
}

void test_hmm_trainG() {
  printf("TEST PROGRAM FOR GUASSIAN HMM DECODE\n");
  srand(time(NULL));

  const char* seq_fn = fx_param_str(NULL, "seq", "seqG.out");
  int L = fx_param_int(NULL, "length", 10);
  int M = fx_param_int(NULL, "nstate", 2);
  
  Matrix seqsmat;
  data::Load(seq_fn, &seqsmat);

  ArrayList<Matrix> seqs;
  mat2arrlstmat(L, seqsmat, &seqs);

  printf("SEQ ARRAY LIST (%d) = \n", seqs.size());
  
  for (int i = 0; i < seqs.size(); i++) {
    char s[100];
    sprintf(s, "SEQ #%d", i);
    print_matrix(seqs[i], s);
  }

  Matrix gTR;
  ArrayList<Vector> gME;
  ArrayList<Matrix> gCO;

  init_gauss_param(M, seqs, &gTR, &gME, &gCO);

  print_matrix(gTR, "GUESS TRANS");
  for (int i = 0; i < M; i++) {
    printf("STATE #%d:\n", i);
    print_vector(gME[i], "  GUESS MEAN");
    print_matrix(gCO[i], "  GUESS COV");
  }
  printf("\nSTART...\n");

  hmm_trainG(seqs, &gTR, &gME, &gCO, 500, 1e-2);

  printf("\nFINISHED ...\n");
  print_matrix(gTR, "TRANS");
  for (int i = 0; i < M; i++) {
    printf("STATE #%d:\n", i);
    print_vector(gME[i], "  MEAN");
    print_matrix(gCO[i], "  COV");
  }

  data::Save("trans.out", gTR);
}

void test_hmm_generateM() {
  const char* seq_fn = fx_param_str(NULL, "seq", "seqG.out");
  const int L = fx_param_int(NULL, "L", 10);

  srand(time(NULL));

  MixtureGauss mix1;
  mix1.InitFromFile("meansG.txt","covsG.txt");
  mix1.print_mixture("MIX1");

  MixtureGauss mix2;
  mix2.InitFromFile("meansG1.txt","covsG1.txt");
  mix2.print_mixture("MIX2");

  Matrix trans;
  data::Load("trans.txt", &trans);
  print_matrix(trans, "TRANS");

  ArrayList<MixtureGauss> mixs;
  mixs.Init();
  mixs.AddBackItem(mix1);
  mixs.AddBackItem(mix2);

  Matrix seq;
  Vector states;
  hmm_generateM_init(L, trans, mixs, &seq, &states);

  print_matrix(seq, "SEQ");
  print_vector(states, "STATES");

  Matrix states_;
  states_.AliasColVector(states);

  data::Save(seq_fn, seq);
  data::Save("states.out", states_);
}

void test_hmm_estimateM() {
  printf("TEST PROGRAM FOR MIXTURE OF GUASSIAN HMM ESTIMATE\n");
  srand(time(NULL));

  const char* seq_fn = fx_param_str(NULL, "seq", "seqG.out");
  const char* state_fn = fx_param_str(NULL, "state", "states.out");
  const int K = fx_param_int(NULL, "K", 2);
  
  Matrix seq;
  Matrix statesmat;
  Vector states;

  data::Load(seq_fn, &seq);
  data::Load(state_fn, &statesmat);
  statesmat.MakeColumnVector(0, &states);

  Matrix trans;
  ArrayList<MixtureGauss> mixs;

  hmm_estimateM_init(K, seq, states, &trans, &mixs);

  for (int i = 0; i < mixs.size(); i++) {
    char s[100];
    sprintf(s, "STATE #%d:\n", i);
    mixs[i].print_mixture(s);
  }

  print_matrix(trans, "TRANS");

}

void test_hmm_decodeM() {
  printf("TEST PROGRAM FOR MIXTURE OF GUASSIAN HMM DECODE\n");
  //srand(time(NULL));

  const char* seq_fn = fx_param_str(NULL, "seq", "seqG.out");

  ArrayList<MixtureGauss> mixs;
  Matrix trans, seq;

  MixtureGauss mix1;
  mix1.InitFromFile("meansG.txt","covsG.txt");

  MixtureGauss mix2;
  mix2.InitFromFile("meansG1.txt","covsG1.txt");
  
  data::Load("trans.txt", &trans);
  data::Load(seq_fn, &seq);

  mixs.Init();
  mixs.AddBackItem(mix1);
  mixs.AddBackItem(mix2);

  DEBUG_ASSERT_MSG(trans.n_rows() == trans.n_cols(), "CHECK TRANS");

  int M = trans.n_rows(); // number of states
  int L = seq.n_cols();   // sequence length

  Matrix emis_prob;
  emis_prob.Init(M, L);
  
  for (int t = 0; t < L; t++) {
    Vector e;
    seq.MakeColumnVector(t, &e);
    for (int i = 0; i < M; i++)
      emis_prob.ref(i, t) = mixs[i].getPDF(e);
  }

  for (int i = 0; i < mixs.size(); i++) {
    char s[100];
    sprintf(s,"STATE #%d:\n", i);
    mixs[i].print_mixture(s);
  }

  print_matrix(trans, "TRANS");
  print_matrix(seq, "SEQ");
  print_matrix(emis_prob, "EMIS_PROB");

  Matrix ps, fs, bs;
  Vector s;

  ps.Init(M, L);
  fs.Init(M, L);
  bs.Init(M, L);
  s.Init(L);

  double logpseq = hmm_decodeG(trans, emis_prob, &ps, &fs, &bs, &s);

  print_matrix(ps, "PS");
  printf("LOGPSEQ = %8.4f\n", logpseq);
}

void test_hmm_viterbiM() {
  printf("TEST PROGRAM FOR MIXTURE OF GUASSIAN HMM VITERBI\n");
  //srand(time(NULL));

  const char* seq_fn = fx_param_str(NULL, "seq", "seqG.out");

  ArrayList<MixtureGauss> mixs;
  Matrix trans, seq;

  MixtureGauss mix1;
  mix1.InitFromFile("meansG.txt","covsG.txt");

  MixtureGauss mix2;
  mix2.InitFromFile("meansG1.txt","covsG1.txt");
  
  data::Load("trans.txt", &trans);
  data::Load(seq_fn, &seq);

  mixs.Init();
  mixs.AddBackItem(mix1);
  mixs.AddBackItem(mix2);

  DEBUG_ASSERT_MSG(trans.n_rows() == trans.n_cols(), "CHECK TRANS");

  int M = trans.n_rows(); // number of states
  int L = seq.n_cols();   // sequence length

  Matrix emis_prob;
  emis_prob.Init(M, L);
  
  for (int t = 0; t < L; t++) {
    Vector e;
    seq.MakeColumnVector(t, &e);
    for (int i = 0; i < M; i++)
      emis_prob.ref(i, t) = mixs[i].getPDF(e);
  }

  for (int i = 0; i < mixs.size(); i++) {
    char s[100];
    sprintf(s,"STATE #%d:\n", i);
    mixs[i].print_mixture(s);
  }

  print_matrix(trans, "TRANS");
  print_matrix(seq, "SEQ");
  print_matrix(emis_prob, "EMIS_PROB");

  Vector s;

  double logpseq = hmm_viterbiG_init(trans, emis_prob, &s);

  print_vector(s, "MOST PROBABLE STATES");
  printf("LOGPSEQ = %8.4f\n", logpseq);

  Matrix s_;
  s_.AliasColVector(s);

  data::Save("states.txt", s_);
}

void test_hmm_trainM() {
  printf("TEST PROGRAM FOR MIXTURE OF GUASSIAN HMM TRAIN\n");
  srand(time(NULL));

  const char* seq_fn = fx_param_str(NULL, "seq", "seqG.out");
  const char* state_fn = fx_param_str(NULL, "state", "states.out");
  const int K = fx_param_int(NULL, "K", 2);
  int L = fx_param_int(NULL, "length", 10);
  int MAX_ITER = fx_param_int(NULL, "iter", 100);
    
  Matrix seq;
  Matrix statesmat;
  Vector states;

  data::Load(seq_fn, &seq);
  data::Load(state_fn, &statesmat);
  statesmat.MakeColumnVector(0, &states);

  Matrix trans;
  ArrayList<MixtureGauss> mixs;

  hmm_estimateM_init(K, seq, states, &trans, &mixs);

  Matrix seqsmat;
  data::Load(seq_fn, &seqsmat);

  ArrayList<Matrix> seqs;
  mat2arrlstmat(L, seqsmat, &seqs);

  for (int i = 0; i < mixs.size(); i++) {
    char s[100];
    sprintf(s, "STATE #%d:\n", i);
    mixs[i].print_mixture(s);
  }

  print_matrix(trans, "TRANS");
  printf("\nSTART...\n");

  hmm_trainM(seqs, &trans, &mixs, MAX_ITER, 1e-2);

  printf("\nFINISHED...\n");
  for (int i = 0; i < mixs.size(); i++) {
    char s[100];
    sprintf(s, "STATE #%d:\n", i);
    mixs[i].print_mixture(s);
  }

  print_matrix(trans, "TRANS");

}

int main(int argc, char* argv[]) {
  fx_init(argc, argv);
  //test_hmm_estimateG();
  //test_hmm_generateG();
  //test_hmm_decodeG();
  //test_hmm_viterbiG();
  //test_hmm_trainG();
  //test_hmm_generateM();
  //test_hmm_estimateM();
  //test_hmm_decodeM();
  //test_hmm_viterbiM();
  test_hmm_trainM();
  
  /*
  MixtureGauss mix;
  mix.InitFromFile("meansG.txt", "covsG.txt");
  
  Matrix m;
  Vector v;
  mix.generate(&v);

  int N = fx_param_int(NULL, "N", 1000);
  m.Init(v.length(), N);
  for (int i = 0; i < N; i++) {
    Vector t;
    mix.generate(&t);
    for (int j = 0; j < t.length(); j++)
      m.ref(j, i) = t[j];
  }

  data::Save("mixsample.out", m);
  */
  /*
  Matrix d;
  data::Load("mixsample.out", &d);
  mix.print_mixture("----BEFORE ACCUMULATE----");

  mix.start_accumulate();
  for (int i = 0; i < d.n_cols(); i++) {
    Vector v;
    d.MakeColumnVector(i, &v);
    for (int k = 0; k < mix.n_clusters(); k++)
      mix.accumulate(mix.getPDF(k, v)/mix.getPDF(v), k, v);
  }
  mix.end_accumulate();

  mix.print_mixture("----AFTER ACCUMULATE-----");
  */
  fx_done();
  return 0;
}


/*
int main() {
  Vector a, b;
  RAND_NORMAL_01_INIT(5000, &a);
  RAND_NORMAL_01_INIT(5000, &b);
  for (int i = 0; i < b.length(); i++) b[i]+=10;

  Matrix ma, mb;

  ma.AliasColVector(a);
  mb.AliasColVector(b);

  data::Save("a.txt", ma);
  data::Save("b.txt", mb);
}
*/
