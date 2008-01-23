#include "fastlib/fastlib.h"
#include "support.h"
#include "gaussianHMM.h"

using namespace hmm_support;

void GaussianHMM::setModel(const Matrix& transmission,  const ArrayList<Vector>& list_mean_vec,const ArrayList<Matrix>& list_covariance_mat) {
  DEBUG_ASSERT(transmission.n_rows() == transmission.n_cols());
  DEBUG_ASSERT(transmission.n_rows() == list_mean_vec.size());
  DEBUG_ASSERT(transmission.n_rows() == list_covariance_mat.size());
  for (int i = 1; i < list_mean_vec.size(); i++) {
    DEBUG_ASSERT(list_mean_vec[0].length() == list_covariance_mat[i].n_rows());
    DEBUG_ASSERT(list_mean_vec[0].length() == list_covariance_mat[i].n_cols());
    DEBUG_ASSERT(list_mean_vec[0].length() == list_mean_vec[i].length());
  }
  transmission_.Destruct();
  list_mean_vec_.Destruct();
  list_covariance_mat_.Destruct();
  transmission_.Copy(transmission);
  list_mean_vec_.Copy(list_mean_vec);
  list_covariance_mat_.Copy(list_covariance_mat);
  CalculateInverse();
}

void GaussianHMM::Init(const Matrix& transmission,  const ArrayList<Vector>& list_mean_vec,const ArrayList<Matrix>& list_covariance_mat) {
  transmission_.Copy(transmission);
  list_mean_vec_.Copy(list_mean_vec);
  list_covariance_mat_.Copy(list_covariance_mat);
  DEBUG_ASSERT(transmission.n_rows() == transmission.n_cols());
  DEBUG_ASSERT(transmission.n_rows() == list_mean_vec.size());
  DEBUG_ASSERT(transmission.n_rows() == list_covariance_mat.size());
  for (int i = 1; i < list_mean_vec.size(); i++) {
    DEBUG_ASSERT(list_mean_vec[0].length() == list_covariance_mat[i].n_rows());
    DEBUG_ASSERT(list_mean_vec[0].length() == list_covariance_mat[i].n_cols());
    DEBUG_ASSERT(list_mean_vec[0].length() == list_mean_vec[i].length());
  }  
  CalculateInverse();
}

void GaussianHMM::InitFromFile(const char* profile) {
  if (!PASSED(GaussianHMM::LoadProfile(profile, &transmission_, &list_mean_vec_, &list_covariance_mat_)))
    FATAL("Couldn't open '%s' for reading.", profile);
  list_inverse_cov_mat_.Copy(list_covariance_mat_);
  gauss_const_vec_.Init(list_covariance_mat_.size());
  CalculateInverse();
}

void GaussianHMM::InitFromData(const ArrayList<Matrix>& list_data_seq, int numstate) {
  GaussianHMM::InitGaussParameter(numstate, list_data_seq, &transmission_, &list_mean_vec_, &list_covariance_mat_);
  list_inverse_cov_mat_.Copy(list_covariance_mat_);
  gauss_const_vec_.Init(list_covariance_mat_.size());
  CalculateInverse();
}

void GaussianHMM::InitFromData(const Matrix& data_seq, const Vector& state_seq)  {
  GaussianHMM::EstimateInit(data_seq, state_seq, &transmission_, 
			    &list_mean_vec_, &list_covariance_mat_);
  CalculateInverse();
}

void GaussianHMM::LoadProfile(const char* profile) {
  transmission_.Destruct();
  list_mean_vec_.Destruct();
  list_covariance_mat_.Destruct();
  list_inverse_cov_mat_.Destruct();
  gauss_const_vec_.Destruct();
  InitFromFile(profile);
}

void GaussianHMM::SaveProfile(const char* profile) const {
  GaussianHMM::SaveProfile(profile, transmission_, list_mean_vec_, list_covariance_mat_);
}

void GaussianHMM::CalculateInverse() {
  int M = transmission_.n_rows();
  int N = list_mean_vec_[0].length();
  for (int i = 0; i < M; i++) {
    la::InverseOverwrite(list_covariance_mat_[i], &list_inverse_cov_mat_[i]);
    gauss_const_vec_[i] = pow(2.0*math::PI, -N/2.0) * pow(la::Determinant(list_covariance_mat_[i]), -0.5);
  }
}

void GaussianHMM::GenerateSequence(int L, Matrix* data_seq, Vector* state_seq) const {
  GaussianHMM::GenerateInit(L, transmission_, list_mean_vec_, list_covariance_mat_, data_seq, state_seq);
}

void GaussianHMM::EstimateModel(const Matrix& data_seq, const Vector& state_seq) {
  transmission_.Destruct();
  list_mean_vec_.Destruct();
  list_covariance_mat_.Destruct();
  GaussianHMM::EstimateInit(data_seq, state_seq, &transmission_, 
			    &list_mean_vec_, &list_covariance_mat_);
  CalculateInverse();
}

void GaussianHMM::EstimateModel(int numstate, const Matrix& data_seq, const Vector& state_seq) {
  transmission_.Destruct();
  list_mean_vec_.Destruct();
  list_covariance_mat_.Destruct();
  GaussianHMM::EstimateInit(numstate, data_seq, state_seq, &transmission_, 
			    &list_mean_vec_, &list_covariance_mat_);
  CalculateInverse();
}

void GaussianHMM::DecodeOverwrite(const Matrix& data_seq, Matrix* state_prob_mat, 
				  Matrix* forward_prob_mat, Matrix* backward_prob_mat, Vector* scale_vec) const {
  int M = transmission_.n_rows();
  int L = data_seq.n_cols();  
  Matrix emission_prob_mat(M, L);
  GaussianHMM::CalculateEmissionProb(data_seq, list_mean_vec_, list_inverse_cov_mat_,
				     gauss_const_vec_, &emission_prob_mat);
  GaussianHMM::Decode(transmission_, emission_prob_mat, state_prob_mat, forward_prob_mat,
		      backward_prob_mat, scale_vec);
}

void GaussianHMM::DecodeInit(const Matrix& data_seq, Matrix* state_prob_mat, 
			     Matrix* forward_prob_mat, Matrix* backward_prob_mat, Vector* scale_vec) const {
  int M = transmission_.n_rows();
  int L = data_seq.n_cols();  
  state_prob_mat->Init(M, L);
  forward_prob_mat->Init(M, L);
  backward_prob_mat->Init(M, L);
  scale_vec->Init(L);
  Matrix emission_prob_mat(M, L);
  GaussianHMM::CalculateEmissionProb(data_seq, list_mean_vec_, list_inverse_cov_mat_,
				     gauss_const_vec_, &emission_prob_mat);
  GaussianHMM::Decode(transmission_, emission_prob_mat, state_prob_mat, forward_prob_mat,
		      backward_prob_mat, scale_vec);
}

double GaussianHMM::ComputeLogLikelihood(const Matrix& data_seq) const {
  int L = data_seq.n_cols();
  int M = transmission_.n_rows();
  Matrix fs(M, L), emis_prob(M, L);
  Vector sc;
  sc.Init(L);
  GaussianHMM::CalculateEmissionProb(data_seq, list_mean_vec_, list_inverse_cov_mat_, gauss_const_vec_, &emis_prob);
  GaussianHMM::ForwardProcedure(L, transmission_, emis_prob, &sc, &fs);
  double loglik = 0;
  for (int t = 0; t < L; t++)
    loglik += log(sc[t]);
  return loglik;
}

void GaussianHMM::ComputeLogLikelihood(const ArrayList<Matrix>& list_data_seq, ArrayList<double>* list_likelihood) const {
  int L = 0;
  for (int i = 0; i < list_data_seq.size(); i++)
    if (list_data_seq[i].n_cols() > L) L = list_data_seq[i].n_cols();
  int M = transmission_.n_rows();
  Matrix fs(M, L), emis_prob(M, L);
  Vector sc;
  sc.Init(L);
  list_likelihood->Init();
  for (int i = 0; i < list_data_seq.size(); i++) {
    int L = list_data_seq[i].n_cols();
    GaussianHMM::CalculateEmissionProb(list_data_seq[i], list_mean_vec_, list_inverse_cov_mat_, gauss_const_vec_, &emis_prob);
    GaussianHMM::ForwardProcedure(L, transmission_, emis_prob, &sc, &fs);
    double loglik = 0;
    for (int t = 0; t < L; t++)
      loglik += log(sc[t]);
    list_likelihood->AddBackItem(loglik);
  }
}

void GaussianHMM::ComputeViterbiStateSequence(const Matrix& data_seq, Vector* state_seq) const {
  int M = transmission_.n_rows();
  int L = data_seq.n_cols();
  Matrix emis_prob(M, L);
  GaussianHMM::CalculateEmissionProb(data_seq, list_mean_vec_, list_inverse_cov_mat_, gauss_const_vec_, &emis_prob);
  GaussianHMM::ViterbiInit(transmission_, emis_prob, state_seq);
}

void GaussianHMM::TrainBaumWelch(const ArrayList<Matrix>& list_data_seq, int max_iteration, double tolerance) {
  GaussianHMM::Train(list_data_seq, &transmission_, &list_mean_vec_, &list_covariance_mat_, max_iteration, tolerance);
  CalculateInverse();
}

void GaussianHMM::TrainViterbi(const ArrayList<Matrix>& list_data_seq, int max_iteration, double tolerance) {
  GaussianHMM::TrainViterbi(list_data_seq, &transmission_, &list_mean_vec_, &list_covariance_mat_, max_iteration, tolerance);
  CalculateInverse();
}

success_t GaussianHMM::LoadProfile(const char* profile, Matrix* trans, ArrayList<Vector>* means, ArrayList<Matrix>* covs) {
  ArrayList<Matrix> matlst;
  if (!PASSED(load_matrix_list(profile, &matlst)))
    return SUCCESS_FAIL;

  DEBUG_ASSERT(matlst.size() > 0);
  trans->Copy(matlst[0]);
  means->Init();
  covs->Init();
  int M = trans->n_rows(); // num of states
  DEBUG_ASSERT(matlst.size() == 2*M+1);
  int N = matlst[1].n_rows(); // dimension
  for (int i = 1; i < 2*M+1; i+=2) {
    DEBUG_ASSERT(matlst[i].n_rows()==N && matlst[i].n_cols()==1);
    DEBUG_ASSERT(matlst[i+1].n_rows()==N && matlst[i+1].n_cols()==N);
    Vector m;
    matlst[i].MakeColumnVector(0, &m);
    means->AddBackItem(m);
    covs->AddBackItem(matlst[i+1]);
  }
  return SUCCESS_PASS;
}

success_t GaussianHMM::SaveProfile(const char* profile, const Matrix& trans, const ArrayList<Vector>& means, const ArrayList<Matrix>& covs) {
  TextWriter w_pro;
  if (!PASSED(w_pro.Open(profile))) {
    NONFATAL("Couldn't open '%s' for writing.", profile);
    return SUCCESS_FAIL;
  }
  int M = trans.n_rows(); // num of states
  DEBUG_ASSERT(means.size() == M && covs.size() == M);
  int N = means[0].length(); // dimension
  print_matrix(w_pro, trans, "% transmission", "%f,");
  for (int i = 0; i < M; i++) {
    DEBUG_ASSERT(means[i].length() == N);
    DEBUG_ASSERT(covs[i].n_rows()==N && covs[i].n_cols()==N);
    char s[100];
    sprintf(s, "%% mean - state %d", i);
    print_vector(w_pro, means[i], s, "%f,");    
    sprintf(s, "%% covariance - state%d", i);
    print_matrix(w_pro, covs[i], s, "%f,");    
  }
  return SUCCESS_PASS;
}

void GaussianHMM::GenerateInit(int L, const Matrix& trans, const ArrayList<Vector>& means, const ArrayList<Matrix>& covs, Matrix* seq, Vector* states){
  DEBUG_ASSERT_MSG((trans.n_rows()==trans.n_cols() && trans.n_rows()==means.size() && trans.n_rows()==covs.size()), "hmm_generateG_init: matrices sizes do not match");
  Matrix trsum;
  Matrix& seq_ = *seq;
  Vector& states_ = *states;
  int M, N;
  int cur_state;

  M = trans.n_rows();
  N = means[0].length();  // emission vector length

  trsum.Copy(trans);

  for (int i = 0; i < M; i++)
    for (int j = 1; j < M; j++)
      trsum.set(i, j, trsum.get(i, j) + trsum.get(i, j-1));

  seq_.Init(N, L);
  states_.Init(L);

  cur_state = 0; // starting state is 0
  
  for (int i = 0; i < L; i++) {
    int j;

    // next state
    double r = RAND_UNIFORM_01();
    for (j = 0; j < M; j++)
      if (r <= trsum.get(cur_state, j)) break;
    cur_state = j;
	
    // emission
    Vector e;
    RAND_NORMAL_INIT(means[cur_state], covs[cur_state], &e);
    for (j = 0; j < N; j++)
      seq_.ref(j, i) = e[j];
    states_[i] = cur_state;
  }
}

void GaussianHMM::EstimateInit(const Matrix& seq, const Vector& states, Matrix* trans, ArrayList<Vector>* means, ArrayList<Matrix>* covs) {
  DEBUG_ASSERT_MSG((seq.n_cols()==states.length()), "hmm_estimateG_init: sequence and states length must be the same");
  int M = 0;
  for (int i = 0; i < seq.n_cols(); i++)
    if (states[i] > M) M = (int) states[i];
  M++;
  GaussianHMM::EstimateInit(M, seq, states, trans, means, covs);
}

void GaussianHMM::EstimateInit(int numStates, const Matrix& seq, const Vector& states, Matrix* trans, ArrayList<Vector>* means, ArrayList<Matrix>* covs) {
  DEBUG_ASSERT_MSG((seq.n_cols()==states.length()), "hmm_estimateD_init: sequence and states length must be the same");
  
  int N = seq.n_rows(); // emission vector length
  int M = numStates;    // number of states
  int L = seq.n_cols(); // sequence length
  
  Matrix &trans_ = *trans;
  ArrayList<Vector>& mean_ = *means;
  ArrayList<Matrix>& cov_ = *covs;
  Vector stateSum;

  trans_.Init(M, M);
  mean_.Init();
  cov_.Init();
  stateSum.Init(M);

  trans_.SetZero();
  for (int i = 0; i < M; i++) {
    Vector m;
    m.Init(N); m.SetZero();
    mean_.AddBackItem(m);

    Matrix c;
    c.Init(N, N); c.SetZero();
    cov_.AddBackItem(c);
  }

  stateSum.SetZero();
  for (int i = 0; i < L-1; i++) {
    int state = (int) states[i];
    int next_state = (int) states[i+1];
    stateSum[state]++;
    trans_.ref(state, next_state)++;
  }
  for (int i = 0; i < M; i++) {
    if (stateSum[i] == 0) stateSum[i] = -INFINITY;
    for (int j = 0; j < M; j++)
      trans_.ref(i, j) /= stateSum[i];
  }

  stateSum.SetZero();
  for (int i = 0; i < L; i++) {
    int state = (int) states[i];
    Vector e;
    seq.MakeColumnVector(i, &e);

    stateSum[state]++;
    la::AddTo(e, &mean_[state]);
  }

  for (int i = 0; i < M; i++)
    if (stateSum[i] != 0) 
      la::Scale(1.0/stateSum[i], &mean_[i]);
  
  for (int i = 0; i < L; i++) {
    int state = (int) states[i];
    Vector e;
    seq.MakeColumnVector(i, &e);
    Vector d;
    la::SubInit(e, mean_[state], &d);
    Matrix D;
    D.AliasColVector(d);
    la::MulExpert(1.0, false, D, true, D, 1.0, &cov_[state]);
  }

  for (int i = 0; i < M; i++)
    if (stateSum[i] != 0)
      la::Scale(1.0/stateSum[i], &cov_[i]);
}

void GaussianHMM::ForwardProcedure(int L, const Matrix& trans, const Matrix& emis_prob, Vector *scales, Matrix* fs) {
  int M = trans.n_rows();

  Matrix& fs_ = *fs;
  Vector& s_ = *scales;

  fs_.SetZero();
  s_.SetZero();
  // NOTE: start state is 0
  // time t = 0
  for (int i = 0; i < M; i++) {
    fs_.ref(i, 0) = trans.get(0, i) * emis_prob.get(i, 0);
    s_[0] += fs_.get(i, 0);
  }
  for (int i = 0; i < M; i++)
    fs_.ref(i, 0) /= s_[0];

  // time t = 1 -> L-1
  for (int t = 1; t < L; t++) {
    for (int j = 0; j < M; j++) {
      for (int i = 0; i < M; i++)
	fs_.ref(j, t) += fs_.get(i, t-1)*trans.get(i, j);
      fs_.ref(j, t) *= emis_prob.get(j, t);
      s_[t] += fs_.get(j, t);
    }
    for (int j = 0; j < M; j++)
      fs_.ref(j, t) /= s_[t];
  }
}

void GaussianHMM::BackwardProcedure(int L, const Matrix& trans, const Matrix& emis_prob, const Vector& scales, Matrix* bs) {
  int M = trans.n_rows();

  Matrix& bs_ = *bs;
  bs_.SetZero();
  for (int i = 0; i < M; i++)
    bs_.ref(i, L-1) = 1.0;

  for (int t = L-2; t >= 0; t--) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < M; j++)
	bs_.ref(i, t) += trans.get(i, j) * bs_.ref(j, t+1) * emis_prob.get(j, t+1);
      bs_.ref(i, t) /= scales[t+1];
    }
  }
}

double GaussianHMM::Decode(int L, const Matrix& trans, const Matrix& emis_prob, Matrix* pstates, Matrix* fs, Matrix* bs, Vector* scales) {
  int M = trans.n_rows();

  DEBUG_ASSERT_MSG((L==pstates->n_cols() && L==fs->n_cols() && L == bs->n_cols() && 
		    M==trans.n_cols() && M==emis_prob.n_rows()),"hmm_decodeG: sizes do not match");
  
  Matrix& ps_ = *pstates;
  Vector& s_ = *scales;

  GaussianHMM::ForwardProcedure(L, trans, emis_prob, &s_, fs);
  GaussianHMM::BackwardProcedure(L, trans, emis_prob, s_, bs);

  for (int i = 0; i < M; i++)
    for (int t = 0; t < L; t++)
      ps_.ref(i, t) = fs->get(i,t) * bs->get(i,t);

  double logpseq = 0;
  for (int t = 0; t < L; t++) 
    logpseq += log(s_[t]);

  return logpseq;
}

double GaussianHMM::Decode(const Matrix& trans, const Matrix& emis_prob, Matrix* pstates, Matrix* fs, Matrix* bs, Vector* scales) {
  int L = emis_prob.n_cols();
  return GaussianHMM::Decode(L, trans, emis_prob, pstates, fs, bs, scales);
}

double GaussianHMM::ViterbiInit(const Matrix& trans, const Matrix& emis_prob, Vector* states) {
  int L = emis_prob.n_cols();
  return GaussianHMM::ViterbiInit(L, trans, emis_prob, states);
}

double GaussianHMM::ViterbiInit(int L, const Matrix& trans, const Matrix& emis_prob, Vector* states) {
  int M = trans.n_rows();
  DEBUG_ASSERT_MSG((M==trans.n_cols() && M==emis_prob.n_rows()),"hmm_viterbiG: sizes do not match");
  
  Vector& s_ = *states;
  s_.Init(L);
  
  Vector v, vOld;
  v.Init(M);
  v.SetAll(-INFINITY);
  v[0] = 0;
  vOld.Copy(v);

  Matrix w;
  w.Init(M, L);

  Matrix logtrans;
  logtrans.Init(M, M);

  for (int i = 0; i < M; i++)
    for (int j = 0; j < M; j++) logtrans.ref(i, j) = log(trans.get(i, j));


  for (int t = 0; t < L; t++) {
    for (int j = 0; j < M; j++) {
      double bestVal = -INFINITY;
      double bestPtr = -1;      
      for (int i = 0; i < M; i++) {
	double val = vOld[i] + logtrans.get(i, j);
	if (val > bestVal) {
	  bestVal = val;
	  bestPtr = i;
	}
      }
      v[j] = bestVal + log(emis_prob.get(j, t));
      w.ref(j, t) = bestPtr;
    }
    vOld.CopyValues(v);
  }

  double bestVal = -INFINITY;
  double bestPtr = -1;
  for (int i = 0; i < M; i++)
    if (v[i] > bestVal) {
      bestVal = v[i];
      bestPtr = i;
    }
  
  s_[L-1] = bestPtr;
  for (int t = L-2; t >= 0; t--) {
    s_[t] = w.get((int)s_[t+1], t+1);
  }

  return bestVal;
}

void GaussianHMM::CalculateEmissionProb(const Matrix& seq, const ArrayList<Vector>& means, const ArrayList<Matrix>& inv_covs, const Vector& det, Matrix* emis_prob) {
  int L = seq.n_cols();
  int M = means.size();
  for (int t = 0; t < L; t++) {
    Vector e;
    seq.MakeColumnVector(t, &e);
    for (int i = 0; i < M; i++)
      emis_prob->ref(i, t) = NORMAL_DENSITY(e, means[i], inv_covs[i], det[i]);
  }
}

void GaussianHMM::InitGaussParameter(int M, const ArrayList<Matrix>& seqs, Matrix* guessTR, ArrayList<Vector>* guessME, ArrayList<Matrix>* guessCO) {
  int N = seqs[0].n_rows();
  Matrix& gTR = *guessTR;
  ArrayList<Vector>& gME = *guessME;
  ArrayList<Matrix>& gCO = *guessCO;
  ArrayList<int> labels;
  Vector sumState;

  kmeans(seqs, M, &labels, &gME, 1000, 1e-5);

  //for (int i = 0; i < labels.size(); i++) printf("%8d", labels[i]);
  //printf("---1---\n");

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
  //printf("---3---\n");

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
  //printf("---4---\n");
}

void GaussianHMM::TrainViterbi(const ArrayList<Matrix>& seqs, Matrix* guessTR, ArrayList<Vector>* guessME, ArrayList<Matrix>* guessCO, int max_iter, double tol) {
  Matrix &gTR = *guessTR;
  ArrayList<Vector>& gME = *guessME;
  ArrayList<Matrix>& gCO = *guessCO;
  int L = -1;
  int M = gTR.n_rows();
  int N = gME[0].length();
  DEBUG_ASSERT_MSG((M==gTR.n_cols() && M==gME.size() && M == gCO.size()),"hmm_trainD: sizes do not match");
  
  for (int i = 0; i < seqs.size(); i++)
    if (seqs[i].n_cols() > L) L = seqs[i].n_cols();

  Matrix TR; // accumulating transition
  ArrayList<Vector> ME; // accumulating mean
  ArrayList<Matrix> CO; // accumulating covariance
  ArrayList<Matrix> INV_CO; // inverse matrix of the covariance
  Vector DET; // the determinant * constant of the Normal PDF formula
  TR.Init(M, M);
  ME.Copy(gME);
  CO.Copy(gCO);
  INV_CO.Copy(CO);
  DET.Init(M);

  Matrix emis_prob;
  Vector sumState; // the denominator for each state

  emis_prob.Init(M, L);
  sumState.Init(M);

  double loglik = 0, oldlog;
  for (int iter = 0; iter < max_iter; iter++) {
    oldlog = loglik;
    loglik = 0;

    // set the accumulating values to zeros and compute the inverse matrices and determinant constants
    TR.SetZero();
    for (int i = 0; i < M; i++) {
      ME[i].SetZero();
      CO[i].SetZero();
      la::InverseOverwrite(gCO[i], &INV_CO[i]);
      DET[i] = pow(2.0*math::PI, -N/2.0) * pow(la::Determinant(gCO[i]), -0.5);
    }
    sumState.SetZero();

    // for each sequence, we will use forward-backward procedure and then accumulate
    for (int idx = 0; idx < seqs.size(); idx++) {
      L = seqs[idx].n_cols();
      Vector states;
      GaussianHMM::CalculateEmissionProb(seqs[idx], gME, INV_CO, DET, &emis_prob); // first calculate the emission probabilities of the sequence
      loglik += GaussianHMM::ViterbiInit(L, gTR, emis_prob, &states); // get the most probable state sequence
      
      // accumulate expected transition & mean & covariance
      for (int t = 0; t < L-1; t++) {
	int i = (int) states[t];
	int j = (int) states[t+1];
	TR.ref(i, j) ++;
      }
      
      for (int t = 0; t < L; t++) {
	Vector e;
	seqs[idx].MakeColumnVector(t, &e);
	int i = (int) states[t];
	sumState[i] ++;
	la::AddTo(e, &ME[i]);

	Vector d;
	la::SubInit(e, gME[i], &d);
	Matrix D;
	D.AliasColVector(d);
	la::MulExpert(1, false, D, true, D, 1.0, &CO[i]);
      }
      // end accumulate
    }

    // after accumulate all sequences: re-estimate transition & mean & covariance for the next iteration
    for (int i = 0; i < M; i++) {
      double s = 0;
      for (int j = 0; j < M; j++) s += TR.get(i, j);
      if (s == 0) {
	for (int j = 0; j < M; j++) gTR.ref(i, j) = 0;
	gTR.ref(i, i) = 1;
      }
      else {
	for (int j = 0; j < M; j++) gTR.ref(i, j) = TR.get(i, j) / s;
      }
      
      if (sumState[i] != 0) {
	la::ScaleOverwrite(1.0/sumState[i], ME[i], &gME[i]);
	la::ScaleOverwrite(1.0/sumState[i], CO[i], &gCO[i]);
      }
    }
    // end re-estimate

    printf("Iter = %d Loglik = %8.4f\n", iter, loglik);
    if (fabs(oldlog - loglik) < tol) {
      printf("\nConverged after %d iterations\n", iter);
      break;
    }
    oldlog = loglik;
  }
}


void GaussianHMM::Train(const ArrayList<Matrix>& seqs, Matrix* guessTR, ArrayList<Vector>* guessME, ArrayList<Matrix>* guessCO, int max_iter, double tol) {
  Matrix &gTR = *guessTR;
  ArrayList<Vector>& gME = *guessME;
  ArrayList<Matrix>& gCO = *guessCO;
  int L = -1;
  int M = gTR.n_rows();
  int N = gME[0].length();
  DEBUG_ASSERT_MSG((M==gTR.n_cols() && M==gME.size() && M == gCO.size()),"hmm_trainD: sizes do not match");
  
  for (int i = 0; i < seqs.size(); i++)
    if (seqs[i].n_cols() > L) L = seqs[i].n_cols();

  Matrix TR; // guess transition and emission matrix
  ArrayList<Vector> ME; // accumulating mean
  ArrayList<Matrix> CO; // accumulating covariance
  ArrayList<Matrix> INV_CO; // inverse matrix of the covariance
  Vector DET; // the determinant * constant of the Normal PDF formula
  TR.Init(M, M);
  ME.Copy(gME);
  CO.Copy(gCO);
  INV_CO.Copy(CO);
  DET.Init(M);

  Matrix ps, fs, bs, emis_prob; // to hold hmm_decodeG results
  Vector s; // scaling factors
  Vector sumState; // the denominator for each state

  ps.Init(M, L);
  fs.Init(M, L);
  bs.Init(M, L);
  s.Init(L);
  emis_prob.Init(M, L);
  sumState.Init(M);

  double loglik = 0, oldlog;
  for (int iter = 0; iter < max_iter; iter++) {
    oldlog = loglik;
    loglik = 0;

    // set the accumulating values to zeros and compute the inverse matrices and determinant constants
    TR.SetZero();
    for (int i = 0; i < M; i++) {
      ME[i].SetZero();
      CO[i].SetZero();
      la::InverseOverwrite(gCO[i], &INV_CO[i]);
      DET[i] = pow(2.0*math::PI, -N/2.0) * pow(la::Determinant(gCO[i]), -0.5);
    }
    sumState.SetZero();

    // for each sequence, we will use forward-backward procedure and then accumulate
    for (int idx = 0; idx < seqs.size(); idx++) {
      // first calculate the emission probabilities of the sequence
      L = seqs[idx].n_cols();
      for (int t = 0; t < L; t++) {
	Vector e;
	seqs[idx].MakeColumnVector(t, &e);
	for (int i = 0; i < M; i++)
	  emis_prob.ref(i, t) = NORMAL_DENSITY(e, gME[i], INV_CO[i], DET[i]);
      }
      
      loglik += GaussianHMM::Decode(L, gTR, emis_prob, &ps, &fs, &bs, &s); // forward - backward procedure
      
      // accumulate expected transition & mean & covariance
      for (int t = 0; t < L-1; t++) {
	for (int i = 0; i < M; i++)
	  for (int j = 0; j < M; j++)
	    TR.ref(i, j) += fs.get(i, t) * gTR.get(i, j) * emis_prob.get(j, t+1) * bs.get(j, t+1) / s[t+1];
      }
      
      for (int t = 0; t < L; t++) {
	Vector e;
	seqs[idx].MakeColumnVector(t, &e);
	for (int i = 0; i < M; i++) {
	  sumState[i] += ps.get(i, t);
	  la::AddExpert(ps.get(i, t), e, &ME[i]);

	  Matrix D;
	  D.AliasColVector(e);
	  la::MulExpert(ps.get(i, t), false, D, true, D, 1.0, &CO[i]);
	}
      }
      // end accumulate
    }

    // after accumulate all sequences: re-estimate transition & mean & covariance for the next iteration
    for (int i = 0; i < M; i++) {
      double s = 0;
      for (int j = 0; j < M; j++) s += TR.get(i, j);
      if (s == 0) {
	for (int j = 0; j < M; j++) gTR.ref(i, j) = 0;
	gTR.ref(i, i) = 1;
      }
      else {
	for (int j = 0; j < M; j++) gTR.ref(i, j) = TR.get(i, j) / s;
      }
      
      if (sumState[i] != 0) {
	la::ScaleOverwrite(1.0/sumState[i], ME[i], &gME[i]);
	Matrix D;
	D.AliasColVector(gME[i]);
	la::MulExpert(-1.0, false, D, true, D, 1.0/sumState[i], &CO[i]);
	gCO[i].CopyValues(CO[i]);
      }
    }
    // end re-estimate

    printf("Iter = %d Loglik = %8.4f\n", iter, loglik);
    if (fabs(oldlog - loglik) < tol) {
      printf("\nConverged after %d iterations\n", iter);
      break;
    }
    oldlog = loglik;
  }
}
