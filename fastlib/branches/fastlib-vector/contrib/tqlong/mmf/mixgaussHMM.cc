/**
 * @file mixgaussHMM.cc
 *
 * This file contains the implementation of functions in mixgaussHMM.h
 */

#include "fastlib/fastlib.h"
#include "support.h"
#include "mixgaussHMM.h"
#include "gaussianHMM.h"

using namespace hmm_support;

void MixtureofGaussianHMM::setModel(const Matrix& transmission,
				    const ArrayList<MixtureGauss>& list_mixture_gauss) {
  DEBUG_ASSERT(transmission.n_rows() == transmission.n_cols());
  DEBUG_ASSERT(transmission.n_rows() == list_mixture_gauss.size());
  transmission_.Destruct();
  list_mixture_gauss_.Renew();
  transmission_.Copy(transmission);
  list_mixture_gauss_.InitCopy(list_mixture_gauss);
}

void MixtureofGaussianHMM::InitFromFile(const char* profile) {
  if (!PASSED(MixtureofGaussianHMM::LoadProfile(profile, &transmission_, &list_mixture_gauss_)))
    FATAL("Couldn't open '%s' for reading.", profile);
}

void MixtureofGaussianHMM::LoadProfile(const char* profile) {
  transmission_.Destruct();
  list_mixture_gauss_.Renew();
  InitFromFile(profile);
}

void MixtureofGaussianHMM::SaveProfile(const char* profile) const {
  MixtureofGaussianHMM::SaveProfile(profile, transmission_, list_mixture_gauss_);
}

void MixtureofGaussianHMM::GenerateSequence(int L, Matrix* data_seq, Vector* state_seq) const {
  MixtureofGaussianHMM::GenerateInit(L, transmission_, list_mixture_gauss_, data_seq, state_seq);
}

void MixtureofGaussianHMM::EstimateModel(int numcluster, const Matrix& data_seq, 
					 const Vector& state_seq) {
  transmission_.Destruct();
  list_mixture_gauss_.Renew();
  MixtureofGaussianHMM::EstimateInit(numcluster, data_seq, state_seq, &transmission_, 
				     &list_mixture_gauss_);
}

void MixtureofGaussianHMM::EstimateModel(int numstate, int numcluster, 
					 const Matrix& data_seq, const Vector& state_seq) {
  transmission_.Destruct();
  list_mixture_gauss_.Renew();
  MixtureofGaussianHMM::EstimateInit(numstate, numcluster, data_seq, state_seq, 
				     &transmission_, &list_mixture_gauss_);
}

void MixtureofGaussianHMM::DecodeOverwrite(const Matrix& data_seq, Matrix* state_prob_mat, 
					   Matrix* forward_prob_mat, 
					   Matrix* backward_prob_mat, Vector* scale_vec) const {
  int M = transmission_.n_rows();
  int L = data_seq.n_cols();  
  Matrix emission_prob_mat(M, L);
  MixtureofGaussianHMM::CalculateEmissionProb(data_seq, list_mixture_gauss_,
					      &emission_prob_mat);
  MixtureofGaussianHMM::Decode(transmission_, emission_prob_mat, state_prob_mat, 
			       forward_prob_mat, backward_prob_mat, scale_vec);
}

void MixtureofGaussianHMM::DecodeInit(const Matrix& data_seq, Matrix* state_prob_mat, 
				      Matrix* forward_prob_mat, Matrix* backward_prob_mat, 
				      Vector* scale_vec) const {
  int M = transmission_.n_rows();
  int L = data_seq.n_cols();  
  state_prob_mat->Init(M, L);
  forward_prob_mat->Init(M, L);
  backward_prob_mat->Init(M, L);
  scale_vec->Init(L);
  Matrix emission_prob_mat(M, L);
  MixtureofGaussianHMM::CalculateEmissionProb(data_seq, list_mixture_gauss_, 
					      &emission_prob_mat);
  MixtureofGaussianHMM::Decode(transmission_, emission_prob_mat, state_prob_mat, 
			       forward_prob_mat, backward_prob_mat, scale_vec);  
}

double MixtureofGaussianHMM::ComputeLogLikelihood(const Matrix& data_seq) const {
  int L = data_seq.n_cols();
  int M = transmission_.n_rows();
  Matrix fs(M, L), emis_prob(M, L);
  Vector sc;
  sc.Init(L);

  MixtureofGaussianHMM::CalculateEmissionProb(data_seq, list_mixture_gauss_, &emis_prob);
  MixtureofGaussianHMM::ForwardProcedure(L, transmission_, emis_prob, &sc, &fs);
  double loglik = 0;
  for (int t = 0; t < L; t++)
    loglik += log(sc[t]);
  return loglik;
}

void MixtureofGaussianHMM::ComputeLogLikelihood(const ArrayList<Matrix>& list_data_seq, ArrayList<double>* list_likelihood) const {
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
    MixtureofGaussianHMM::CalculateEmissionProb(list_data_seq[i], list_mixture_gauss_, &emis_prob);
    MixtureofGaussianHMM::ForwardProcedure(L, transmission_, emis_prob, &sc, &fs);
    double loglik = 0;
    for (int t = 0; t < L; t++)
      loglik += log(sc[t]);
    list_likelihood->PushBackCopy(loglik);
  }
}

void MixtureofGaussianHMM::ComputeViterbiStateSequence(const Matrix& data_seq, Vector* state_seq) const {
  int M = transmission_.n_rows();
  int L = data_seq.n_cols();
  Matrix emis_prob(M, L);
  MixtureofGaussianHMM::CalculateEmissionProb(data_seq, list_mixture_gauss_, &emis_prob);
  MixtureofGaussianHMM::ViterbiInit(transmission_, emis_prob, state_seq);
}

void MixtureofGaussianHMM::TrainBaumWelch(const ArrayList<Matrix>& list_data_seq, int max_iteration, double tolerance) {
  MixtureofGaussianHMM::Train(list_data_seq, &transmission_, &list_mixture_gauss_, max_iteration, tolerance);
}

void MixtureofGaussianHMM::TrainViterbi(const ArrayList<Matrix>& list_data_seq, int max_iteration, double tolerance) {
  MixtureofGaussianHMM::TrainViterbi(list_data_seq, &transmission_, &list_mixture_gauss_, max_iteration, tolerance);
}

success_t MixtureofGaussianHMM::LoadProfile(const char* profile, Matrix* trans, ArrayList<MixtureGauss>* mixs) {
  ArrayList<Matrix> matlst;
  if (!PASSED(load_matrix_list(profile, &matlst))) {
    NONFATAL("Couldn't open '%s' for reading.", profile);
    return SUCCESS_FAIL;
  }
  DEBUG_ASSERT(matlst.size() >= 4); // at least 1 trans, 1 prior, 1 mean, 1 cov
  trans->Copy(matlst[0]);
  mixs->Init();
  int M = trans->n_rows(); // num of states
  int N = matlst[2].n_rows(); // dimension
  int p = 1;
  for (int i = 0; i < M; i++) {
    int K = matlst[p].n_rows(); // num of clusters
    //DEBUG: printf("load p=%d K=%d\n", p, K);
    DEBUG_ASSERT(matlst.size() > p+2*K);
    MixtureGauss mix;
    mix.InitFromProfile(matlst, p, N);
    mixs->PushBackCopy(mix);
    p += 2*K+1;
  }
  return SUCCESS_PASS;
}

success_t MixtureofGaussianHMM::SaveProfile(const char* profile, const Matrix& trans, const ArrayList<MixtureGauss>& mixs) {
  TextWriter w_pro;
  if (!PASSED(w_pro.Open(profile))) {
    NONFATAL("Couldn't open '%s' for writing.", profile);
    return SUCCESS_FAIL;
  }
  int M = trans.n_rows(); // num of states
  print_matrix(w_pro, trans, "% transmission", "%E,");
  for (int i = 0; i < M; i++) {
    int K = mixs[i].n_clusters(); // num of clusters
    char s[100];
    sprintf(s, "%% prior - state %d", i);
    print_vector(w_pro, mixs[i].get_prior(), s, "%E,");
    for (int k=0; k < K; k++) {
      sprintf(s, "%% mean %d - state %d", k, i);
      print_vector(w_pro, mixs[i].get_mean(k), s, "%E,");
      sprintf(s, "%% covariance %d - state %d", k, i);
      print_matrix(w_pro, mixs[i].get_cov(k), s, "%E,");
    }
  }
  return SUCCESS_PASS;
}

void MixtureofGaussianHMM::GenerateInit(int L, const Matrix& trans, const ArrayList<MixtureGauss>& mixs, Matrix* seq, Vector* states){
  DEBUG_ASSERT_MSG((trans.n_rows()==trans.n_cols() && trans.n_rows()==mixs.size()), "hmm_generateM_init: matrices sizes do not match");
  Matrix trsum;
  Matrix& seq_ = *seq;
  Vector& states_ = *states;
  int M, N;
  int cur_state;

  M = trans.n_rows();
  N = mixs[0].v_length();  // emission vector length

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
    mixs[cur_state].generate(&e);
    for (j = 0; j < N; j++)
      seq_.ref(j, i) = e[j];
    states_[i] = cur_state;
  }
}

void MixtureofGaussianHMM::EstimateInit(int numStates, int numClusters, const Matrix& seq, const Vector& states, Matrix* trans, ArrayList<MixtureGauss>* mixs) {
  DEBUG_ASSERT_MSG((seq.n_cols()==states.length()), "hmm_estimateM_init: sequence and states length must be the same");
  
  int N = seq.n_rows(); // emission vector length
  int M = numStates;    // number of states
  int L = seq.n_cols(); // sequence length
  int K = numClusters;
  
  Matrix &trans_ = *trans;
  ArrayList<MixtureGauss>& mix_ = *mixs;
  Vector stateSum;

  trans_.Init(M, M);
  mix_.Init();
  stateSum.Init(M);

  trans_.SetZero();

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

  ArrayList<Matrix> data;
  data.Init();
  Vector n_data;
  n_data.Init(M); n_data.SetZero();
  for (int i = 0; i < L; i++) {
    int state = (int) states[i];
    n_data[state]++;
  }
  for (int i = 0; i < M; i++) {
    Matrix m;
    //printf("n[%d]=%8.0f\n", i, n_data[i]);
    m.Init(N, (int)n_data[i]);
    data.PushBackCopy(m);
  }
  n_data.SetZero();
  for (int i = 0; i < L; i++) {
    int state = (int) states[i];
    for (int j = 0; j < N; j++) data[state].ref(j, (int)n_data[state]) = seq.get(j, i);
    n_data[state]++;
    //printf("%d %d %8.0f\n", i, state, n_data[state]);
  }
  for (int i = 0; i < M; i++) {
    ArrayList<int> labels;
    ArrayList<Vector> means;
    kmeans(data[i], K, &labels, &means, 500, 1e-3);

    //printf("STATE #%d %d\n", i, K);
    MixtureGauss m;
    m.Init(K, data[i], labels);
    mix_.PushBackCopy(m);
  }
}

void MixtureofGaussianHMM::EstimateInit(int NumClusters, const Matrix& seq, const Vector& states, Matrix* trans, ArrayList<MixtureGauss>* mixs) {
  DEBUG_ASSERT_MSG((seq.n_cols()==states.length()), "hmm_estimateG_init: sequence and states length must be the same");
  int M = 0;
  for (int i = 0; i < seq.n_cols(); i++)
    if (states[i] > M) M = (int) states[i];
  M++;
  MixtureofGaussianHMM::EstimateInit(M, NumClusters, seq, states, trans, mixs);
}

void MixtureofGaussianHMM::ForwardProcedure(int L, const Matrix& trans, const Matrix& emis_prob, 
		      Vector *scales, Matrix* fs) {
  GaussianHMM::ForwardProcedure(L, trans, emis_prob, scales, fs);
}

void MixtureofGaussianHMM::BackwardProcedure(int L, const Matrix& trans, const Matrix& emis_prob, 
		       const Vector& scales, Matrix* bs) {
  GaussianHMM::BackwardProcedure(L, trans, emis_prob, scales, bs);
}

double MixtureofGaussianHMM::Decode(const Matrix& trans, const Matrix& emis_prob, Matrix* pstates, 
	      Matrix* fs, Matrix* bs, Vector* scales) {
  return GaussianHMM::Decode(trans, emis_prob, pstates, fs, bs, scales);
}

double MixtureofGaussianHMM::Decode(int L, const Matrix& trans, const Matrix& emis_prob, 
	      Matrix* pstates, Matrix* fs, Matrix* bs, Vector* scales) {
  return GaussianHMM::Decode(L, trans, emis_prob, pstates, fs, bs, scales);
}

double MixtureofGaussianHMM::ViterbiInit(const Matrix& trans, const Matrix& emis_prob, Vector* states) {
  return GaussianHMM::ViterbiInit(trans, emis_prob, states);
}

double MixtureofGaussianHMM::ViterbiInit(int L, const Matrix& trans, const Matrix& emis_prob, Vector* states) {
  return GaussianHMM::ViterbiInit(L, trans, emis_prob, states);
}

void MixtureofGaussianHMM::CalculateEmissionProb(const Matrix& seq, const ArrayList<MixtureGauss>& mixs, Matrix* emis_prob) {
  int M = mixs.size();
  int L = seq.n_cols();
  for (int t = 0; t < L; t++) {
    Vector e;
    seq.MakeColumnVector(t, &e);
    for (int i = 0; i < M; i++)
      emis_prob->ref(i, t) = mixs[i].getPDF(e);
  }
}

void MixtureofGaussianHMM::Train(const ArrayList<Matrix>& seqs, Matrix* guessTR, ArrayList<MixtureGauss>* guessMG, int max_iter, double tol) {
  Matrix &gTR = *guessTR;
  ArrayList<MixtureGauss>& gMG = *guessMG;
  int L = -1;
  int M = gTR.n_rows();
  DEBUG_ASSERT_MSG((M==gTR.n_cols() && M==gMG.size()),"hmm_trainM: sizes do not match");
  
  for (int i = 0; i < seqs.size(); i++)
    if (seqs[i].n_cols() > L) L = seqs[i].n_cols();

  Matrix TR; // guess transition and emission matrix
  TR.Init(M, M);

  Matrix ps, fs, bs, emis_prob; // to hold hmm_decodeG results
  ArrayList<Matrix> emis_prob_cluster;
  Vector s; // scaling factors
  Vector sumState; // the denominator for each state

  ps.Init(M, L);
  fs.Init(M, L);
  bs.Init(M, L);
  s.Init(L);
  emis_prob.Init(M, L);
  sumState.Init(M);
  emis_prob_cluster.Init();
  for (int i = 0; i < M; i++) {
    Matrix m;
    int K = gMG[i].n_clusters();
    m.Init(K, L);
    emis_prob_cluster.PushBackCopy(m);
  }

  double loglik = 0, oldlog;
  for (int iter = 0; iter < max_iter; iter++) {
    oldlog = loglik;
    loglik = 0;

    // set the accumulating values to zeros and compute the inverse matrices and determinant constants
    TR.SetZero();
    for (int i = 0; i < M; i++) 
      gMG[i].start_accumulate();

    sumState.SetZero();

    // for each sequence, we will use forward-backward procedure and then accumulate
    for (int idx = 0; idx < seqs.size(); idx++) {
      // first calculate the emission probabilities of the sequence
      L = seqs[idx].n_cols();
      for (int t = 0; t < L; t++) {
	Vector e;
	seqs[idx].MakeColumnVector(t, &e);
	for (int i = 0; i < M; i++) {
	  double s = 0;
	  int K = gMG[i].n_clusters();
	  for (int j = 0; j < K; j++) {
	    emis_prob_cluster[i].ref(j, t) = gMG[i].getPDF(j, e);
	    s += emis_prob_cluster[i].ref(j, t);
	  }
	  emis_prob.ref(i, t) = s;
	}
      }
      
      loglik += MixtureofGaussianHMM::Decode(L, gTR, emis_prob, &ps, &fs, &bs, &s); // forward - backward procedure
      
      // accumulate expected transition & gaussian mixture parameters
      for (int t = 0; t < L-1; t++) {
	for (int i = 0; i < M; i++)
	  for (int j = 0; j < M; j++)
	    TR.ref(i, j) += fs.get(i, t) * gTR.get(i, j) * emis_prob.get(j, t+1) * bs.get(j, t+1) / s[t+1];
      }
      
      for (int t = 0; t < L; t++) {
	Vector e;
	seqs[idx].MakeColumnVector(t, &e);
	for (int i = 0; i < M; i++) {
	  double v = ps.get(i, t);
	  int K = gMG[i].n_clusters();
	  for (int j = 0; j < K; j++) 
	    gMG[i].accumulate(v * emis_prob_cluster[i].get(j, t) / emis_prob.get(i, t), j, e);
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
      
      gMG[i].end_accumulate();
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

void MixtureofGaussianHMM::TrainViterbi(const ArrayList<Matrix>& seqs, Matrix* guessTR, ArrayList<MixtureGauss>* guessMG, int max_iter, double tol) {
  Matrix &gTR = *guessTR;
  ArrayList<MixtureGauss>& gMG = *guessMG;
  int L = -1;
  int M = gTR.n_rows();
  DEBUG_ASSERT_MSG((M==gTR.n_cols() && M==gMG.size()),"hmm_trainM: sizes do not match");
  
  for (int i = 0; i < seqs.size(); i++)
    if (seqs[i].n_cols() > L) L = seqs[i].n_cols();

  Matrix TR; // guess transition and emission matrix
  TR.Init(M, M);

  Matrix emis_prob; // to hold hmm_decodeG results
  ArrayList<Matrix> emis_prob_cluster;

  emis_prob.Init(M, L);
  emis_prob_cluster.Init();
  for (int i = 0; i < M; i++) {
    Matrix m;
    int K = gMG[i].n_clusters();
    m.Init(K, L);
    emis_prob_cluster.PushBackCopy(m);
  }

  double loglik = 0, oldlog;
  for (int iter = 0; iter < max_iter; iter++) {
    oldlog = loglik;
    loglik = 0;

    // set the accumulating values to zeros and compute the inverse matrices and determinant constants
    TR.SetZero();
    for (int i = 0; i < M; i++) 
      gMG[i].start_accumulate();

    // for each sequence, we will use viterbi procedure to find the most probable state sequence and then accumulate
    for (int idx = 0; idx < seqs.size(); idx++) {
      Vector states;
      // first calculate the emission probabilities of the sequence
      L = seqs[idx].n_cols();
      for (int t = 0; t < L; t++) {
	Vector e;
	seqs[idx].MakeColumnVector(t, &e);
	for (int i = 0; i < M; i++) {
	  double s = 0;
	  int K = gMG[i].n_clusters();
	  for (int j = 0; j < K; j++) {
	    emis_prob_cluster[i].ref(j, t) = gMG[i].getPDF(j, e);
	    s += emis_prob_cluster[i].ref(j, t);
	  }
	  emis_prob.ref(i, t) = s;
	}
      }
      
      loglik += GaussianHMM::ViterbiInit(L, gTR, emis_prob, &states); // viterbi procedure
      
      // accumulate expected transition & gaussian mixture parameters
      for (int t = 0; t < L-1; t++) {
	int i = (int) states[t];
	int j = (int) states[t+1];
	TR.ref(i, j)++;
      }
      
      for (int t = 0; t < L; t++) {
	Vector e;
	int i = (int) states[t];
	seqs[idx].MakeColumnVector(t, &e);
	int K = gMG[i].n_clusters();
	for (int j = 0; j < K; j++) 
	  gMG[i].accumulate(emis_prob_cluster[i].get(j, t) / emis_prob.get(i, t), j, e);
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
      
      gMG[i].end_accumulate();
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

