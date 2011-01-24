/**
 * @file mixgaussHMM.cc
 *
 * This file contains the implementation of functions in mixgaussHMM.h
 */

#include <fastlib/fastlib.h>
#include "support.h"
#include "mixgaussHMM.h"
#include "gaussianHMM.h"

using namespace hmm_support;

void MixtureofGaussianHMM::setModel(const arma::mat& transmission,
				    const std::vector<MixtureGauss>& list_mixture_gauss) {
  DEBUG_ASSERT(transmission.n_rows == transmission.n_cols);
  DEBUG_ASSERT(transmission.n_rows == list_mixture_gauss.size());

  transmission_ = transmission;
  list_mixture_gauss_ = list_mixture_gauss;
}

void MixtureofGaussianHMM::InitFromFile(const char* profile) {
  if (!PASSED(MixtureofGaussianHMM::LoadProfile(profile, transmission_, list_mixture_gauss_)))
    FATAL("Couldn't open '%s' for reading.", profile);
}

void MixtureofGaussianHMM::LoadProfile(const char* profile) {
  InitFromFile(profile);
}

void MixtureofGaussianHMM::SaveProfile(const char* profile) const {
  MixtureofGaussianHMM::SaveProfile(profile, transmission_, list_mixture_gauss_);
}

void MixtureofGaussianHMM::GenerateSequence(int L, arma::mat& data_seq, arma::vec& state_seq) const {
  MixtureofGaussianHMM::GenerateInit(L, transmission_, list_mixture_gauss_, data_seq, state_seq);
}

void MixtureofGaussianHMM::EstimateModel(int numcluster, const arma::mat& data_seq, 
					 const arma::vec& state_seq) {
  MixtureofGaussianHMM::EstimateInit(numcluster, data_seq, state_seq, transmission_,
				     list_mixture_gauss_);
}

void MixtureofGaussianHMM::EstimateModel(int numstate, int numcluster, 
					 const arma::mat& data_seq, const arma::vec& state_seq) {
  MixtureofGaussianHMM::EstimateInit(numstate, numcluster, data_seq, state_seq, 
				     transmission_, list_mixture_gauss_);
}

void MixtureofGaussianHMM::DecodeOverwrite(const arma::mat& data_seq, arma::mat& state_prob_mat, 
					   arma::mat& forward_prob_mat, 
					   arma::mat& backward_prob_mat, arma::vec& scale_vec) const {
  int M = transmission_.n_rows;
  int L = data_seq.n_cols;

  arma::mat emission_prob_mat(M, L);
  MixtureofGaussianHMM::CalculateEmissionProb(data_seq, list_mixture_gauss_,
					      emission_prob_mat);
  MixtureofGaussianHMM::Decode(transmission_, emission_prob_mat, state_prob_mat, 
			       forward_prob_mat, backward_prob_mat, scale_vec);
}

void MixtureofGaussianHMM::DecodeInit(const arma::mat& data_seq, arma::mat& state_prob_mat, 
				      arma::mat& forward_prob_mat, arma::mat& backward_prob_mat, 
				      arma::vec& scale_vec) const {
  int M = transmission_.n_rows;
  int L = data_seq.n_cols;
  state_prob_mat.set_size(M, L);
  forward_prob_mat.set_size(M, L);
  backward_prob_mat.set_size(M, L);
  scale_vec.set_size(L);

  arma::mat emission_prob_mat(M, L);

  MixtureofGaussianHMM::CalculateEmissionProb(data_seq, list_mixture_gauss_, 
					      emission_prob_mat);
  MixtureofGaussianHMM::Decode(transmission_, emission_prob_mat, state_prob_mat, 
			       forward_prob_mat, backward_prob_mat, scale_vec);  
}

double MixtureofGaussianHMM::ComputeLogLikelihood(const arma::mat& data_seq) const {
  int L = data_seq.n_cols;
  int M = transmission_.n_rows;
  arma::mat fs(M, L), emis_prob(M, L);
  arma::vec sc(L);

  MixtureofGaussianHMM::CalculateEmissionProb(data_seq, list_mixture_gauss_, emis_prob);
  MixtureofGaussianHMM::ForwardProcedure(L, transmission_, emis_prob, sc, fs);

  double loglik = 0;
  for (int t = 0; t < L; t++)
    loglik += log(sc[t]);
  return loglik;
}

void MixtureofGaussianHMM::ComputeLogLikelihood(const std::vector<arma::mat>& list_data_seq, std::vector<double>& list_likelihood) const {
  int L = 0;
  for (int i = 0; i < list_data_seq.size(); i++) {
    if (list_data_seq[i].n_cols > L)
      L = list_data_seq[i].n_cols;
  }
  int M = transmission_.n_rows;
  arma::mat fs(M, L), emis_prob(M, L);
  arma::vec sc(L);

  for (int i = 0; i < list_data_seq.size(); i++) {
    int L = list_data_seq[i].n_cols;

    MixtureofGaussianHMM::CalculateEmissionProb(list_data_seq[i], list_mixture_gauss_, emis_prob);
    MixtureofGaussianHMM::ForwardProcedure(L, transmission_, emis_prob, sc, fs);

    double loglik = 0;
    for (int t = 0; t < L; t++)
      loglik += log(sc[t]);
    list_likelihood.push_back(loglik);
  }
}

void MixtureofGaussianHMM::ComputeViterbiStateSequence(const arma::mat& data_seq, arma::vec& state_seq) const {
  int M = transmission_.n_rows;
  int L = data_seq.n_cols;
  arma::mat emis_prob(M, L);
  MixtureofGaussianHMM::CalculateEmissionProb(data_seq, list_mixture_gauss_, emis_prob);
  MixtureofGaussianHMM::ViterbiInit(transmission_, emis_prob, state_seq);
}

void MixtureofGaussianHMM::TrainBaumWelch(const std::vector<arma::mat>& list_data_seq, int max_iteration, double tolerance) {
  MixtureofGaussianHMM::Train(list_data_seq, transmission_, list_mixture_gauss_, max_iteration, tolerance);
}

void MixtureofGaussianHMM::TrainViterbi(const std::vector<arma::mat>& list_data_seq, int max_iteration, double tolerance) {
  MixtureofGaussianHMM::TrainViterbi(list_data_seq, transmission_, list_mixture_gauss_, max_iteration, tolerance);
}

success_t MixtureofGaussianHMM::LoadProfile(const char* profile, arma::mat& trans, std::vector<MixtureGauss>& mixs) {
  std::vector<arma::mat> matlst;
  if (!PASSED(load_matrix_list(profile, matlst))) {
    NONFATAL("Couldn't open '%s' for reading.", profile);
    return SUCCESS_FAIL;
  }
  DEBUG_ASSERT(matlst.size() >= 4); // at least 1 trans, 1 prior, 1 mean, 1 cov
  trans = matlst[0];
  int M = trans.n_rows; // num of states
  int N = matlst[2].n_rows; // dimension
  int p = 1;
  for (int i = 0; i < M; i++) {
    int K = matlst[p].n_rows; // num of clusters
    //DEBUG: printf("load p=%d K=%d\n", p, K);
    DEBUG_ASSERT(matlst.size() > p + 2 * K);
    MixtureGauss mix;
    mix.InitFromProfile(matlst, p, N);
    mixs.push_back(mix);
    p += (2 * K + 1);
  }

  return SUCCESS_PASS;
}

success_t MixtureofGaussianHMM::SaveProfile(const char* profile, const arma::mat& trans, const std::vector<MixtureGauss>& mixs) {
  TextWriter w_pro;
  if (!PASSED(w_pro.Open(profile))) {
    NONFATAL("Couldn't open '%s' for writing.", profile);
    return SUCCESS_FAIL;
  }
  int M = trans.n_rows; // num of states
  print_matrix(w_pro, trans, "% transmission", "%E,");
  for (int i = 0; i < M; i++) {
    int K = mixs[i].n_clusters(); // num of clusters
    char s[100];
    sprintf(s, "%% prior - state %d", i);
    print_vector(w_pro, mixs[i].get_prior(), s, "%E,");
    for (int k = 0; k < K; k++) {
      sprintf(s, "%% mean %d - state %d", k, i);
      print_vector(w_pro, mixs[i].get_mean(k), s, "%E,");
      sprintf(s, "%% covariance %d - state %d", k, i);
      print_matrix(w_pro, mixs[i].get_cov(k), s, "%E,");
    }
  }

  return SUCCESS_PASS;
}

void MixtureofGaussianHMM::GenerateInit(int L, const arma::mat& trans, const std::vector<MixtureGauss>& mixs, arma::mat& seq, arma::vec& states){
  DEBUG_ASSERT_MSG((trans.n_rows == trans.n_cols && trans.n_rows == mixs.size()),
      "MixtureOfGaussianHMM::GenerateInit(): matrices sizes do not match");

  arma::mat trsum;
  int M, N;
  int cur_state;

  M = trans.n_rows;
  N = mixs[0].v_length();  // emission vector length

  trsum = trans;

  for (int i = 0; i < M; i++) {
    for (int j = 1; j < M; j++)
      trsum(i, j) += trsum(i, j - 1);
  }

  seq.set_size(N, L);
  states.set_size(L);

  cur_state = 0; // starting state is 0
  
  for (int i = 0; i < L; i++) {
    int j;

    // next state
    double r = RAND_UNIFORM_01();
    for (j = 0; j < M; j++) {
      if (r <= trsum(cur_state, j))
        break;
    }
    cur_state = j;
	
    // emission
    arma::vec e;
    mixs[cur_state].generate(e);
    for (j = 0; j < N; j++)
      seq(j, i) = e[j];
    states[i] = cur_state;
  }
}

void MixtureofGaussianHMM::EstimateInit(int numStates, int numClusters, const arma::mat& seq, const arma::vec& states, arma::mat& trans, std::vector<MixtureGauss>& mixs) {
  DEBUG_ASSERT_MSG((seq.n_cols == states.n_elem),
      "MixtureOfGaussianHMM::EstimateInit(): sequence and states length must be the same");
  
  int N = seq.n_rows; // emission vector length
  int M = numStates;  // number of states
  int L = seq.n_cols; // sequence length
  int K = numClusters;
  
  trans.zeros(M, M);
  arma::vec stateSum(M);
  stateSum.zeros();

  for (int i = 0; i < L - 1; i++) {
    int state = (int) states[i];
    int next_state = (int) states[i + 1];
    stateSum[state]++;
    trans(state, next_state)++;
  }

  for (int i = 0; i < M; i++) {
    if (stateSum[i] == 0)
      stateSum[i] = -INFINITY;

    for (int j = 0; j < M; j++)
      trans(i, j) /= stateSum[i];
  }

  std::vector<arma::mat> data;
  arma::vec n_data(M);
  n_data.zeros();

  for (int i = 0; i < L; i++) {
    int state = (int) states[i];
    n_data[state]++;
  }

  for (int i = 0; i < M; i++) {
    arma::mat m(N, (int) n_data[i]);
    //printf("n[%d]=%8.0f\n", i, n_data[i]);
    data.push_back(m);
  }

  n_data.zeros();
  for (int i = 0; i < L; i++) {
    int state = (int) states[i];
    for (int j = 0; j < N; j++)
      data[state](j, (int) n_data[state]) = seq(j, i);

    n_data[state]++;
    //printf("%d %d %8.0f\n", i, state, n_data[state]);
  }

  for (int i = 0; i < M; i++) {
    std::vector<int> labels;
    std::vector<arma::vec> means;
    kmeans(data[i], K, labels, means, 500, 1e-3);

    //printf("STATE #%d %d\n", i, K);
    MixtureGauss m;
    m.Init(K, data[i], labels);
    mixs.push_back(m);
  }
}

void MixtureofGaussianHMM::EstimateInit(int NumClusters, const arma::mat& seq, const arma::vec& states, arma::mat& trans, std::vector<MixtureGauss>& mixs) {
  DEBUG_ASSERT_MSG((seq.n_cols == states.n_elem),
      "MixtureofGaussianHMM::EstimateInit(): sequence and states length must be the same");

  int M = 0;
  for (int i = 0; i < seq.n_cols; i++) {
    if (states[i] > M)
      M = (int) states[i];
  }
  M++;
  MixtureofGaussianHMM::EstimateInit(M, NumClusters, seq, states, trans, mixs);
}

void MixtureofGaussianHMM::ForwardProcedure(int L, const arma::mat& trans, const arma::mat& emis_prob, arma::vec& scales, arma::mat& fs) {
  GaussianHMM::ForwardProcedure(L, trans, emis_prob, scales, fs);
}

void MixtureofGaussianHMM::BackwardProcedure(int L, const arma::mat& trans, const arma::mat& emis_prob, const arma::vec& scales, arma::mat& bs) {
  GaussianHMM::BackwardProcedure(L, trans, emis_prob, scales, bs);
}

double MixtureofGaussianHMM::Decode(const arma::mat& trans, const arma::mat& emis_prob, arma::mat& pstates, arma::mat& fs, arma::mat& bs, arma::vec& scales) {
  return GaussianHMM::Decode(trans, emis_prob, pstates, fs, bs, scales);
}

double MixtureofGaussianHMM::Decode(int L, const arma::mat& trans, const arma::mat& emis_prob, arma::mat& pstates, arma::mat& fs, arma::mat& bs, arma::vec& scales) {
  return GaussianHMM::Decode(L, trans, emis_prob, pstates, fs, bs, scales);
}

double MixtureofGaussianHMM::ViterbiInit(const arma::mat& trans, const arma::mat& emis_prob, arma::vec& states) {
  return GaussianHMM::ViterbiInit(trans, emis_prob, states);
}

double MixtureofGaussianHMM::ViterbiInit(int L, const arma::mat& trans, const arma::mat& emis_prob, arma::vec& states) {
  return GaussianHMM::ViterbiInit(L, trans, emis_prob, states);
}

void MixtureofGaussianHMM::CalculateEmissionProb(const arma::mat& seq, const std::vector<MixtureGauss>& mixs, arma::mat& emis_prob) {
  int M = mixs.size();
  int L = seq.n_cols;
  for (int t = 0; t < L; t++) {
    arma::vec e = seq.unsafe_col(t);
    
    for (int i = 0; i < M; i++)
      emis_prob(i, t) = mixs[i].getPDF(e);
  }
}

void MixtureofGaussianHMM::Train(const std::vector<arma::mat>& seqs, arma::mat& guessTR, std::vector<MixtureGauss>& guessMG, int max_iter, double tol) {
  int L = -1;
  int M = guessTR.n_rows;

  DEBUG_ASSERT_MSG((M == guessTR.n_cols && M == guessMG.size()),
      "MixtureofGaussianHMM::Train(): sizes do not match");
  
  for (int i = 0; i < seqs.size(); i++) {
    if (seqs[i].n_cols > L)
      L = seqs[i].n_cols;
  }

  arma::mat TR(M, M); // guess transition and emission matrix

  arma::mat ps(M, L), fs(M, L), bs(M, L), emis_prob(M, L); // to hold hmm_decodeG results
  std::vector<arma::mat> emis_prob_cluster;
  arma::vec s(L); // scaling factors
  arma::vec sumState(M); // the denominator for each state

  for (int i = 0; i < M; i++) {
    arma::mat m;
    int K = guessMG[i].n_clusters();
    m.set_size(K, L);
    emis_prob_cluster.push_back(m);
  }

  double loglik = 0, oldlog;
  for (int iter = 0; iter < max_iter; iter++) {
    oldlog = loglik;
    loglik = 0;

    // set the accumulating values to zeros and compute the inverse matrices and determinant constants
    TR.zeros();
    for (int i = 0; i < M; i++) 
      guessMG[i].start_accumulate();

    sumState.zeros();

    // for each sequence, we will use forward-backward procedure and then accumulate
    for (int idx = 0; idx < seqs.size(); idx++) {
      // first calculate the emission probabilities of the sequence
      L = seqs[idx].n_cols;
      for (int t = 0; t < L; t++) {
	arma::vec e = seqs[idx].unsafe_col(t);
	for (int i = 0; i < M; i++) {
	  double s = 0;
	  int K = guessMG[i].n_clusters();
	  for (int j = 0; j < K; j++) {
	    emis_prob_cluster[i](j, t) = guessMG[i].getPDF(j, e);
	    s += emis_prob_cluster[i](j, t);
	  }
	  emis_prob(i, t) = s;
	}
      }
      
      loglik += MixtureofGaussianHMM::Decode(L, guessTR, emis_prob, ps, fs, bs, s); // forward - backward procedure
      
      // accumulate expected transition & gaussian mixture parameters
      for (int t = 0; t < L-1; t++) {
	for (int i = 0; i < M; i++)
	  for (int j = 0; j < M; j++)
	    TR(i, j) += fs(i, t) * guessTR(i, j) * emis_prob(j, t + 1) * bs(j, t + 1) / s[t + 1];
      }
      
      for (int t = 0; t < L; t++) {
	arma::vec e = seqs[idx].unsafe_col(t);
	for (int i = 0; i < M; i++) {
	  double v = ps(i, t);
	  int K = guessMG[i].n_clusters();
	  for (int j = 0; j < K; j++) 
	    guessMG[i].accumulate(v * emis_prob_cluster[i](j, t) / emis_prob(i, t), j, e);
	}
      }
      // end accumulate
    }

    // after accumulate all sequences: re-estimate transition & mean & covariance for the next iteration
    for (int i = 0; i < M; i++) {
      double s = 0;
      for (int j = 0; j < M; j++)
        s += TR(i, j);

      if (s == 0) {
	for (int j = 0; j < M; j++)
          guessTR(i, j) = 0;
	guessTR(i, i) = 1;
      } else {
	for (int j = 0; j < M; j++)
          guessTR(i, j) = TR(i, j) / s;
      }
      
      guessMG[i].end_accumulate();
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

void MixtureofGaussianHMM::TrainViterbi(const std::vector<arma::mat>& seqs, arma::mat& guessTR, std::vector<MixtureGauss>& guessMG, int max_iter, double tol) {
  int L = -1;
  int M = guessTR.n_rows;
  DEBUG_ASSERT_MSG((M == guessTR.n_cols && M == guessMG.size()),
      "MixtureofGaussianHMM::TrainViterbi(): sizes do not match");
  
  for (int i = 0; i < seqs.size(); i++)
    if (seqs[i].n_cols > L)
      L = seqs[i].n_cols;

  arma::mat TR(M, M); // guess transition and emission matrix

  arma::mat emis_prob(M, L); // to hold hmm_decodeG results
  std::vector<arma::mat> emis_prob_cluster;

  for (int i = 0; i < M; i++) {
    arma::mat m;
    int K = guessMG[i].n_clusters();
    m.set_size(K, L);
    emis_prob_cluster.push_back(m);
  }

  double loglik = 0, oldlog;
  for (int iter = 0; iter < max_iter; iter++) {
    oldlog = loglik;
    loglik = 0;

    // set the accumulating values to zeros and compute the inverse matrices and determinant constants
    TR.zeros();
    for (int i = 0; i < M; i++) 
      guessMG[i].start_accumulate();

    // for each sequence, we will use viterbi procedure to find the most probable state sequence and then accumulate
    for (int idx = 0; idx < seqs.size(); idx++) {
      arma::vec states;
      // first calculate the emission probabilities of the sequence
      L = seqs[idx].n_cols;
      for (int t = 0; t < L; t++) {
	arma::vec e = seqs[idx].unsafe_col(t);
	for (int i = 0; i < M; i++) {
	  double s = 0;
	  int K = guessMG[i].n_clusters();
	  for (int j = 0; j < K; j++) {
	    emis_prob_cluster[i](j, t) = guessMG[i].getPDF(j, e);
	    s += emis_prob_cluster[i](j, t);
	  }
	  emis_prob(i, t) = s;
	}
      }
      
      loglik += GaussianHMM::ViterbiInit(L, guessTR, emis_prob, states); // viterbi procedure
      
      // accumulate expected transition & gaussian mixture parameters
      for (int t = 0; t < L-1; t++) {
	int i = (int) states[t];
	int j = (int) states[t + 1];
	TR(i, j)++;
      }
      
      for (int t = 0; t < L; t++) {
	arma::vec e = seqs[idx].unsafe_col(t);
	int i = (int) states[t];
	int K = guessMG[i].n_clusters();
	for (int j = 0; j < K; j++) 
	  guessMG[i].accumulate(emis_prob_cluster[i](j, t) / emis_prob(i, t), j, e);
      }
      // end accumulate
    }

    // after accumulate all sequences: re-estimate transition & mean & covariance for the next iteration
    for (int i = 0; i < M; i++) {
      double s = 0;
      for (int j = 0; j < M; j++)
        s += TR(i, j);

      if (s == 0) {
	for (int j = 0; j < M; j++)
          guessTR(i, j) = 0;
	guessTR(i, i) = 1;
      }
      else {
	for (int j = 0; j < M; j++)
          guessTR(i, j) = TR(i, j) / s;
      }
      
      guessMG[i].end_accumulate();
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

