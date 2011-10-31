/**
 * @file mixgaussHMM.cc
 *
 * This file contains the implementation of functions in mixgaussHMM.h
 */
#include <mlpack/core.h>

#include "support.hpp"
#include "mixgaussHMM.hpp"
#include "gaussianHMM.hpp"

using namespace mlpack;
using namespace hmm;
using namespace hmm_support;

void MixtureofGaussianHMM::setModel(const arma::mat& transmission,
				    const std::vector<MixtureGauss>& list_mixture_gauss) {
  mlpack::Log::Assert(transmission.n_rows == transmission.n_cols);
  mlpack::Log::Assert(transmission.n_rows == list_mixture_gauss.size());

  transmission_ = transmission;
  list_mixture_gauss_ = list_mixture_gauss;
}

void MixtureofGaussianHMM::InitFromFile(const char* profile) {
  if (!(MixtureofGaussianHMM::LoadProfile(profile, transmission_, list_mixture_gauss_)))
    mlpack::Log::Fatal << "Couldn't open " << profile << " for reading." <<
        std::endl;
}

void MixtureofGaussianHMM::LoadProfile(const char* profile) {
  InitFromFile(profile);
}

void MixtureofGaussianHMM::SaveProfile(const char* profile) const {
  MixtureofGaussianHMM::SaveProfile(profile, transmission_, list_mixture_gauss_);
}

void MixtureofGaussianHMM::GenerateSequence(size_t L, arma::mat& data_seq, arma::vec& state_seq) const {
  MixtureofGaussianHMM::GenerateInit(L, transmission_, list_mixture_gauss_, data_seq, state_seq);
}

void MixtureofGaussianHMM::EstimateModel(size_t numcluster, const arma::mat& data_seq,
					 const arma::vec& state_seq) {
  MixtureofGaussianHMM::EstimateInit(numcluster, data_seq, state_seq, transmission_,
				     list_mixture_gauss_);
}

void MixtureofGaussianHMM::EstimateModel(size_t numstate, size_t numcluster,
					 const arma::mat& data_seq, const arma::vec& state_seq) {
  MixtureofGaussianHMM::EstimateInit(numstate, numcluster, data_seq, state_seq,
				     transmission_, list_mixture_gauss_);
}

void MixtureofGaussianHMM::DecodeOverwrite(const arma::mat& data_seq, arma::mat& state_prob_mat,
					   arma::mat& forward_prob_mat,
					   arma::mat& backward_prob_mat, arma::vec& scale_vec) const {
  size_t M = transmission_.n_rows;
  size_t L = data_seq.n_cols;

  arma::mat emission_prob_mat(M, L);
  MixtureofGaussianHMM::CalculateEmissionProb(data_seq, list_mixture_gauss_,
					      emission_prob_mat);
  MixtureofGaussianHMM::Decode(transmission_, emission_prob_mat, state_prob_mat,
			       forward_prob_mat, backward_prob_mat, scale_vec);
}

void MixtureofGaussianHMM::DecodeInit(const arma::mat& data_seq, arma::mat& state_prob_mat,
				      arma::mat& forward_prob_mat, arma::mat& backward_prob_mat,
				      arma::vec& scale_vec) const {
  size_t M = transmission_.n_rows;
  size_t L = data_seq.n_cols;
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
  size_t L = data_seq.n_cols;
  size_t M = transmission_.n_rows;
  arma::mat fs(M, L), emis_prob(M, L);
  arma::vec sc(L);

  MixtureofGaussianHMM::CalculateEmissionProb(data_seq, list_mixture_gauss_, emis_prob);
  MixtureofGaussianHMM::ForwardProcedure(L, transmission_, emis_prob, sc, fs);

  double loglik = 0;
  for (size_t t = 0; t < L; t++)
    loglik += log(sc[t]);
  return loglik;
}

void MixtureofGaussianHMM::ComputeLogLikelihood(const std::vector<arma::mat>& list_data_seq, std::vector<double>& list_likelihood) const {
  size_t L = 0;
  for (size_t i = 0; i < list_data_seq.size(); i++) {
    if (list_data_seq[i].n_cols > L)
      L = list_data_seq[i].n_cols;
  }
  size_t M = transmission_.n_rows;
  arma::mat fs(M, L), emis_prob(M, L);
  arma::vec sc(L);

  for (size_t i = 0; i < list_data_seq.size(); i++) {
    size_t L = list_data_seq[i].n_cols;

    MixtureofGaussianHMM::CalculateEmissionProb(list_data_seq[i], list_mixture_gauss_, emis_prob);
    MixtureofGaussianHMM::ForwardProcedure(L, transmission_, emis_prob, sc, fs);

    double loglik = 0;
    for (size_t t = 0; t < L; t++)
      loglik += log(sc[t]);
    list_likelihood.push_back(loglik);
  }
}

void MixtureofGaussianHMM::ComputeViterbiStateSequence(const arma::mat& data_seq, arma::vec& state_seq) const {
  size_t M = transmission_.n_rows;
  size_t L = data_seq.n_cols;
  arma::mat emis_prob(M, L);
  MixtureofGaussianHMM::CalculateEmissionProb(data_seq, list_mixture_gauss_, emis_prob);
  MixtureofGaussianHMM::ViterbiInit(transmission_, emis_prob, state_seq);
}

void MixtureofGaussianHMM::TrainBaumWelch(const std::vector<arma::mat>& list_data_seq, size_t max_iteration, double tolerance) {
  MixtureofGaussianHMM::Train(list_data_seq, transmission_, list_mixture_gauss_, max_iteration, tolerance);
}

void MixtureofGaussianHMM::TrainViterbi(const std::vector<arma::mat>& list_data_seq, size_t max_iteration, double tolerance) {
  MixtureofGaussianHMM::TrainViterbi(list_data_seq, transmission_, list_mixture_gauss_, max_iteration, tolerance);
}

bool MixtureofGaussianHMM::LoadProfile(const char* profile, arma::mat& trans, std::vector<MixtureGauss>& mixs) {
  std::vector<arma::mat> matlst;
  if (!(load_matrix_list(profile, matlst))) {
    mlpack::Log::Warn << "Couldn't open " << profile << " for reading." <<
        std::endl;
    return false;
  }
  mlpack::Log::Assert(matlst.size() >= 4); // at least 1 trans, 1 prior, 1 mean, 1 cov
  trans = matlst[0];
  size_t M = trans.n_rows; // num of states
  size_t N = matlst[2].n_rows; // dimension
  size_t p = 1;
  for (size_t i = 0; i < M; i++) {
    size_t K = matlst[p].n_rows; // num of clusters
    //DEBUG: printf("load p=%d K=%d\n", p, K);
    mlpack::Log::Assert(matlst.size() > p + 2 * K);
    MixtureGauss mix;
    mix.InitFromProfile(matlst, p, N);
    mixs.push_back(mix);
    p += (2 * K + 1);
  }

  return true;
}

bool MixtureofGaussianHMM::SaveProfile(const char* profile, const arma::mat& trans, const std::vector<MixtureGauss>& mixs) {
  /** need something better
  TextWriter w_pro;
  if (!(w_pro.Open(profile))) {
    mlpack::Log::Warn << "Couldn't open " << profile << " for writing." <<
        std::endl;
    return false;
  }
  size_t M = trans.n_rows; // num of states
  print_matrix(w_pro, trans, "% transmission", "%E,");
  for (size_t i = 0; i < M; i++) {
    size_t K = mixs[i].n_clusters(); // num of clusters
    char s[100];
    sprintf(s, "%% prior - state %zu", i);
    print_vector(w_pro, mixs[i].get_prior(), s, "%E,");
    for (size_t k = 0; k < K; k++) {
      sprintf(s, "%% mean %zu - state %zu", k, i);
      print_vector(w_pro, mixs[i].get_mean(k), s, "%E,");
      sprintf(s, "%% covariance %zu - state %zu", k, i);
      print_matrix(w_pro, mixs[i].get_cov(k), s, "%E,");
    }
  }
  */

  return true;
}

void MixtureofGaussianHMM::GenerateInit(size_t L, const arma::mat& trans, const std::vector<MixtureGauss>& mixs, arma::mat& seq, arma::vec& states){
  mlpack::Log::Assert((trans.n_rows == trans.n_cols && trans.n_rows == mixs.size()),
      "MixtureOfGaussianHMM::GenerateInit(): matrices sizes do not match");

  arma::mat trsum;
  size_t M, N;
  size_t cur_state;

  M = trans.n_rows;
  N = mixs[0].v_length();  // emission vector length

  trsum = trans;

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 1; j < M; j++)
      trsum(i, j) += trsum(i, j - 1);
  }

  seq.set_size(N, L);
  states.set_size(L);

  cur_state = 0; // starting state is 0

  for (size_t i = 0; i < L; i++) {
    size_t j;

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

void MixtureofGaussianHMM::EstimateInit(size_t numStates, size_t numClusters, const arma::mat& seq, const arma::vec& states, arma::mat& trans, std::vector<MixtureGauss>& mixs) {
  mlpack::Log::Assert((seq.n_cols == states.n_elem),
      "MixtureOfGaussianHMM::EstimateInit(): sequence and states length must be the same");

  size_t N = seq.n_rows; // emission vector length
  size_t M = numStates;  // number of states
  size_t L = seq.n_cols; // sequence length
  size_t K = numClusters;

  trans.zeros(M, M);
  arma::vec stateSum(M);
  stateSum.zeros();

  for (size_t i = 0; i < L - 1; i++) {
    size_t state = (size_t) states[i];
    size_t next_state = (size_t) states[i + 1];
    stateSum[state]++;
    trans(state, next_state)++;
  }

  for (size_t i = 0; i < M; i++) {
    if (stateSum[i] == 0)
      stateSum[i] = -INFINITY;

    for (size_t j = 0; j < M; j++)
      trans(i, j) /= stateSum[i];
  }

  std::vector<arma::mat> data;
  arma::vec n_data(M);
  n_data.zeros();

  for (size_t i = 0; i < L; i++) {
    size_t state = (size_t) states[i];
    n_data[state]++;
  }

  for (size_t i = 0; i < M; i++) {
    arma::mat m(N, (size_t) n_data[i]);
    //printf("n[%d]=%8.0f\n", i, n_data[i]);
    data.push_back(m);
  }

  n_data.zeros();
  for (size_t i = 0; i < L; i++) {
    size_t state = (size_t) states[i];
    for (size_t j = 0; j < N; j++)
      data[state](j, (size_t) n_data[state]) = seq(j, i);

    n_data[state]++;
    //printf("%d %d %8.0f\n", i, state, n_data[state]);
  }

  for (size_t i = 0; i < M; i++) {
    std::vector<size_t> labels;
    std::vector<arma::vec> means;
    kmeans(data[i], K, labels, means, 500, 1e-3);

    //printf("STATE #%d %d\n", i, K);
    MixtureGauss m;
    m.Init(K, data[i], labels);
    mixs.push_back(m);
  }
}

void MixtureofGaussianHMM::EstimateInit(size_t NumClusters, const arma::mat& seq, const arma::vec& states, arma::mat& trans, std::vector<MixtureGauss>& mixs) {
  mlpack::Log::Assert((seq.n_cols == states.n_elem),
      "MixtureofGaussianHMM::EstimateInit(): sequence and states length must be the same");

  size_t M = 0;
  for (size_t i = 0; i < seq.n_cols; i++) {
    if (states[i] > M)
      M = (size_t) states[i];
  }
  M++;
  MixtureofGaussianHMM::EstimateInit(M, NumClusters, seq, states, trans, mixs);
}

void MixtureofGaussianHMM::ForwardProcedure(size_t L, const arma::mat& trans, const arma::mat& emis_prob, arma::vec& scales, arma::mat& fs) {
  GaussianHMM::ForwardProcedure(L, trans, emis_prob, scales, fs);
}

void MixtureofGaussianHMM::BackwardProcedure(size_t L, const arma::mat& trans, const arma::mat& emis_prob, const arma::vec& scales, arma::mat& bs) {
  GaussianHMM::BackwardProcedure(L, trans, emis_prob, scales, bs);
}

double MixtureofGaussianHMM::Decode(const arma::mat& trans, const arma::mat& emis_prob, arma::mat& pstates, arma::mat& fs, arma::mat& bs, arma::vec& scales) {
  return GaussianHMM::Decode(trans, emis_prob, pstates, fs, bs, scales);
}

double MixtureofGaussianHMM::Decode(size_t L, const arma::mat& trans, const arma::mat& emis_prob, arma::mat& pstates, arma::mat& fs, arma::mat& bs, arma::vec& scales) {
  return GaussianHMM::Decode(L, trans, emis_prob, pstates, fs, bs, scales);
}

double MixtureofGaussianHMM::ViterbiInit(const arma::mat& trans, const arma::mat& emis_prob, arma::vec& states) {
  return GaussianHMM::ViterbiInit(trans, emis_prob, states);
}

double MixtureofGaussianHMM::ViterbiInit(size_t L, const arma::mat& trans, const arma::mat& emis_prob, arma::vec& states) {
  return GaussianHMM::ViterbiInit(L, trans, emis_prob, states);
}

void MixtureofGaussianHMM::CalculateEmissionProb(const arma::mat& seq, const std::vector<MixtureGauss>& mixs, arma::mat& emis_prob) {
  size_t M = mixs.size();
  size_t L = seq.n_cols;
  for (size_t t = 0; t < L; t++) {
    arma::vec e = seq.unsafe_col(t);

    for (size_t i = 0; i < M; i++)
      emis_prob(i, t) = mixs[i].getPDF(e);
  }
}

void MixtureofGaussianHMM::Train(const std::vector<arma::mat>& seqs, arma::mat& guessTR, std::vector<MixtureGauss>& guessMG, size_t max_iter, double tol) {
  size_t L = -1;
  size_t M = guessTR.n_rows;

  mlpack::Log::Assert((M == guessTR.n_cols && M == guessMG.size()),
      "MixtureofGaussianHMM::Train(): sizes do not match");

  for (size_t i = 0; i < seqs.size(); i++) {
    if (seqs[i].n_cols > L)
      L = seqs[i].n_cols;
  }

  arma::mat TR(M, M); // guess transition and emission matrix

  arma::mat ps(M, L), fs(M, L), bs(M, L), emis_prob(M, L); // to hold hmm_decodeG results
  std::vector<arma::mat> emis_prob_cluster;
  arma::vec s(L); // scaling factors
  arma::vec sumState(M); // the denominator for each state

  for (size_t i = 0; i < M; i++) {
    arma::mat m;
    size_t K = guessMG[i].n_clusters();
    m.set_size(K, L);
    emis_prob_cluster.push_back(m);
  }

  double loglik = 0, oldlog;
  for (size_t iter = 0; iter < max_iter; iter++) {
    oldlog = loglik;
    loglik = 0;

    // set the accumulating values to zeros and compute the inverse matrices and determinant constants
    TR.zeros();
    for (size_t i = 0; i < M; i++)
      guessMG[i].start_accumulate();

    sumState.zeros();

    // for each sequence, we will use forward-backward procedure and then accumulate
    for (size_t idx = 0; idx < seqs.size(); idx++) {
      // first calculate the emission probabilities of the sequence
      L = seqs[idx].n_cols;
      for (size_t t = 0; t < L; t++) {
	arma::vec e = seqs[idx].unsafe_col(t);
	for (size_t i = 0; i < M; i++) {
	  double s = 0;
	  size_t K = guessMG[i].n_clusters();
	  for (size_t j = 0; j < K; j++) {
	    emis_prob_cluster[i](j, t) = guessMG[i].getPDF(j, e);
	    s += emis_prob_cluster[i](j, t);
	  }
	  emis_prob(i, t) = s;
	}
      }

      loglik += MixtureofGaussianHMM::Decode(L, guessTR, emis_prob, ps, fs, bs, s); // forward - backward procedure

      // accumulate expected transition & gaussian mixture parameters
      for (size_t t = 0; t < L-1; t++) {
	for (size_t i = 0; i < M; i++)
	  for (size_t j = 0; j < M; j++)
	    TR(i, j) += fs(i, t) * guessTR(i, j) * emis_prob(j, t + 1) * bs(j, t + 1) / s[t + 1];
      }

      for (size_t t = 0; t < L; t++) {
	arma::vec e = seqs[idx].unsafe_col(t);
	for (size_t i = 0; i < M; i++) {
	  double v = ps(i, t);
	  size_t K = guessMG[i].n_clusters();
	  for (size_t j = 0; j < K; j++)
	    guessMG[i].accumulate(v * emis_prob_cluster[i](j, t) / emis_prob(i, t), j, e);
	}
      }
      // end accumulate
    }

    // after accumulate all sequences: re-estimate transition & mean & covariance for the next iteration
    for (size_t i = 0; i < M; i++) {
      double s = 0;
      for (size_t j = 0; j < M; j++)
        s += TR(i, j);

      if (s == 0) {
	for (size_t j = 0; j < M; j++)
          guessTR(i, j) = 0;
	guessTR(i, i) = 1;
      } else {
	for (size_t j = 0; j < M; j++)
          guessTR(i, j) = TR(i, j) / s;
      }

      guessMG[i].end_accumulate();
    }
    // end re-estimate

    printf("Iter = %zu Loglik = %8.4f\n", iter, loglik);
    if (fabs(oldlog - loglik) < tol) {
      printf("\nConverged after %zu iterations\n", iter);
      break;
    }
    oldlog = loglik;
  }
}

void MixtureofGaussianHMM::TrainViterbi(const std::vector<arma::mat>& seqs, arma::mat& guessTR, std::vector<MixtureGauss>& guessMG, size_t max_iter, double tol) {
  size_t L = -1;
  size_t M = guessTR.n_rows;
  mlpack::Log::Assert((M == guessTR.n_cols && M == guessMG.size()),
      "MixtureofGaussianHMM::TrainViterbi(): sizes do not match");

  for (size_t i = 0; i < seqs.size(); i++)
    if (seqs[i].n_cols > L)
      L = seqs[i].n_cols;

  arma::mat TR(M, M); // guess transition and emission matrix

  arma::mat emis_prob(M, L); // to hold hmm_decodeG results
  std::vector<arma::mat> emis_prob_cluster;

  for (size_t i = 0; i < M; i++) {
    arma::mat m;
    size_t K = guessMG[i].n_clusters();
    m.set_size(K, L);
    emis_prob_cluster.push_back(m);
  }

  double loglik = 0, oldlog;
  for (size_t iter = 0; iter < max_iter; iter++) {
    oldlog = loglik;
    loglik = 0;

    // set the accumulating values to zeros and compute the inverse matrices and determinant constants
    TR.zeros();
    for (size_t i = 0; i < M; i++)
      guessMG[i].start_accumulate();

    // for each sequence, we will use viterbi procedure to find the most probable state sequence and then accumulate
    for (size_t idx = 0; idx < seqs.size(); idx++) {
      arma::vec states;
      // first calculate the emission probabilities of the sequence
      L = seqs[idx].n_cols;
      for (size_t t = 0; t < L; t++) {
	arma::vec e = seqs[idx].unsafe_col(t);
	for (size_t i = 0; i < M; i++) {
	  double s = 0;
	  size_t K = guessMG[i].n_clusters();
	  for (size_t j = 0; j < K; j++) {
	    emis_prob_cluster[i](j, t) = guessMG[i].getPDF(j, e);
	    s += emis_prob_cluster[i](j, t);
	  }
	  emis_prob(i, t) = s;
	}
      }

      loglik += GaussianHMM::ViterbiInit(L, guessTR, emis_prob, states); // viterbi procedure

      // accumulate expected transition & gaussian mixture parameters
      for (size_t t = 0; t < L-1; t++) {
	size_t i = (size_t) states[t];
	size_t j = (size_t) states[t + 1];
	TR(i, j)++;
      }

      for (size_t t = 0; t < L; t++) {
	arma::vec e = seqs[idx].unsafe_col(t);
	size_t i = (size_t) states[t];
	size_t K = guessMG[i].n_clusters();
	for (size_t j = 0; j < K; j++)
	  guessMG[i].accumulate(emis_prob_cluster[i](j, t) / emis_prob(i, t), j, e);
      }
      // end accumulate
    }

    // after accumulate all sequences: re-estimate transition & mean & covariance for the next iteration
    for (size_t i = 0; i < M; i++) {
      double s = 0;
      for (size_t j = 0; j < M; j++)
        s += TR(i, j);

      if (s == 0) {
	for (size_t j = 0; j < M; j++)
          guessTR(i, j) = 0;
	guessTR(i, i) = 1;
      }
      else {
	for (size_t j = 0; j < M; j++)
          guessTR(i, j) = TR(i, j) / s;
      }

      guessMG[i].end_accumulate();
    }
    // end re-estimate

    printf("Iter = %zu Loglik = %8.4f\n", iter, loglik);
    if (fabs(oldlog - loglik) < tol) {
      printf("\nConverged after %zu iterations\n", iter);
      break;
    }
    oldlog = loglik;
  }
}

