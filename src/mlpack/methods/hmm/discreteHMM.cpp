/**
 * @file discreteHMM.cc
 *
 * This file contains the implementation of functions in discreteHMM.h
 */
#include <mlpack/core.h>

#include "support.hpp"
#include "discreteHMM.hpp"

namespace mlpack {
namespace hmm {

using namespace hmm_support;

void DiscreteHMM::setModel(const arma::mat& transmission, const arma::mat& emission) {
  mlpack::Log::Assert(transmission.n_rows == transmission.n_cols);
  mlpack::Log::Assert(transmission.n_rows == emission.n_rows);

  transmission_ = transmission;
  emission_ = emission;
}

void DiscreteHMM::Init(const arma::mat& transmission, const arma::mat& emission) {
  transmission_ = transmission;
  emission_ = emission;

  mlpack::Log::Assert(transmission.n_rows == transmission.n_cols);
  mlpack::Log::Assert(transmission.n_rows == emission.n_rows);
}

void DiscreteHMM::InitFromFile(const char* profile) {
  std::vector<arma::mat> list_mat;
  load_matrix_list(profile, list_mat);
  if (list_mat.size() < 2)
    mlpack::Log::Fatal << "Number of matrices in the file should be at least 2."
        << std::endl;
  else if (list_mat.size() > 2)
    mlpack::Log::Warn << "Number of matrices in the file should be 2 (any after the second are ignored."
        << std::endl;

  transmission_ = list_mat[0];
  emission_ = list_mat[1];

  mlpack::Log::Assert(transmission_.n_rows == transmission_.n_cols);
  mlpack::Log::Assert(transmission_.n_rows == emission_.n_rows);
}

void DiscreteHMM::InitFromData(const std::vector<arma::vec>& list_data_seq, size_t numstate) {
  size_t numsymbol = 0;
  size_t maxseq = 0;

  for (size_t i = 0; i < list_data_seq.size(); i++) {
    if (list_data_seq[i].n_elem > list_data_seq[maxseq].n_elem)
      maxseq = i;
  }

  for (size_t i = 0; i < list_data_seq[maxseq].n_elem; i++) {
    if (list_data_seq[maxseq][i] > numsymbol)
      numsymbol = (size_t) list_data_seq[maxseq][i];
  }
  numsymbol++;

  arma::vec states;
  size_t L = list_data_seq[maxseq].n_elem;
  states.set_size(L);
  for (size_t i = 0; i < L; i++)
    states[i] = rand() % numstate;

  DiscreteHMM::EstimateInit(numsymbol, numstate, list_data_seq[maxseq], states, transmission_, emission_);
}

void DiscreteHMM::LoadProfile(const char* profile) {
  InitFromFile(profile);
}

void DiscreteHMM::SaveProfile(const char* profile) const {
  /*** need something better
  TextWriter w_pro;
  if (!(w_pro.Open(profile))) {
    mlpack::Log::Warn << "Couldn't open " << profile << " for writing." <<
        std::endl;
    return;
  }

  print_matrix(w_pro, transmission_, "%% transmission", "%f,");
  print_matrix(w_pro, emission_, "%% emission", "%f,");
  */
}

void DiscreteHMM::GenerateSequence(size_t length, arma::vec& data_seq, arma::vec& state_seq) const {
  DiscreteHMM::GenerateInit(length, transmission_, emission_, data_seq, state_seq);
}

void DiscreteHMM::EstimateModel(const arma::vec& data_seq, const arma::vec& state_seq) {
  DiscreteHMM::EstimateInit(data_seq, state_seq, transmission_, emission_);
}

void DiscreteHMM::EstimateModel(size_t numstate, size_t numsymbol, const arma::vec& data_seq, const arma::vec& state_seq) {
  DiscreteHMM::EstimateInit(numsymbol, numstate, data_seq, state_seq, transmission_, emission_);
}

void DiscreteHMM::DecodeOverwrite(const arma::vec& data_seq, arma::mat& state_prob_mat, arma::mat& forward_prob_mat, arma::mat& backward_prob_mat, arma::vec& scale_vec) const {
  DiscreteHMM::Decode(data_seq, transmission_, emission_, state_prob_mat, forward_prob_mat, backward_prob_mat, scale_vec);
}

void DiscreteHMM::DecodeInit(const arma::vec& data_seq, arma::mat& state_prob_mat, arma::mat& forward_prob_mat, arma::mat& backward_prob_mat, arma::vec& scale_vec) const {
  size_t M = transmission_.n_rows;
  size_t L = data_seq.n_elem;

  state_prob_mat.set_size(M, L);
  forward_prob_mat.set_size(M, L);
  backward_prob_mat.set_size(M, L);
  scale_vec.set_size(L);

  DiscreteHMM::Decode(data_seq, transmission_, emission_, state_prob_mat, forward_prob_mat, backward_prob_mat, scale_vec);
}

void forward_procedure(const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, arma::vec& scales, arma::mat& fs);

double DiscreteHMM::ComputeLogLikelihood(const arma::vec& data_seq) const {
  size_t L = data_seq.n_elem;
  size_t M = transmission_.n_rows;

  arma::mat fs(M, L);
  arma::vec sc(L);

  DiscreteHMM::ForwardProcedure(data_seq, transmission_, emission_, sc, fs);

  double loglik = 0;
  for (size_t t = 0; t < L; t++)
    loglik += log(sc[t]);
  return loglik;
}

void DiscreteHMM::ComputeLogLikelihood(const std::vector<arma::vec>& list_data_seq, std::vector<double>& list_likelihood) const {
  size_t L = 0;
  for (size_t i = 0; i < list_data_seq.size(); i++) {
    if (list_data_seq[i].n_elem > L)
      L = list_data_seq[i].n_elem;
  }

  size_t M = transmission_.n_rows;

  arma::mat fs(M, L);
  arma::vec sc(L);

  for (size_t i = 0; i < list_data_seq.size(); i++) {
    DiscreteHMM::ForwardProcedure(list_data_seq[i], transmission_, emission_, sc, fs);

    size_t L = list_data_seq[i].n_elem;
    double loglik = 0;
    for (size_t t = 0; t < L; t++)
      loglik += log(sc[t]);

    list_likelihood.push_back(loglik);
  }
}

void DiscreteHMM::ComputeViterbiStateSequence(const arma::vec& data_seq, arma::vec& state_seq) const {
  DiscreteHMM::ViterbiInit(data_seq, transmission_, emission_, state_seq);
}

void DiscreteHMM::TrainBaumWelch(const std::vector<arma::vec>& list_data_seq, size_t max_iteration, double tolerance) {
  DiscreteHMM::Train(list_data_seq, transmission_, emission_, max_iteration, tolerance);
}

void DiscreteHMM::TrainViterbi(const std::vector<arma::vec>& list_data_seq, size_t max_iteration, double tolerance) {
  DiscreteHMM::TrainViterbi(list_data_seq, transmission_, emission_, max_iteration, tolerance);
}

void DiscreteHMM::GenerateInit(size_t L, const arma::mat& trans, const arma::mat& emis, arma::vec& seq, arma::vec& states) {
  mlpack::Log::Assert((trans.n_rows == trans.n_cols && trans.n_rows == emis.n_rows),
    "DiscreteHMM::GenerateInit(): matrix sizes do not match");

  arma::mat trsum, esum;

  size_t M, N;
  size_t cur_state;

  M = trans.n_rows;
  N = emis.n_cols;

  trsum = trans;
  esum = emis;

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 1; j < M; j++)
      trsum(i, j) += trsum(i, j - 1);
    for (size_t j = 1; j < N; j++)
      esum(i, j) += esum(i, j - 1);
  }

  seq.set_size(L);
  states.set_size(L);

  cur_state = 0; // starting state is 0

  for (size_t i = 0; i < L; i++) {
    size_t j;
    double r;

    // next state
    r = (double) rand() / (double) RAND_MAX;
    for (j = 0; j < M; j++) {
      if (r <= trsum(cur_state, j))
        break;
    }
    cur_state = j;

    // emission
    r = (double) rand() / (double) RAND_MAX;
    for (j = 0; j < N; j++) {
      if (r <= esum(cur_state, j))
        break;
    }

    seq[i] = j;
    states[i] = cur_state;
  }
}

void DiscreteHMM::EstimateInit(const arma::vec& seq, const arma::vec& states, arma::mat& trans, arma::mat& emis) {
  mlpack::Log::Assert((seq.n_elem == states.n_elem),
      "DiscreteHMM::EstimateInit(): sequence and states length must be the same");

  size_t M = 0;
  size_t N = 0;

  for (size_t i = 0; i < seq.n_elem; i++) {
    if (seq[i] > N)
      N = (size_t) seq[i];
    if (states[i] > M)
      M = (size_t) states[i];
  }

  M++;
  N++;

  DiscreteHMM::EstimateInit(N, M, seq, states, trans, emis);
}

void DiscreteHMM::EstimateInit(size_t numSymbols, size_t numStates, const arma::vec& seq, const arma::vec& states, arma::mat& trans, arma::mat& emis){
  mlpack::Log::Assert((seq.n_elem == states.n_elem),
    "DiscreteHMM::EstimateInit(): sequence and states length must be the same");

  size_t N = numSymbols;
  size_t M = numStates;
  size_t L = seq.n_elem;

  arma::vec stateSum;

  trans.zeros(M, M);
  emis.zeros(M, N);
  stateSum.zeros(M);

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

  stateSum.zeros();

  for (size_t i = 0; i < L; i++) {
    size_t state = (size_t) states[i];
    size_t emission = (size_t) seq[i];
    stateSum[state]++;
    emis(state, emission)++;
  }

  for (size_t i = 0; i < M; i++) {
    if (stateSum[i] == 0)
      stateSum[i] = -INFINITY;
    for (size_t j = 0; j < N; j++)
      emis(i, j) /= stateSum[i];
  }
}

void DiscreteHMM::ForwardProcedure(const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, arma::vec& scales, arma::mat& fs) {
  size_t L = seq.n_elem;
  size_t M = trans.n_rows;

  fs.zeros();
  scales.zeros();

  // NOTE: start state is 0
  // time t = 0
  size_t e = (size_t) seq[0];
  for (size_t i = 0; i < M; i++) {
    fs(i, 0) = trans(0, i) * emis(i, e);
    scales[0] += fs(i, 0);
  }
  for (size_t i = 0; i < M; i++)
    fs(i, 0) /= scales[0];

  // time t = 1 -> L-1
  for (size_t t = 1; t < L; t++) {
    e = (size_t) seq[t];
    for (size_t j = 0; j < M; j++) {
      for (size_t i = 0; i < M; i++)
	fs(j, t) += fs(i, t - 1) * trans(i, j);
      fs(j, t) *= emis(j, e);
      scales[t] += fs(j, t);
    }
    for (size_t j = 0; j < M; j++)
      fs(j, t) /= scales[t];
  }
}

void DiscreteHMM::BackwardProcedure(const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, const arma::vec& scales, arma::mat& bs) {
  size_t L = seq.n_elem;
  size_t M = trans.n_rows;

  bs.zeros();

  for (size_t i = 0; i < M; i++)
    bs(i, L - 1) = 1.0;

  for (size_t t = L - 2; t + 1 > 0; t--) {
    size_t e = (size_t) seq[t + 1];
    for (size_t i = 0; i < M; i++) {
      for (size_t j = 0; j < M; j++)
	bs(i, t) += trans(i, j) * bs(j, t + 1) * emis(j, e);
      bs(i, t) /= scales[t + 1];
    }
  }
}

double DiscreteHMM::Decode(const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, arma::mat& pstates, arma::mat& fs, arma::mat& bs, arma::vec& scales) {
  size_t L = seq.n_elem;
  size_t M = trans.n_rows;

  mlpack::Log::Assert((L == pstates.n_cols && L == fs.n_cols && L == bs.n_cols &&
		    M == trans.n_cols    && M == emis.n_rows),
                    "DiscreteHMM::Decode(): sizes do not match");

  DiscreteHMM::ForwardProcedure(seq, trans, emis, scales, fs);
  DiscreteHMM::BackwardProcedure(seq, trans, emis, scales, bs);

  for (size_t i = 0; i < M; i++) {
    for (size_t t = 0; t < L; t++)
      pstates(i, t) = fs(i,t) * bs(i,t);
  }

  double logpseq = 0;
  for (size_t t = 0; t < L; t++)
    logpseq += log(scales[t]);

  return logpseq;
}

double DiscreteHMM::ViterbiInit(const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, arma::vec& states) {
  size_t L = seq.n_elem;

  return DiscreteHMM::ViterbiInit(L, seq, trans, emis, states);
}

double DiscreteHMM::ViterbiInit(size_t L, const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, arma::vec& states) {
  size_t M = trans.n_rows;
  size_t N = emis.n_cols;

  mlpack::Log::Assert((M == trans.n_cols && M == emis.n_rows),
      "DiscreteHMM::ViterbiInit(): sizes do not match");

  states.set_size(L);

  arma::vec v(M);
  v.fill(-INFINITY);
  v[0] = 0;

  arma::vec v_old = v;

  arma::mat w(M, L);

  arma::mat logtrans(M, M), logemis(M, N);

  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < M; j++)
      logtrans(i, j) = log(trans(i, j));

    for (size_t j = 0; j < N; j++)
      logemis(i, j) = log(emis(i, j));
  }

  for (size_t t = 0; t < L; t++) {
    size_t e = (size_t) seq[t];

    for (size_t j = 0; j < M; j++) {
      double bestVal = -INFINITY;
      double bestPtr = -1;

      for (size_t i = 0; i < M; i++) {
	double val = v_old[i] + logtrans(i, j);
	if (val > bestVal) {
	  bestVal = val;
	  bestPtr = i;
	}
      }
      v[j] = bestVal + logemis(j, e);
      w(j, t) = bestPtr;
    }
    v_old = v;
  }

  double bestVal = -INFINITY;
  double bestPtr = -1;

  for (size_t i = 0; i < M; i++)
    if (v[i] > bestVal) {
      bestVal = v[i];
      bestPtr = i;
    }

  states[L - 1] = bestPtr;
  for (size_t t = L - 2; t + 1 > 0; t--)
    states[t] = w((size_t) states[t + 1], t + 1);

  return bestVal;
}

void DiscreteHMM::Train(const std::vector<arma::vec>& seqs, arma::mat& guessTR, arma::mat& guessEM, size_t max_iter, double tol) {
  size_t L = -1;
  size_t M = guessTR.n_rows;
  size_t N = guessEM.n_cols;

  mlpack::Log::Assert((M == guessTR.n_cols && M == guessEM.n_rows),
      "DiscreteHMM::Train(): sizes do not match");

  for (size_t i = 0; i < seqs.size(); i++) {
    if (seqs[i].n_elem > L)
      L = seqs[i].n_elem;
  }

  arma::mat TR(M, M), EM(M, N); // guess transition and emission matrix

  arma::mat ps(M, L), fs(M, L), bs(M, L);
  arma::vec s(L);

  double loglik = 0, oldlog;

  for (size_t iter = 0; iter < max_iter; iter++) {
    oldlog = loglik;
    loglik = 0;

    TR.zeros();
    EM.zeros();
    for (size_t idx = 0; idx < seqs.size(); idx++) {
      L = seqs[idx].n_elem;
      loglik += DiscreteHMM::Decode(seqs[idx], guessTR, guessEM, ps, fs, bs, s);

      for (size_t t = 0; t < L - 1; t++) {
	size_t e = (size_t) seqs[idx][t + 1];
	for (size_t i = 0; i < M; i++) {
	  for (size_t j = 0; j < M; j++)
	    TR(i, j) += fs(i, t) * guessTR(i, j) * guessEM(j, e) * bs(j, t + 1) / s[t + 1];
        }
      }

      for (size_t t = 0; t < L; t++) {
	size_t e = (size_t) seqs[idx][t];
	for (size_t i = 0; i < M; i++)
	  EM(i, e) += ps(i, t);
      }
    }

    double s;
    for (size_t i = 0; i < M; i++) {
      s = 0;
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

      s = 0;
      for (size_t j = 0; j < N; j++)
        s += EM(i, j);
      for (size_t j = 0; j < N; j++)
        guessEM(i, j) = EM(i, j) / s;
    }

    printf("Iter = %zu Loglik = %8.4f\n", iter, loglik);
    if (fabs(oldlog - loglik) < tol) {
      printf("\nConverged after %zu iterations\n", iter);
      break;
    }
    oldlog = loglik;
  }
}

void DiscreteHMM::TrainViterbi(const std::vector<arma::vec>& seqs, arma::mat& guessTR, arma::mat& guessEM, size_t max_iter, double tol) {
  size_t L = -1;
  size_t M = guessTR.n_rows;
  size_t N = guessEM.n_cols;

  mlpack::Log::Assert((M == guessTR.n_cols && M == guessEM.n_rows),
      "DiscreteHMM::TrainViterbi(): sizes do not match");

  for (size_t i = 0; i < seqs.size(); i++) {
    if (seqs[i].n_elem > L)
      L = seqs[i].n_elem;
  }

  arma::mat TR(M, M), EM(M, N); // guess transition and emission matrix

  double loglik = 0, oldlog;
  for (size_t iter = 0; iter < max_iter; iter++) {
    oldlog = loglik;
    loglik = 0;

    TR.fill(1e-4);
    EM.fill(1e-4);
    for (size_t idx = 0; idx < seqs.size(); idx++) {
      arma::vec states;
      L = seqs[idx].n_elem;
      loglik += DiscreteHMM::ViterbiInit(L, seqs[idx], guessTR, guessEM, states);

      for (size_t t = 0; t < L-1; t++) {
	size_t i = (size_t) states[t];
	size_t j = (size_t) states[t + 1];
	TR(i, j)++;
      }

      for (size_t t = 0; t < L; t++) {
	size_t e = (size_t) seqs[idx][t];
	size_t i = (size_t) states[t];
	EM(i, e)++;
      }
    }

    double s;
    print_matrix(TR, "TR");
    for (size_t i = 0; i < M; i++) {
      s = 0;
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

      s = 0;
      for (size_t j = 0; j < N; j++)
        s += EM(i, j);
      for (size_t j = 0; j < N; j++)
        guessEM(i, j) = EM(i, j) / s;
    }

    printf("Iter = %zu Loglik = %8.4f\n", iter, loglik);
    if (fabs(oldlog - loglik) < tol) {
      printf("\nConverged after %zu iterations\n", iter);
      break;
    }
    oldlog = loglik;
  }
}

}; // namespace hmm
}; // namespace mlpack
