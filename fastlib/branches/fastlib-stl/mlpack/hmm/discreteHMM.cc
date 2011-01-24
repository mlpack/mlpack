/**
 * @file discreteHMM.cc
 *
 * This file contains the implementation of functions in discreteHMM.h
 */

#include <fastlib/fastlib.h>
#include "support.h"
#include "discreteHMM.h"

using namespace hmm_support;

void DiscreteHMM::setModel(const arma::mat& transmission, const arma::mat& emission) {
  DEBUG_ASSERT(transmission.n_rows == transmission.n_cols);
  DEBUG_ASSERT(transmission.n_rows == emission.n_rows);
  
  transmission_ = transmission;
  emission_ = emission;
}

void DiscreteHMM::Init(const arma::mat& transmission, const arma::mat& emission) {
  transmission_ = transmission;
  emission_ = emission;

  DEBUG_ASSERT(transmission.n_rows == transmission.n_cols);
  DEBUG_ASSERT(transmission.n_rows == emission.n_rows);
}

void DiscreteHMM::InitFromFile(const char* profile) {
  std::vector<arma::mat> list_mat;
  load_matrix_list(profile, list_mat);
  if (list_mat.size() < 2)
    FATAL("Number of matrices in the file should be at least 2.");
  else if (list_mat.size() > 2)
    NONFATAL("Number of matrices in the file should be 2 (any after the second are ignored.");

  transmission_ = list_mat[0];
  emission_ = list_mat[1];

  DEBUG_ASSERT(transmission_.n_rows == transmission_.n_cols);
  DEBUG_ASSERT(transmission_.n_rows == emission_.n_rows);
}

void DiscreteHMM::InitFromData(const std::vector<arma::vec>& list_data_seq, int numstate) {
  int numsymbol = 0;
  int maxseq = 0;

  for (int i = 0; i < list_data_seq.size(); i++) {
    if (list_data_seq[i].n_elem > list_data_seq[maxseq].n_elem)
      maxseq = i;
  }

  for (int i = 0; i < list_data_seq[maxseq].n_elem; i++) {
    if (list_data_seq[maxseq][i] > numsymbol)
      numsymbol = (int) list_data_seq[maxseq][i];
  }
  numsymbol++;

  arma::vec states;
  int L = list_data_seq[maxseq].n_elem;
  states.set_size(L);
  for (int i = 0; i < L; i++)
    states[i] = rand() % numstate;

  DiscreteHMM::EstimateInit(numsymbol, numstate, list_data_seq[maxseq], states, transmission_, emission_);
}

void DiscreteHMM::LoadProfile(const char* profile) {
  InitFromFile(profile);
}

void DiscreteHMM::SaveProfile(const char* profile) const {
  TextWriter w_pro;
  if (!PASSED(w_pro.Open(profile))) {
    NONFATAL("Couldn't open '%s' for writing.", profile);
    return;
  }

  print_matrix(w_pro, transmission_, "%% transmission", "%f,");
  print_matrix(w_pro, emission_, "%% emission", "%f,");
}

void DiscreteHMM::GenerateSequence(int length, arma::vec& data_seq, arma::vec& state_seq) const {
  DiscreteHMM::GenerateInit(length, transmission_, emission_, data_seq, state_seq);
}

void DiscreteHMM::EstimateModel(const arma::vec& data_seq, const arma::vec& state_seq) {
  DiscreteHMM::EstimateInit(data_seq, state_seq, transmission_, emission_);
}

void DiscreteHMM::EstimateModel(int numstate, int numsymbol, const arma::vec& data_seq, const arma::vec& state_seq) {
  DiscreteHMM::EstimateInit(numsymbol, numstate, data_seq, state_seq, transmission_, emission_);
}

void DiscreteHMM::DecodeOverwrite(const arma::vec& data_seq, arma::mat& state_prob_mat, arma::mat& forward_prob_mat, arma::mat& backward_prob_mat, arma::vec& scale_vec) const {
  DiscreteHMM::Decode(data_seq, transmission_, emission_, state_prob_mat, forward_prob_mat, backward_prob_mat, scale_vec);
}

void DiscreteHMM::DecodeInit(const arma::vec& data_seq, arma::mat& state_prob_mat, arma::mat& forward_prob_mat, arma::mat& backward_prob_mat, arma::vec& scale_vec) const {
  int M = transmission_.n_rows;
  int L = data_seq.n_elem;

  state_prob_mat.set_size(M, L);
  forward_prob_mat.set_size(M, L);
  backward_prob_mat.set_size(M, L);
  scale_vec.set_size(L);

  DiscreteHMM::Decode(data_seq, transmission_, emission_, state_prob_mat, forward_prob_mat, backward_prob_mat, scale_vec);
}

void forward_procedure(const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, arma::vec& scales, arma::mat& fs);

double DiscreteHMM::ComputeLogLikelihood(const arma::vec& data_seq) const {
  int L = data_seq.n_elem;
  int M = transmission_.n_rows;

  arma::mat fs(M, L);
  arma::vec sc(L);
  
  DiscreteHMM::ForwardProcedure(data_seq, transmission_, emission_, sc, fs);

  double loglik = 0;
  for (int t = 0; t < L; t++)
    loglik += log(sc[t]);
  return loglik;
}

void DiscreteHMM::ComputeLogLikelihood(const std::vector<arma::vec>& list_data_seq, std::vector<double>& list_likelihood) const {
  int L = 0;
  for (int i = 0; i < list_data_seq.size(); i++) {
    if (list_data_seq[i].n_elem > L)
      L = list_data_seq[i].n_elem;
  }

  int M = transmission_.n_rows;

  arma::mat fs(M, L);
  arma::vec sc(L);
  
  for (int i = 0; i < list_data_seq.size(); i++) {
    DiscreteHMM::ForwardProcedure(list_data_seq[i], transmission_, emission_, sc, fs);

    int L = list_data_seq[i].n_elem;
    double loglik = 0;
    for (int t = 0; t < L; t++)
      loglik += log(sc[t]);

    list_likelihood.push_back(loglik);
  }
}

void DiscreteHMM::ComputeViterbiStateSequence(const arma::vec& data_seq, arma::vec& state_seq) const {
  DiscreteHMM::ViterbiInit(data_seq, transmission_, emission_, state_seq);
}

void DiscreteHMM::TrainBaumWelch(const std::vector<arma::vec>& list_data_seq, int max_iteration, double tolerance) {
  DiscreteHMM::Train(list_data_seq, transmission_, emission_, max_iteration, tolerance);
}

void DiscreteHMM::TrainViterbi(const std::vector<arma::vec>& list_data_seq, int max_iteration, double tolerance) {
  DiscreteHMM::TrainViterbi(list_data_seq, transmission_, emission_, max_iteration, tolerance);
}

void DiscreteHMM::GenerateInit(int L, const arma::mat& trans, const arma::mat& emis, arma::vec& seq, arma::vec& states) {
  DEBUG_ASSERT_MSG((trans.n_rows == trans.n_cols && trans.n_rows == emis.n_rows),
    "DiscreteHMM::GenerateInit(): matrix sizes do not match");

  arma::mat trsum, esum;
  
  int M, N;
  int cur_state;

  M = trans.n_rows;
  N = emis.n_cols;

  trsum = trans;
  esum = emis;

  for (int i = 0; i < M; i++) {
    for (int j = 1; j < M; j++)
      trsum(i, j) += trsum(i, j - 1);
    for (int j = 1; j < N; j++) 
      esum(i, j) += esum(i, j - 1);
  }

  seq.set_size(L);
  states.set_size(L);

  cur_state = 0; // starting state is 0
  
  for (int i = 0; i < L; i++) {
    int j;
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
  DEBUG_ASSERT_MSG((seq.n_elem == states.n_elem),
      "DiscreteHMM::EstimateInit(): sequence and states length must be the same");

  int M = 0;
  int N = 0;

  for (int i = 0; i < seq.n_elem; i++) {
    if (seq[i] > N)
      N = (int) seq[i];
    if (states[i] > M)
      M = (int) states[i];
  }

  M++;
  N++;

  DiscreteHMM::EstimateInit(N, M, seq, states, trans, emis);
}

void DiscreteHMM::EstimateInit(int numSymbols, int numStates, const arma::vec& seq, const arma::vec& states, arma::mat& trans, arma::mat& emis){
  DEBUG_ASSERT_MSG((seq.n_elem == states.n_elem),
    "DiscreteHMM::EstimateInit(): sequence and states length must be the same");

  int N = numSymbols;
  int M = numStates;
  int L = seq.n_elem;
  
  arma::vec stateSum;

  trans.zeros(M, M);
  emis.zeros(M, N);
  stateSum.zeros(M);

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

  stateSum.zeros();

  for (int i = 0; i < L; i++) {
    int state = (int) states[i];
    int emission = (int) seq[i];
    stateSum[state]++;
    emis(state, emission)++;
  }

  for (int i = 0; i < M; i++) {
    if (stateSum[i] == 0)
      stateSum[i] = -INFINITY;
    for (int j = 0; j < N; j++)
      emis(i, j) /= stateSum[i];
  }
}

void DiscreteHMM::ForwardProcedure(const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, arma::vec& scales, arma::mat& fs) {
  int L = seq.n_elem;
  int M = trans.n_rows;

  fs.zeros();
  scales.zeros();

  // NOTE: start state is 0
  // time t = 0
  int e = (int) seq[0];
  for (int i = 0; i < M; i++) {
    fs(i, 0) = trans(0, i) * emis(i, e);
    scales[0] += fs(i, 0);
  }
  for (int i = 0; i < M; i++)
    fs(i, 0) /= scales[0];

  // time t = 1 -> L-1
  for (int t = 1; t < L; t++) {
    e = (int) seq[t];
    for (int j = 0; j < M; j++) {
      for (int i = 0; i < M; i++)
	fs(j, t) += fs(i, t - 1) * trans(i, j);
      fs(j, t) *= emis(j, e);
      scales[t] += fs(j, t);
    }
    for (int j = 0; j < M; j++)
      fs(j, t) /= scales[t];
  }
}

void DiscreteHMM::BackwardProcedure(const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, const arma::vec& scales, arma::mat& bs) {
  int L = seq.n_elem;
  int M = trans.n_rows;

  bs.zeros();

  for (int i = 0; i < M; i++)
    bs(i, L - 1) = 1.0;

  for (int t = L - 2; t >= 0; t--) {
    int e = (int) seq[t + 1];
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < M; j++)
	bs(i, t) += trans(i, j) * bs(j, t + 1) * emis(j, e);
      bs(i, t) /= scales[t + 1];
    }
  }
}

double DiscreteHMM::Decode(const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, arma::mat& pstates, arma::mat& fs, arma::mat& bs, arma::vec& scales) {
  int L = seq.n_elem;
  int M = trans.n_rows;

  DEBUG_ASSERT_MSG((L == pstates.n_cols && L == fs.n_cols && L == bs.n_cols && 
		    M == trans.n_cols    && M == emis.n_rows),
                    "DiscreteHMM::Decode(): sizes do not match");
  
  DiscreteHMM::ForwardProcedure(seq, trans, emis, scales, fs);
  DiscreteHMM::BackwardProcedure(seq, trans, emis, scales, bs);

  for (int i = 0; i < M; i++) {
    for (int t = 0; t < L; t++)
      pstates(i, t) = fs(i,t) * bs(i,t);
  }

  double logpseq = 0;
  for (int t = 0; t < L; t++) 
    logpseq += log(scales[t]);

  return logpseq;
}

double DiscreteHMM::ViterbiInit(const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, arma::vec& states) {
  int L = seq.n_elem;

  return DiscreteHMM::ViterbiInit(L, seq, trans, emis, states);
}

double DiscreteHMM::ViterbiInit(int L, const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, arma::vec& states) {
  int M = trans.n_rows;
  int N = emis.n_cols;

  DEBUG_ASSERT_MSG((M == trans.n_cols && M == emis.n_rows),
      "DiscreteHMM::ViterbiInit(): sizes do not match");
  
  states.set_size(L);
  
  arma::vec v(M);
  v.fill(-INFINITY);
  v[0] = 0;
  
  arma::vec v_old = v;

  arma::mat w(M, L);

  arma::mat logtrans(M, M), logemis(M, N);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++)
      logtrans(i, j) = log(trans(i, j));

    for (int j = 0; j < N; j++)
      logemis(i, j) = log(emis(i, j));
  }

  for (int t = 0; t < L; t++) {
    int e = (int) seq[t];

    for (int j = 0; j < M; j++) {
      double bestVal = -INFINITY;
      double bestPtr = -1;

      for (int i = 0; i < M; i++) {
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

  for (int i = 0; i < M; i++)
    if (v[i] > bestVal) {
      bestVal = v[i];
      bestPtr = i;
    }
  
  states[L - 1] = bestPtr;
  for (int t = L - 2; t >= 0; t--)
    states[t] = w((int) states[t + 1], t + 1);

  return bestVal;
}

void DiscreteHMM::Train(const std::vector<arma::vec>& seqs, arma::mat& guessTR, arma::mat& guessEM, int max_iter, double tol) {
  int L = -1;
  int M = guessTR.n_rows;
  int N = guessEM.n_cols;

  DEBUG_ASSERT_MSG((M == guessTR.n_cols && M == guessEM.n_rows),
      "DiscreteHMM::Train(): sizes do not match");
  
  for (int i = 0; i < seqs.size(); i++) {
    if (seqs[i].n_elem > L)
      L = seqs[i].n_elem;
  }

  arma::mat TR(M, M), EM(M, N); // guess transition and emission matrix

  arma::mat ps(M, L), fs(M, L), bs(M, L);
  arma::vec s(L);
  
  double loglik = 0, oldlog;

  for (int iter = 0; iter < max_iter; iter++) {
    oldlog = loglik;
    loglik = 0;

    TR.zeros();
    EM.zeros();
    for (int idx = 0; idx < seqs.size(); idx++) {
      L = seqs[idx].n_elem;
      loglik += DiscreteHMM::Decode(seqs[idx], guessTR, guessEM, ps, fs, bs, s);
      
      for (int t = 0; t < L - 1; t++) {
	int e = (int) seqs[idx][t + 1];
	for (int i = 0; i < M; i++) {
	  for (int j = 0; j < M; j++)
	    TR(i, j) += fs(i, t) * guessTR(i, j) * guessEM(j, e) * bs(j, t + 1) / s[t + 1];
        }
      }
      
      for (int t = 0; t < L; t++) {
	int e = (int) seqs[idx][t];
	for (int i = 0; i < M; i++)
	  EM(i, e) += ps(i, t);
      }
    }

    double s;
    for (int i = 0; i < M; i++) {
      s = 0;
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
      
      s = 0;
      for (int j = 0; j < N; j++)
        s += EM(i, j);
      for (int j = 0; j < N; j++)
        guessEM(i, j) = EM(i, j) / s;
    }

    printf("Iter = %d Loglik = %8.4f\n", iter, loglik);
    if (fabs(oldlog - loglik) < tol) {
      printf("\nConverged after %d iterations\n", iter);
      break;
    }
    oldlog = loglik;
  }
}

void DiscreteHMM::TrainViterbi(const std::vector<arma::vec>& seqs, arma::mat& guessTR, arma::mat& guessEM, int max_iter, double tol) {
  int L = -1;
  int M = guessTR.n_rows;
  int N = guessEM.n_cols;

  DEBUG_ASSERT_MSG((M == guessTR.n_cols && M == guessEM.n_rows),
      "DiscreteHMM::TrainViterbi(): sizes do not match");
  
  for (int i = 0; i < seqs.size(); i++) {
    if (seqs[i].n_elem > L)
      L = seqs[i].n_elem;
  }

  arma::mat TR(M, M), EM(M, N); // guess transition and emission matrix

  double loglik = 0, oldlog;
  for (int iter = 0; iter < max_iter; iter++) {
    oldlog = loglik;
    loglik = 0;

    TR.fill(1e-4);
    EM.fill(1e-4);
    for (int idx = 0; idx < seqs.size(); idx++) {
      arma::vec states;
      L = seqs[idx].n_elem;
      loglik += DiscreteHMM::ViterbiInit(L, seqs[idx], guessTR, guessEM, states);
      
      for (int t = 0; t < L-1; t++) {
	int i = (int) states[t];
	int j = (int) states[t + 1];
	TR(i, j)++;
      }
      
      for (int t = 0; t < L; t++) {
	int e = (int) seqs[idx][t];
	int i = (int) states[t];
	EM(i, e)++;
      }
    }

    double s;
    print_matrix(TR, "TR");
    for (int i = 0; i < M; i++) {
      s = 0;
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
      
      s = 0;
      for (int j = 0; j < N; j++)
        s += EM(i, j);
      for (int j = 0; j < N; j++)
        guessEM(i, j) = EM(i, j) / s;
    }

    printf("Iter = %d Loglik = %8.4f\n", iter, loglik);
    if (fabs(oldlog - loglik) < tol) {
      printf("\nConverged after %d iterations\n", iter);
      break;
    }
    oldlog = loglik;
  }
}
