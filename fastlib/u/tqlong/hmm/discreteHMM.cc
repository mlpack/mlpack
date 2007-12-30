#include "fastlib/fastlib.h"
#include "support.h"
#include "discreteHMM.h"

void hmm_generateD_init(int L, const Matrix& trans, const Matrix& emis, Vector* seq, Vector* states) {
  DEBUG_ASSERT_MSG((trans.n_rows()==trans.n_cols() && trans.n_rows()==emis.n_rows()), "hmm_generateD_init: matrices sizes do not match");
  Matrix trsum, esum;
  Vector &seq_ = *seq, &states_ = *states;
  int M, N;
  int cur_state;

  M = trans.n_rows();
  N = emis.n_cols();

  trsum.Copy(trans);
  esum.Copy(emis);

  for (int i = 0; i < M; i++) {
    for (int j = 1; j < M; j++)
      trsum.set(i, j, trsum.get(i, j) + trsum.get(i, j-1));
    for (int j = 1; j < N; j++) 
      esum.set(i, j, esum.get(i, j) + esum.get(i, j-1));
  }

  seq_.Init(L);
  states_.Init(L);

  cur_state = 0; // starting state is 0
  
  for (int i = 0; i < L; i++) {
    int j;
    double r;

    // next state
    r = RAND_UNIFORM_01;
    for (j = 0; j < M; j++)
      if (r <= trsum.get(cur_state, j)) break;
    cur_state = j;
	
    // emission
    r = RAND_UNIFORM_01;
    for (j = 0; j < N; j++)
      if (r <= esum.get(cur_state, j)) break;
    seq_[i] = j;
    states_[i] = cur_state;
  }
}

void hmm_estimateD_init(const Vector& seq, const Vector& states, Matrix* trans, Matrix* emis) {
  DEBUG_ASSERT_MSG((seq.length()==states.length()), "hmm_estimateD_init: sequence and states length must be the same");
  int M = 0, N=0;
  for (int i = 0; i < seq.length(); i++) {
    if (seq[i] > N) N = (int) seq[i];
    if (states[i] > M) M = (int) states[i];
  }
  M++;
  N++;
  hmm_estimateD_init(N, M, seq, states, trans, emis);
}

void hmm_estimateD_init(int numSymbols, int numStates, const Vector& seq, const Vector& states, Matrix* trans, Matrix* emis){
  DEBUG_ASSERT_MSG((seq.length()==states.length()), "hmm_estimateD_init: sequence and states length must be the same");
  int N = numSymbols;
  int M = numStates;
  int L = seq.length();
  
  Matrix &trans_ = *trans;
  Matrix &emis_ = *emis;
  Vector stateSum;

  trans_.Init(M, M);
  emis_.Init(M, N);
  stateSum.Init(M);

  trans_.SetZero();
  emis_.SetZero();

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
    int emission = (int) seq[i];
    stateSum[state]++;
    emis_.ref(state, emission)++;
  }
  for (int i = 0; i < M; i++) {
    if (stateSum[i] == 0) stateSum[i] = -INFINITY;
    for (int j = 0; j < N; j++)
      emis_.ref(i, j) /= stateSum[i];
  }
}

void forward_procedure(const Vector& seq, const Matrix& trans, const Matrix& emis, Vector *scales, Matrix* fs) {
  int L = seq.length();
  int M = trans.n_rows();

  Matrix& fs_ = *fs;
  Vector& s_ = *scales;

  fs_.SetZero();
  // NOTE: start state is 0
  // time t = 0
  int e = (int) seq[0];
  for (int i = 0; i < M; i++) {
    fs_.ref(i, 0) = trans.get(0, i) * emis.get(i, e);
    s_[0] += fs_.get(i, 0);
  }
  for (int i = 0; i < M; i++)
    fs_.ref(i, 0) /= s_[0];

  // time t = 1 -> L-1
  for (int t = 1; t < L; t++) {
    e = (int) seq[t];
    for (int j = 0; j < M; j++) {
      for (int i = 0; i < M; i++)
	fs_.ref(j, t) += fs_.get(i, t-1)*trans.get(i, j);
      fs_.ref(j, t) *= emis.get(j, e);
      s_[t] += fs_.get(j, t);
    }
    for (int j = 0; j < M; j++)
      fs_.ref(j, t) /= s_[t];
  }
}

void backward_procedure(const Vector& seq, const Matrix& trans, const Matrix& emis, const Vector& scales, Matrix* bs) {
  int L = seq.length();
  int M = trans.n_rows();

  Matrix& bs_ = *bs;
  bs_.SetZero();
  for (int i = 0; i < M; i++)
    bs_.ref(i, L-1) = 1.0;

  for (int t = L-2; t >= 0; t--) {
    int e = (int) seq[t+1];
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < M; j++)
	bs_.ref(i, t) += trans.get(i, j) * bs_.ref(j, t+1) * emis.get(j, e);
      bs_.ref(i, t) /= scales[t+1];
    }
  }
}

double hmm_decodeD(const Vector& seq, const Matrix& trans, const Matrix& emis, Matrix* pstates, Matrix* fs, Matrix* bs, Vector* scales) {
  int L = seq.length();
  int M = trans.n_rows();

  DEBUG_ASSERT_MSG((L==pstates->n_cols() && L==fs->n_cols() && L == bs->n_cols() && 
		    M==trans.n_cols() && M==emis.n_rows()),"hmm_decodeD: sizes do not match");
  
  Matrix& ps_ = *pstates;

  Vector& s_ = *scales;
  s_.SetZero();

  forward_procedure(seq, trans, emis, &s_, fs);
  backward_procedure(seq, trans, emis, s_, bs);

  for (int i = 0; i < M; i++)
    for (int t = 0; t < L; t++)
      ps_.ref(i, t) = fs->get(i,t) * bs->get(i,t);

  double logpseq = 0;
  for (int t = 0; t < L; t++) 
    logpseq += log(s_[t]);

  return logpseq;
}

double hmm_viterbiD_init(const Vector& seq, const Matrix& trans, const Matrix& emis, Vector* states) {
  int L = seq.length();
  int M = trans.n_rows();
  int N = emis.n_cols();
  DEBUG_ASSERT_MSG((M==trans.n_cols() && M==emis.n_rows()),"hmm_viterbiD: sizes do not match");
  
  Vector& s_ = *states;
  s_.Init(L);
  
  Vector v, vOld;
  v.Init(M);
  v.SetAll(-INFINITY);
  v[0] = 0;
  vOld.Copy(v);

  Matrix w;
  w.Init(M, L);

  Matrix logtrans, logemis;
  logtrans.Init(M, M);
  logemis.Init(M, N);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) logtrans.ref(i, j) = log(trans.get(i, j));
    for (int j = 0; j < N; j++) logemis.ref(i, j) = log(emis.get(i, j));
  }

  for (int t = 0; t < L; t++) {
    int e = (int) seq[t];
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
      v[j] = bestVal + logemis.get(j, e);
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

void hmm_trainD(const ArrayList<Vector>& seqs, Matrix* guessTR, Matrix* guessEM, int max_iter, double tol) {
  int L = -1;
  int M = guessTR->n_rows();
  int N = guessEM->n_cols();
  DEBUG_ASSERT_MSG((M==guessTR->n_cols() && M==guessEM->n_rows()),"hmm_trainD: sizes do not match");
  
  for (int i = 0; i < seqs.size(); i++)
    if (seqs[i].length() > L) L = seqs[i].length();

  Matrix &gTR = *guessTR, &gEM = *guessEM;
  Matrix TR, EM; // guess transition and emission matrix
  TR.Init(M, M);
  EM.Init(M, N);

  Matrix ps, fs, bs;
  Vector s;

  ps.Init(M, L);
  fs.Init(M, L);
  bs.Init(M, L);
  s.Init(L);

  double loglik = 0, oldlog;
  for (int iter = 0; iter < max_iter; iter++) {
    oldlog = loglik;
    loglik = 0;

    TR.SetZero();
    EM.SetZero();
    for (int idx = 0; idx < seqs.size(); idx++) {
      L = seqs[idx].length();
      loglik += hmm_decodeD(seqs[idx], gTR, gEM, &ps, &fs, &bs, &s);
      
      for (int t = 0; t < L-1; t++) {
	int e = (int) seqs[idx][t+1];
	for (int i = 0; i < M; i++)
	  for (int j = 0; j < M; j++)
	    TR.ref(i, j) += fs.get(i, t) * gTR.get(i, j) * gEM.get(j, e) * bs.get(j, t+1) / s[t+1];
      }
      
      for (int t = 0; t < L; t++) {
	int e = (int) seqs[idx][t];
	for (int i = 0; i < M; i++)
	  EM.ref(i, e) += ps.get(i, t);
      }
    }

    double s;
    for (int i = 0; i < M; i++) {
      s = 0;
      for (int j = 0; j < M; j++) s += TR.get(i, j);
      if (s == 0) {
	for (int j = 0; j < M; j++) gTR.ref(i, j) = 0;
	gTR.ref(i, i) = 1;
      }
      else {
	for (int j = 0; j < M; j++) gTR.ref(i, j) = TR.get(i, j) / s;
      }
      
      s = 0;
      for (int j = 0; j < N; j++) s += EM.get(i, j);
      for (int j = 0; j < N; j++) gEM.ref(i, j) = EM.get(i, j) / s;
    }

    printf("Iter = %d Loglik = %8.4f\n", iter, loglik);
    if (fabs(oldlog - loglik) < tol) {
      printf("\nConverged after %d iterations\n", iter);
      break;
    }
    oldlog = loglik;
  }
}
