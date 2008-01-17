#include "fastlib/fastlib.h"
#include "support.h"
#include "mixgaussHMM.h"
#include "gaussianHMM.h"

success_t load_profileM(const char* profile, Matrix* trans, ArrayList<MixtureGauss>* mixs) {
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
    //printf("load p=%d K=%d\n", p, K);
    DEBUG_ASSERT(matlst.size() > p+2*K);
    MixtureGauss mix;
    mix.InitFromProfile(matlst, p, N);
    mixs->AddBackItem(mix);
    p += 2*K+1;
  }
  return SUCCESS_PASS;
}

success_t save_profileM(const char* profile, const Matrix& trans, const ArrayList<MixtureGauss>& mixs) {
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

void hmm_generateM_init(int L, const Matrix& trans, const ArrayList<MixtureGauss>& mixs, Matrix* seq, Vector* states){
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
    double r = RAND_UNIFORM_01;
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

void hmm_estimateM_init(int numStates, int numClusters, const Matrix& seq, const Vector& states, Matrix* trans, ArrayList<MixtureGauss>* mixs) {
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
    data.AddBackItem(m);
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
    mix_.AddBackItem(m);
  }
}

void hmm_estimateM_init(int NumClusters, const Matrix& seq, const Vector& states, Matrix* trans, ArrayList<MixtureGauss>* mixs) {
  DEBUG_ASSERT_MSG((seq.n_cols()==states.length()), "hmm_estimateG_init: sequence and states length must be the same");
  int M = 0;
  for (int i = 0; i < seq.n_cols(); i++)
    if (states[i] > M) M = (int) states[i];
  M++;
  hmm_estimateM_init(M, NumClusters, seq, states, trans, mixs);
}

/*
void forward_procedure(int L, const Matrix& trans, const Matrix& emis_prob, Vector *scales, Matrix* fs) {
  int M = trans.n_rows();

  Matrix& fs_ = *fs;
  Vector& s_ = *scales;

  fs_.SetZero();
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

void backward_procedure(int L, const Matrix& trans, const Matrix& emis_prob, const Vector& scales, Matrix* bs) {
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

double hmm_decodeG(int L, const Matrix& trans, const Matrix& emis_prob, Matrix* pstates, Matrix* fs, Matrix* bs, Vector* scales) {
  int M = trans.n_rows();

  DEBUG_ASSERT_MSG((L==pstates->n_cols() && L==fs->n_cols() && L == bs->n_cols() && 
		    M==trans.n_cols() && M==emis_prob.n_rows()),"hmm_decodeG: sizes do not match");
  
  Matrix& ps_ = *pstates;

  Vector& s_ = *scales;
  s_.SetZero();

  forward_procedure(L, trans, emis_prob, &s_, fs);
  backward_procedure(L, trans, emis_prob, s_, bs);

  for (int i = 0; i < M; i++)
    for (int t = 0; t < L; t++)
      ps_.ref(i, t) = fs->get(i,t) * bs->get(i,t);

  double logpseq = 0;
  for (int t = 0; t < L; t++) 
    logpseq += log(s_[t]);

  return logpseq;
}

double hmm_decodeG(const Matrix& trans, const Matrix& emis_prob, Matrix* pstates, Matrix* fs, Matrix* bs, Vector* scales) {
  int L = emis_prob.n_cols();
  return hmm_decodeM(L, trans, emis_prob, pstates, fs, bs, scales);
}
*/
/*
double hmm_viterbiG_init(const Matrix& trans, const Matrix& emis_prob, Vector* states) {
  int L = emis_prob.n_cols();
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
*/

void hmm_cal_emis_probM(const Matrix& seq, const ArrayList<MixtureGauss>& mixs, Matrix* emis_prob) {
  int M = mixs.size();
  int L = seq.n_cols();
  for (int t = 0; t < L; t++) {
    Vector e;
    seq.MakeColumnVector(t, &e);
    for (int i = 0; i < M; i++)
      emis_prob->ref(i, t) = mixs[i].getPDF(e);
  }
}

void hmm_trainM(const ArrayList<Matrix>& seqs, Matrix* guessTR, ArrayList<MixtureGauss>* guessMG, int max_iter, double tol) {
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
    emis_prob_cluster.AddBackItem(m);
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
      
      loglik += hmm_decodeG(L, gTR, emis_prob, &ps, &fs, &bs, &s); // forward - backward procedure
      
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

void hmm_train_viterbiM(const ArrayList<Matrix>& seqs, Matrix* guessTR, ArrayList<MixtureGauss>* guessMG, int max_iter, double tol) {
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
    emis_prob_cluster.AddBackItem(m);
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
      
      loglik += hmm_viterbiG_init(L, gTR, emis_prob, &states); // viterbi procedure
      
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

