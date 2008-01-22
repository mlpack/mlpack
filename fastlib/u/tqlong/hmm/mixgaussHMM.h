#ifndef FASTLIB_MIXGAUSS_HMM_H
#define FASTLIB_MIXGAUSS_HMM_H

#include "fastlib/fastlib.h"
#include "mixtureDST.h"

class MixtureofGaussianHMM {
 private:
  Matrix transmission_;
  ArrayList<MixtureGauss> list_mixture_gauss_;
 public:
  void InitFromFile(const char* profile);
  void Init() {
    transmission_.Init(0, 0);
    list_mixture_gauss_.Init();
  }

  void LoadProfile(const char* profile);
  void SaveProfile(const char* profile);

  void GenerateSequence(int L, Matrix* data_seq, Vector* state_seq);

  double ComputeLogLikelihood(const Matrix& data_seq);
  void ComputeLogLikelihood(const ArrayList<Matrix>& list_data_seq, ArrayList<double>* list_likelihood);

  void ComputeViterbiStateSequence(const Matrix& data_seq, Vector* state_seq);

  void TrainBaumWelch(const ArrayList<Matrix>& list_data_seq, int max_iteration = 500, double tolerance = 1e-3);
  void TrainViterbi(const ArrayList<Matrix>& list_data_seq, int max_iteration = 500, double tolerance = 1e-3);
};

success_t load_profileM(const char* profile, Matrix* trans, ArrayList<MixtureGauss>* mixs);
success_t save_profileM(const char* profile, const Matrix& trans, const ArrayList<MixtureGauss>& mixs);

/**
Generating a sequence and states using transition and emission probabilities.
L: sequence length
trans: Matrix M x M (M states)
means: list of mean vectors length N (emission vector of length N)
covs: list of square root of covariance matrices size N x N

seq: generated sequence, uninitialized matrix, will have size N x L
states: generated states, uninitialized vector, will have length L
*/
void hmm_generateM_init(int L, const Matrix& trans, const ArrayList<MixtureGauss>& mixs, Matrix* seq, Vector* states);

/** Estimate transition and emission distribution from sequence and states
 
*/
void hmm_estimateM_init(int NumClusters, const Matrix& seq, const Vector& states, Matrix* trans, ArrayList<MixtureGauss>* mixs);
void hmm_estimateM_init(int numStates, int NumClusters, const Matrix& seq, const Vector& states, Matrix* trans, ArrayList<MixtureGauss>* mixs);

/** Calculate posteriori probabilities of states at each steps
    Scaled Forward - Backward procedure

    trans: Transition probabilities, size M x M
    emis_prob: Emission probabilities along the sequence, size M x L (L is the sequence length)

    pstates: size M x L
    fs: scaled forward probabities, size M x L
    bs: scaled backward probabities, size M x L
    scales: scale factors, length L
    RETURN: log probabilities of sequence
*/
//double hmm_decodeG(const Matrix& trans, const Matrix& emis_prob, Matrix* pstates, Matrix* fs, Matrix* bs, Vector* scales);

/** Calculate the most probable states for a sequence
    Viterbi algorithm
    trans: Transition probabilities, size M x M
    emis_prob: Emission probabilities, size M x L
    
    states: Unitialized, will have length L
    RETURN: log probability of the most probable sequence
*/
//double hmm_viterbiG_init(const Matrix& trans, const Matrix& emis_prob, Vector* states);
//double hmm_viterbiD_init(const Vector& seq, const Matrix& trans, const Matrix& emis, Vector* states);

/** Baum-Welch estimation of transition and emission distribution (Gaussian)
*/
void hmm_cal_emis_probM(const Matrix& seq, const ArrayList<MixtureGauss>& mixs, Matrix* emis_prob);
void hmm_trainM(const ArrayList<Matrix>& seqs, Matrix* guessTR, ArrayList<MixtureGauss>* guessMG, int max_iter=500, double tol=1e-3);
void hmm_train_viterbiM(const ArrayList<Matrix>& seqs, Matrix* guessTR, ArrayList<MixtureGauss>* guessMG, int max_iter=500, double tol=1e-3);

#endif
