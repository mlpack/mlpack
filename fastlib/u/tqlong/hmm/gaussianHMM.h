#ifndef FASTLIB_GAUSSIAN_HMM_H
#define FASTLIB_GAUSSIAN_HMM_H

#include "fastlib/fastlib.h"

/**
 * A wrapper class for HMM functionals in single Gaussian case
 */

class GaussianHMM {
 private:
  Matrix transmission_;
  ArrayList<Vector> list_mean_vec_;
  ArrayList<Matrix> list_covariance_mat_;
  
  ArrayList<Matrix> list_inverse_cov_mat_;
  Vector gauss_const_vec_;
  void CalculateInverse();
 public:
  void InitFromFile(const char* profile);
  void InitFromData(const ArrayList<Matrix>& list_data_seq, int numstate);

  void LoadProfile(const char* profile);
  void SaveProfile(const char* profile);

  void GenerateSequence(int L, Matrix* data_seq, Vector* state_seq);

  double ComputeLogLikelihood(const Matrix& data_seq);
  void ComputeLogLikelihood(const ArrayList<Matrix>& list_data_seq, ArrayList<double>* list_likelihood);

  void ComputeViterbiStateSequence(const Matrix& data_seq, Vector* state_seq);

  void TrainBaumWelch(const ArrayList<Matrix>& list_data_seq, int max_iteration = 500, double tolerance = 1e-3);
  void TrainViterbi(const ArrayList<Matrix>& list_data_seq, int max_iteration = 500, double tolerance = 1e-3);
};

success_t load_profileG(const char* profile, Matrix* trans, ArrayList<Vector>* means, ArrayList<Matrix>* covs);
success_t save_profileG(const char* profile, const Matrix& trans, const ArrayList<Vector>& means, const ArrayList<Matrix>& covs);
/**
Generating a sequence and states using transition and emission probabilities.
L: sequence length
trans: Matrix M x M (M states)
means: list of mean vectors length N (emission vector of length N)
covs: list of square root of covariance matrices size N x N

seq: generated sequence, uninitialized matrix, will have size N x L
states: generated states, uninitialized vector, will have length L
*/
void hmm_generateG_init(int L, const Matrix& trans, const ArrayList<Vector>& means, const ArrayList<Matrix>& covs, Matrix* seq, Vector* states);

/** Estimate transition and emission distribution from sequence and states
 
*/
void hmm_estimateG_init(const Matrix& seq, const Vector& states, Matrix* trans, ArrayList<Vector>* means, ArrayList<Matrix>* covs);
void hmm_estimateG_init(int numStates, const Matrix& seq, const Vector& states, Matrix* trans, ArrayList<Vector>* means, ArrayList<Matrix>* covs);

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
double hmm_decodeG(const Matrix& trans, const Matrix& emis_prob, Matrix* pstates, Matrix* fs, Matrix* bs, Vector* scales);
double hmm_decodeG(int L, const Matrix& trans, const Matrix& emis_prob, Matrix* pstates, Matrix* fs, Matrix* bs, Vector* scales);
void hmm_cal_emis_prob(const Matrix& seq, const ArrayList<Vector>& means, const ArrayList<Matrix>& inv_covs, const Vector& det, Matrix* emis_prob);

/** Calculate the most probable states for a sequence
    Viterbi algorithm
    trans: Transition probabilities, size M x M
    emis_prob: Emission probabilities, size M x L
    
    states: Unitialized, will have length L
    RETURN: log probability of the most probable sequence
*/
double hmm_viterbiG_init(const Matrix& trans, const Matrix& emis_prob, Vector* states);
double hmm_viterbiG_init(int L, const Matrix& trans, const Matrix& emis_prob, Vector* states);

/** Baum-Welch and Viterbi estimation of transition and emission distribution (Gaussian)
*/
void init_gauss_param(int M, const ArrayList<Matrix>& seqs, Matrix* guessTR, ArrayList<Vector>* guessME, ArrayList<Matrix>* guessCO);

void hmm_trainG(const ArrayList<Matrix>& seqs, Matrix* guessTR, ArrayList<Vector>* guessME, ArrayList<Matrix>* guessCO, int max_iter, double tol);

void hmm_train_viterbiG(const ArrayList<Matrix>& seqs, Matrix* guessTR, ArrayList<Vector>* guessME, ArrayList<Matrix>* guessCO, int max_iter, double tol);

#endif
