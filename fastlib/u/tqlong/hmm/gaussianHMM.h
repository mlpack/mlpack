#ifndef FASTLIB_GAUSSIAN_HMM_H
#define FASTLIB_GAUSSIAN_HMM_H

#include "fastlib/fastlib.h"

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

/** Calculate the most probable states for a sequence
    Viterbi algorithm
    trans: Transition probabilities, size M x M
    emis_prob: Emission probabilities, size M x L
    
    states: Unitialized, will have length L
    RETURN: log probability of the most probable sequence
*/
double hmm_viterbiG_init(const Matrix& trans, const Matrix& emis_prob, Vector* states);
//double hmm_viterbiD_init(const Vector& seq, const Matrix& trans, const Matrix& emis, Vector* states);

/** Baum-Welch estimation of transition and emission distribution (Gaussian)
    

*/
void hmm_trainG(const ArrayList<Matrix>& seqs, Matrix* guessTR, ArrayList<Vector>* guessME, ArrayList<Matrix>* guessCO, int max_iter, double tol);


#endif
