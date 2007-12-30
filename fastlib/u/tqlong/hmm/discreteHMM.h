#ifndef FASTLIB_DISCRETE_HMM_H
#define FASTLIB_DISCRETE_HMM_H

#include "fastlib/fastlib.h"

/**
Generating a sequence and states using transition and emission probabilities.
L: sequence length
trans: Matrix M x M (M states)
emis: Matrix M x N (N emissions)
seq: uninitialized vector, will have length L
states: uninitialized vector, will have length L
*/
void hmm_generateD_init(int L, const Matrix& trans, const Matrix& emis, Vector* seq, Vector * states);


/** Estimate transition and emission probabilities from sequence and states
 */
void hmm_estimateD_init(const Vector& seq, const Vector& states, Matrix* trans, Matrix* emis);
void hmm_estimateD_init(int numSymbols, int numStates, const Vector& seq, const Vector& states, Matrix* trans, Matrix* emis);

/** Calculate posteriori probabilities of states at each steps
    Scaled Forward - Backward procedure

    seq: Vector of length L of emissions
    trans: Transition probabilities, size M x M
    emis: Emission probabilities, size M x N

    pstates: size M x L
    fs: scaled forward probabities, size M x L
    bs: scaled backward probabities, size M x L
    scales: scale factors, length L
    RETURN: log probabilities of sequence
*/
double hmm_decodeD(const Vector& seq, const Matrix& trans, const Matrix& emis, Matrix* pstates, Matrix* fs, Matrix* bs, Vector* scales);

/** Calculate the most probable states for a sequence
    Viterbi algorithm
    seq: Vector of length L of emissions
    trans: Transition probabilities, size M x M
    emis: Emission probabilities, size M x N
    
    states: Unitialized, will have length L
    RETURN: log probability of the most probable sequence
*/
double hmm_viterbiD_init(const Vector& seq, const Matrix& trans, const Matrix& emis, Vector* states);

/** Baum-Welch estimation of transition and emission probabilities
    

*/
void hmm_trainD(const ArrayList<Vector>& seqs, Matrix* guessTR, Matrix* guessEM, int max_iter = 500, double tol = 1e-3);

#endif
