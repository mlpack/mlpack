/**
 * @file discreteHMM.h
 *
 * This file contains functions of a discrete Hidden Markov Models (in C) and a
 * wrapper class (in C++) for these functions. It implements log-likelihood
 * computation, viterbi algorithm for the most probable sequence, Baum-Welch
 * algorithm and Viterbi-like algorithm for parameter estimation. It can also
 * generate sequences from a Hidden Markov  Model.
 */

#ifndef FASTLIB_DISCRETE_HMM_H
#define FASTLIB_DISCRETE_HMM_H

#include "fastlib/fastlib.h"

/**
 * A wrapper class for HMM functionals in discrete case
 * 
 * This class maintains transition probabilities and emission probabilities
 * matrices and performs basic HMM functionals, more details below.
 * 
 */

class DiscreteHMM {
  /////////// Member variables /////////////////////////////////////
 private:
  /** Transmission probabilities matrix between states */
  Matrix transmission_;

  /** Emission probabilities in each state */
  Matrix emission_;

  OT_DEF(DiscreteHMM) {
    OT_MY_OBJECT(transmission_);
    OT_MY_OBJECT(emission_);
  }

 public:
  /** Basic getters */
  const Matrix& transmission() const { return transmission_; }
  const Matrix& emission() const { return emission_; }

  /** Setters used when already initialized */
  void setModel(const Matrix& transmission, const Matrix& emission);

  /** Initializes from computed transmission and emission matrices */
  void Init(const Matrix& transmission, const Matrix& emission);

  /** Initializes by loading from a file */
  void InitFromFile(const char* profile);

  /** Initializes randomly using data as a guide */
  void InitFromData(const ArrayList<Vector>& list_data_seq, int numstate);

  /** Load from file, used when already initialized */
  void LoadProfile(const char* profile);

  /** Save matrices to file */
  void SaveProfile(const char* profile) const;

  /** Generate a random data sequence of a given length */
  void GenerateSequence(int length, Vector* data_seq, Vector* state_seq) const;

  /** 
   * Estimate the matrices by a data sequence and a state sequence 
   * Must be already initialized
   */
  void EstimateModel(const Vector& data_seq, const Vector& state_seq);
  void EstimateModel(int numstate, int numsymbol, const Vector& data_seq, const Vector& state_seq);

  /** 
   * Decode a sequence into probabilities of each state at each time step
   * using scaled forward-backward algorithm.
   * Also return forward, backward probabilities and scale factors
   */
  void DecodeOverwrite(const Vector& data_seq, Matrix* state_prob_mat, Matrix* forward_prob_mat, Matrix* backward_prob_mat, Vector* scale_vec) const;

  /** A decode version that initialized the out matrices */
  void DecodeInit(const Vector& data_seq, Matrix* state_prob_mat, Matrix* forward_prob_mat, Matrix* backward_prob_mat, Vector* scale_vec) const;

  /** Compute the log-likelihood of a sequence */
  double ComputeLogLikelihood(const Vector& data_seq) const;

  /** Compute the log-likelihood of a list of sequences */
  void ComputeLogLikelihood(const ArrayList<Vector>& list_data_seq, ArrayList<double>* list_likelihood) const;

  /** Compute the most probable sequence (Viterbi) */
  void ComputeViterbiStateSequence(const Vector& data_seq, Vector* state_seq) const;

  /** 
   * Train the model with a list of sequences, must be already initialized 
   * using Baum-Welch EM algorithm
   */
  void TrainBaumWelch(const ArrayList<Vector>& list_data_seq, int max_iteration, double tolerance);

  /** 
   * Train the model with a list of sequences, must be already initialized 
   * using Viterbi algorithm to determine the state sequence of each sequence
   */
  void TrainViterbi(const ArrayList<Vector>& list_data_seq, int max_iteration, double tolerance);


  ///////// Static helper functions ///////////////////////////////////////

  /**
   * Generating a sequence and states using transition and emission probabilities.
   * L: sequence length
   * trans: Matrix M x M (M states)
   * emis: Matrix M x N (N emissions)
   * seq: uninitialized vector, will have length L
   * states: uninitialized vector, will have length L
   */
  static void GenerateInit(int L, const Matrix& trans, const Matrix& emis, Vector* seq, Vector * states);

  /** Estimate transition and emission probabilities from sequence and states */
  static void EstimateInit(const Vector& seq, const Vector& states, Matrix* trans, Matrix* emis);
  static void EstimateInit(int numSymbols, int numStates, const Vector& seq, const Vector& states, Matrix* trans, Matrix* emis);

  /** Calculate posteriori probabilities of states at each steps
   * Scaled Forward - Backward procedure
   * seq: Vector of length L of emissions
   * trans: Transition probabilities, size M x M
   * emis: Emission probabilities, size M x N
   * pstates: size M x L
   * fs: scaled forward probabities, size M x L
   * bs: scaled backward probabities, size M x L
   * scales: scale factors, length L
   * RETURN: log probabilities of sequence
   */
  static void ForwardProcedure(const Vector& seq, const Matrix& trans, const Matrix& emis, Vector *scales, Matrix* fs);
  static void BackwardProcedure(const Vector& seq, const Matrix& trans, const Matrix& emis, const Vector& scales, Matrix* bs);
  static double Decode(const Vector& seq, const Matrix& trans, const Matrix& emis, Matrix* pstates, Matrix* fs, Matrix* bs, Vector* scales);

  /** Calculate the most probable states for a sequence
   * Viterbi algorithm
   * seq: Vector of length L of emissions
   * trans: Transition probabilities, size M x M
   * emis: Emission probabilities, size M x N
   * states: Unitialized, will have length L
   * RETURN: log probability of the most probable sequence
   */
  static double ViterbiInit(const Vector& seq, const Matrix& trans, const Matrix& emis, Vector* states);
  static double ViterbiInit(int L, const Vector& seq, const Matrix& trans, const Matrix& emis, Vector* states);

  /** Baum-Welch estimation of transition and emission probabilities */
  static void Train(const ArrayList<Vector>& seqs, Matrix* guessTR, Matrix* guessEM, int max_iter, double tol);

  /** Viterbi estimation of transition and emission probabilities */
  static void TrainViterbi(const ArrayList<Vector>& seqs, Matrix* guessTR, Matrix* guessEM, int max_iter, double tol);

};
#endif
