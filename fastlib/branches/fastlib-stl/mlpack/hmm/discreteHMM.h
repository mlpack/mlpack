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

#include <fastlib/fastlib.h>
#include <armadillo>

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
  arma::mat transmission_;

  /** Emission probabilities in each state */
  arma::mat emission_;

 public:
  /** Basic getters */
  const arma::mat& transmission() const { return transmission_; }
  const arma::mat& emission() const { return emission_; }

  /** Setters used when already initialized */
  void setModel(const arma::mat& transmission, const arma::mat& emission);

  /** Initializes from computed transmission and emission matrices */
  void Init(const arma::mat& transmission, const arma::mat& emission);

  /** Initializes by loading from a file */
  void InitFromFile(const char* profile);

  /** Initializes randomly using data as a guide */
  void InitFromData(const std::vector<arma::vec>& list_data_seq, int numstate);

  /** Load from file, used when already initialized */
  void LoadProfile(const char* profile);

  /** Save matrices to file */
  void SaveProfile(const char* profile) const;

  /** Generate a random data sequence of a given length */
  void GenerateSequence(int length, arma::vec& data_seq, arma::vec& state_seq) const;

  /** 
   * Estimate the matrices by a data sequence and a state sequence 
   * Must be already initialized
   */
  void EstimateModel(const arma::vec& data_seq, const arma::vec& state_seq);
  void EstimateModel(int numstate, int numsymbol, const arma::vec& data_seq, const arma::vec& state_seq);

  /** 
   * Decode a sequence into probabilities of each state at each time step
   * using scaled forward-backward algorithm.
   * Also return forward, backward probabilities and scale factors
   */
  void DecodeOverwrite(const arma::vec& data_seq, arma::mat& state_prob_mat, arma::mat& forward_prob_mat, arma::mat& backward_prob_mat, arma::vec& scale_vec) const;

  /** A decode version that initialized the out matrices */
  void DecodeInit(const arma::vec& data_seq, arma::mat& state_prob_mat, arma::mat& forward_prob_mat, arma::mat& backward_prob_mat, arma::vec& scale_vec) const;

  /** Compute the log-likelihood of a sequence */
  double ComputeLogLikelihood(const arma::vec& data_seq) const;

  /** Compute the log-likelihood of a list of sequences */
  void ComputeLogLikelihood(const std::vector<arma::vec>& list_data_seq, std::vector<double>& list_likelihood) const;

  /** Compute the most probable sequence (Viterbi) */
  void ComputeViterbiStateSequence(const arma::vec& data_seq, arma::vec& state_seq) const;

  /** 
   * Train the model with a list of sequences, must be already initialized 
   * using Baum-Welch EM algorithm
   */
  void TrainBaumWelch(const std::vector<arma::vec>& list_data_seq, int max_iteration, double tolerance);

  /** 
   * Train the model with a list of sequences, must be already initialized 
   * using Viterbi algorithm to determine the state sequence of each sequence
   */
  void TrainViterbi(const std::vector<arma::vec>& list_data_seq, int max_iteration, double tolerance);


  ///////// Static helper functions ///////////////////////////////////////

  /**
   * Generating a sequence and states using transition and emission probabilities.
   * L: sequence length
   * trans: Matrix M x M (M states)
   * emis: Matrix M x N (N emissions)
   * seq: uninitialized vector, will have length L
   * states: uninitialized vector, will have length L
   */
  static void GenerateInit(int L, const arma::mat& trans, const arma::mat& emis, arma::vec& seq, arma::vec& states);

  /** Estimate transition and emission probabilities from sequence and states */
  static void EstimateInit(const arma::vec& seq, const arma::vec& states, arma::mat& trans, arma::mat& emis);
  static void EstimateInit(int numSymbols, int numStates, const arma::vec& seq, const arma::vec& states, arma::mat& trans, arma::mat& emis);

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
  static void ForwardProcedure(const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, arma::vec& scales, arma::mat& fs);
  static void BackwardProcedure(const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, const arma::vec& scales, arma::mat& bs);
  static double Decode(const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, arma::mat& pstates, arma::mat& fs, arma::mat& bs, arma::vec& scales);

  /** Calculate the most probable states for a sequence
   * Viterbi algorithm
   * seq: Vector of length L of emissions
   * trans: Transition probabilities, size M x M
   * emis: Emission probabilities, size M x N
   * states: Unitialized, will have length L
   * RETURN: log probability of the most probable sequence
   */
  static double ViterbiInit(const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, arma::vec& states);
  static double ViterbiInit(int L, const arma::vec& seq, const arma::mat& trans, const arma::mat& emis, arma::vec& states);

  /** Baum-Welch estimation of transition and emission probabilities */
  static void Train(const std::vector<arma::vec>& seqs, arma::mat& guessTR, arma::mat& guessEM, int max_iter, double tol);

  /** Viterbi estimation of transition and emission probabilities */
  static void TrainViterbi(const std::vector<arma::vec>& seqs, arma::mat& guessTR, arma::mat& guessEM, int max_iter, double tol);

};

#endif
