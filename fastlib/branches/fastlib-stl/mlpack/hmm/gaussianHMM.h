/**
 * @file gaussianHMM.h
 *
 * This file contains functions of a Gaussian Hidden Markov Models. 
 * It implements log-likelihood computation, viterbi algorithm for the most 
 * probable sequence, Baum-Welch algorithm and Viterbi-like algorithm for parameter 
 * estimation. It can also generate sequences from a Hidden Markov  Model.
 */

#ifndef FASTLIB_GAUSSIAN_HMM_H
#define FASTLIB_GAUSSIAN_HMM_H

#include <fastlib/fastlib.h>
#include <armadillo>

/**
 * A wrapper class for HMM functionals in single Gaussian case
 *
 * This class maintains transition probabilities and Gaussian parameters (mean and
 * covariance) for each state, more details below.
 */

class GaussianHMM {
  //////////// Member variables ////////////////////////////////////////////////
 private:
  /** Transmission probabilities matrix between states */
  arma::mat transmission_;

  /** List of mean vectors */
  std::vector<arma::vec> list_mean_vec_;

  /** List of covariance matrices */
  std::vector<arma::mat> list_covariance_mat_;
  
  /** List of inverse of the covariances */
  std::vector<arma::mat> list_inverse_cov_mat_;

  /** Vector of constant in the gaussian density fomular */
  arma::vec gauss_const_vec_;

  /** Calculate the inverse covariance and the constant in gaussian fomular */
  void CalculateInverse();
 public:
  /** Getters */
  const arma::mat& transmission() const { return transmission_; }
  const std::vector<arma::vec>& list_mean_vec() const { return list_mean_vec_; }
  const std::vector<arma::mat>& list_covariance_mat() const { return list_covariance_mat_; }

  /** Setter used when already initialized */
  void setModel(const arma::mat& transmission,  const std::vector<arma::vec>& list_mean_vec,
		const std::vector<arma::mat>& list_covariance_mat);

  /** Initializes from computed transmission and Gaussian parameters */
  void Init(const arma::mat& transmission,  const std::vector<arma::vec>& list_mean_vec,
	    const std::vector<arma::mat>& list_covariance_mat);
  
  /** Initializes by loading from a file */
  void InitFromFile(const char* profile);

  /** Initializes using K-means algorithm using data as a guide */
  void InitFromData(const std::vector<arma::mat>& list_data_seq, int numstate);

  /** Initializes using data and state sequence as a guide */  
  void InitFromData(const arma::mat& data_seq, const arma::vec& state_seq);

  /** Load from file, used when already initialized */
  void LoadProfile(const char* profile);

  /** Save matrices to file */
  void SaveProfile(const char* profile) const;

  /** Generate a random data sequence of a given length */
  void GenerateSequence(int L, arma::mat& data_seq, arma::vec& state_seq) const;

  /** 
   * Estimate the matrices by a data sequence and a state sequence 
   * Must be already initialized
   */

  void EstimateModel(const arma::mat& data_seq, const arma::vec& state_seq);
  void EstimateModel(int numstate, 
		     const arma::mat& data_seq, const arma::vec& state_seq);

  /** 
   * Decode a sequence into probabilities of each state at each time step
   * using scaled forward-backward algorithm.
   * Also return forward, backward probabilities and scale factors
   */
  void DecodeOverwrite(const arma::mat& data_seq, arma::mat& state_prob_mat, arma::mat& forward_prob_mat, 
		       arma::mat& backward_prob_mat, arma::vec& scale_vec) const;

  /** A decode version that initialized the output matrices */
  void DecodeInit(const arma::mat& data_seq, arma::mat& state_prob_mat, arma::mat& forward_prob_mat, 
		  arma::mat& backward_prob_mat, arma::vec& scale_vec) const;

  /** Compute the log-likelihood of a sequence */
  double ComputeLogLikelihood(const arma::mat& data_seq) const;

  /** Compute the log-likelihood of a list of sequences */
  void ComputeLogLikelihood(const std::vector<arma::mat>& list_data_seq, 
			    std::vector<double>& list_likelihood) const;
  
  /** Compute the most probable sequence (Viterbi) */
  void ComputeViterbiStateSequence(const arma::mat& data_seq, arma::vec& state_seq) const;

  /** 
   * Train the model with a list of sequences, must be already initialized 
   * using Baum-Welch EM algorithm
   */
  void TrainBaumWelch(const std::vector<arma::mat>& list_data_seq, 
		      int max_iteration, double tolerance);

  /** 
   * Train the model with a list of sequences, must be already initialized 
   * using Viterbi algorithm to determine the state sequence of each sequence
   */
  void TrainViterbi(const std::vector<arma::mat>& list_data_seq, 
		    int max_iteration, double tolerance);


  ////////// Static helper functions ///////////////////////////////////////

  static success_t LoadProfile(const char* profile, arma::mat& trans, 
			       std::vector<arma::vec>& means, std::vector<arma::mat>& covs);
  static success_t SaveProfile(const char* profile, const arma::mat& trans, 
			       const std::vector<arma::vec>& means, 
			       const std::vector<arma::mat>& covs);
  /**
   * Generating a sequence and states using transition and emission probabilities.
   * L: sequence length
   * trans: Matrix M x M (M states)
   * means: list of mean vectors length N (emission vector of length N)
   * covs: list of square root of covariance matrices size N x N
   * seq: generated sequence, uninitialized matrix, will have size N x L
   * states: generated states, uninitialized vector, will have length L
   */
  static void GenerateInit(int L, const arma::mat& trans, const std::vector<arma::vec>& means, 
			   const std::vector<arma::mat>& covs, arma::mat& seq, arma::vec& states);

  /** Estimate transition and emission distribution from sequence and states */
  static void EstimateInit(const arma::mat& seq, const arma::vec& states, arma::mat& trans, 
			   std::vector<arma::vec>& means, std::vector<arma::mat>& covs);
  static void EstimateInit(int numStates, const arma::mat& seq, const arma::vec& states,
			   arma::mat& trans, std::vector<arma::vec>& means, 
			   std::vector<arma::mat>& covs);

  /** 
   * Calculate posteriori probabilities of states at each steps
   * Scaled Forward - Backward procedure
   * trans: Transition probabilities, size M x M
   * emis_prob: Emission probabilities along the sequence, 
   *            size M x L (L is the sequence length)
   * pstates: size M x L
   * fs: scaled forward probabities, size M x L
   * bs: scaled backward probabities, size M x L
   * scales: scale factors, length L
   * RETURN: log probabilities of sequence
   */
  static void ForwardProcedure(int L, const arma::mat& trans, const arma::mat& emis_prob, 
			       arma::vec& scales, arma::mat& fs);
  static void BackwardProcedure(int L, const arma::mat& trans, const arma::mat& emis_prob, 
				const arma::vec& scales, arma::mat& bs);
  static double Decode(const arma::mat& trans, const arma::mat& emis_prob, arma::mat& pstates, 
		       arma::mat& fs, arma::mat& bs, arma::vec& scales);
  static double Decode(int L, const arma::mat& trans, const arma::mat& emis_prob, 
		       arma::mat& pstates, arma::mat& fs, arma::mat& bs, arma::vec& scales);
  static void CalculateEmissionProb(const arma::mat& seq, const std::vector<arma::vec>& means, 
				    const std::vector<arma::mat>& inv_covs, const arma::vec& det,
				    arma::mat& emis_prob);

  /** 
   * Calculate the most probable states for a sequence
   * Viterbi algorithm
   * trans: Transition probabilities, size M x M
   * emis_prob: Emission probabilities, size M x L
   * states: Unitialized, will have length L
   * RETURN: log probability of the most probable sequence
   */
  static double ViterbiInit(const arma::mat& trans, const arma::mat& emis_prob, arma::vec& states);
  static double ViterbiInit(int L, const arma::mat& trans, const arma::mat& emis_prob, arma::vec& states);

  /** 
   * Baum-Welch and Viterbi estimation of transition and emission 
   * distribution (Gaussian)
   */
  static void InitGaussParameter(int M, const std::vector<arma::mat>& seqs, 
				 arma::mat& guessTR, std::vector<arma::vec>& guessME, std::vector<arma::mat>& guessCO);

  static void Train(const std::vector<arma::mat>& seqs, arma::mat& guessTR, 
		    std::vector<arma::vec>& guessME, std::vector<arma::mat>& guessCO, 
		    int max_iter, double tol);

  static void TrainViterbi(const std::vector<arma::mat>& seqs, arma::mat& guessTR, 
			   std::vector<arma::vec>& guessME, std::vector<arma::mat>& guessCO, 
			   int max_iter, double tol);
};

#endif
