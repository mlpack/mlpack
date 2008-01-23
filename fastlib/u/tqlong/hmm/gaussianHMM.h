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

#include "fastlib/fastlib.h"

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
  Matrix transmission_;

  /** List of mean vectors */
  ArrayList<Vector> list_mean_vec_;

  /** List of covariance matrices */
  ArrayList<Matrix> list_covariance_mat_;
  
  /** List of inverse of the covariances */
  ArrayList<Matrix> list_inverse_cov_mat_;

  /** Vector of constant in the gaussian density fomular */
  Vector gauss_const_vec_;

  /** Calculate the inverse covariance and the constant in gaussian fomular */
  void CalculateInverse();
 public:
  /** Getters */
  const Matrix& transmission() const { return transmission_; }
  const ArrayList<Vector>& list_mean_vec() const { return list_mean_vec_; }
  const ArrayList<Matrix>& list_covariance_mat() const { return list_covariance_mat_; }

  /** Setter used when already initialized */
  void setModel(const Matrix& transmission,  const ArrayList<Vector>& list_mean_vec,
		const ArrayList<Matrix>& list_covariance_mat);

  /** Initializes from computed transmission and Gaussian parameters */
  void Init(const Matrix& transmission,  const ArrayList<Vector>& list_mean_vec,
	    const ArrayList<Matrix>& list_covariance_mat);
  
  /** Initializes by loading from a file */
  void InitFromFile(const char* profile);

  /** Initializes using K-means algorithm using data as a guide */
  void InitFromData(const ArrayList<Matrix>& list_data_seq, int numstate);

  /** Initializes using data and state sequence as a guide */  
  void InitFromData(const Matrix& data_seq, const Vector& state_seq);

  /** Load from file, used when already initialized */
  void LoadProfile(const char* profile);

  /** Save matrices to file */
  void SaveProfile(const char* profile) const;

  /** Generate a random data sequence of a given length */
  void GenerateSequence(int L, Matrix* data_seq, Vector* state_seq) const;

  /** 
   * Estimate the matrices by a data sequence and a state sequence 
   * Must be already initialized
   */

  void EstimateModel(const Matrix& data_seq, const Vector& state_seq);
  void EstimateModel(int numstate, 
		     const Matrix& data_seq, const Vector& state_seq);

  /** 
   * Decode a sequence into probabilities of each state at each time step
   * using scaled forward-backward algorithm.
   * Also return forward, backward probabilities and scale factors
   */
  void DecodeOverwrite(const Matrix& data_seq, Matrix* state_prob_mat, Matrix* forward_prob_mat, 
		       Matrix* backward_prob_mat, Vector* scale_vec) const;

  /** A decode version that initialized the output matrices */
  void DecodeInit(const Matrix& data_seq, Matrix* state_prob_mat, Matrix* forward_prob_mat, 
		  Matrix* backward_prob_mat, Vector* scale_vec) const;

  /** Compute the log-likelihood of a sequence */
  double ComputeLogLikelihood(const Matrix& data_seq) const;

  /** Compute the log-likelihood of a list of sequences */
  void ComputeLogLikelihood(const ArrayList<Matrix>& list_data_seq, 
			    ArrayList<double>* list_likelihood) const;
  
  /** Compute the most probable sequence (Viterbi) */
  void ComputeViterbiStateSequence(const Matrix& data_seq, Vector* state_seq) const;

  /** 
   * Train the model with a list of sequences, must be already initialized 
   * using Baum-Welch EM algorithm
   */
  void TrainBaumWelch(const ArrayList<Matrix>& list_data_seq, 
		      int max_iteration, double tolerance);

  /** 
   * Train the model with a list of sequences, must be already initialized 
   * using Viterbi algorithm to determine the state sequence of each sequence
   */
  void TrainViterbi(const ArrayList<Matrix>& list_data_seq, 
		    int max_iteration, double tolerance);


  ////////// Static helper functions ///////////////////////////////////////

  static success_t LoadProfile(const char* profile, Matrix* trans, 
			       ArrayList<Vector>* means, ArrayList<Matrix>* covs);
  static success_t SaveProfile(const char* profile, const Matrix& trans, 
			       const ArrayList<Vector>& means, 
			       const ArrayList<Matrix>& covs);
  /**
   * Generating a sequence and states using transition and emission probabilities.
   * L: sequence length
   * trans: Matrix M x M (M states)
   * means: list of mean vectors length N (emission vector of length N)
   * covs: list of square root of covariance matrices size N x N
   * seq: generated sequence, uninitialized matrix, will have size N x L
   * states: generated states, uninitialized vector, will have length L
   */
  static void GenerateInit(int L, const Matrix& trans, const ArrayList<Vector>& means, 
			   const ArrayList<Matrix>& covs, Matrix* seq, Vector* states);

  /** Estimate transition and emission distribution from sequence and states */
  static void EstimateInit(const Matrix& seq, const Vector& states, Matrix* trans, 
			   ArrayList<Vector>* means, ArrayList<Matrix>* covs);
  static void EstimateInit(int numStates, const Matrix& seq, const Vector& states, 
			   Matrix* trans, ArrayList<Vector>* means, 
			   ArrayList<Matrix>* covs);

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
  static void ForwardProcedure(int L, const Matrix& trans, const Matrix& emis_prob, 
			       Vector *scales, Matrix* fs);
  static void BackwardProcedure(int L, const Matrix& trans, const Matrix& emis_prob, 
				const Vector& scales, Matrix* bs);
  static double Decode(const Matrix& trans, const Matrix& emis_prob, Matrix* pstates, 
		       Matrix* fs, Matrix* bs, Vector* scales);
  static double Decode(int L, const Matrix& trans, const Matrix& emis_prob, 
		       Matrix* pstates, Matrix* fs, Matrix* bs, Vector* scales);
  static void CalculateEmissionProb(const Matrix& seq, const ArrayList<Vector>& means, 
				    const ArrayList<Matrix>& inv_covs, const Vector& det,
				    Matrix* emis_prob);

  /** 
   * Calculate the most probable states for a sequence
   * Viterbi algorithm
   * trans: Transition probabilities, size M x M
   * emis_prob: Emission probabilities, size M x L
   * states: Unitialized, will have length L
   * RETURN: log probability of the most probable sequence
   */
  static double ViterbiInit(const Matrix& trans, const Matrix& emis_prob, Vector* states);
  static double ViterbiInit(int L, const Matrix& trans, const Matrix& emis_prob, Vector* states);

  /** 
   * Baum-Welch and Viterbi estimation of transition and emission 
   * distribution (Gaussian)
   */
  static void InitGaussParameter(int M, const ArrayList<Matrix>& seqs, 
				 Matrix* guessTR, ArrayList<Vector>* guessME, ArrayList<Matrix>* guessCO);

  static void Train(const ArrayList<Matrix>& seqs, Matrix* guessTR, 
		    ArrayList<Vector>* guessME, ArrayList<Matrix>* guessCO, 
		    int max_iter, double tol);

  static void TrainViterbi(const ArrayList<Matrix>& seqs, Matrix* guessTR, 
			   ArrayList<Vector>* guessME, ArrayList<Matrix>* guessCO, 
			   int max_iter, double tol);
};
#endif
