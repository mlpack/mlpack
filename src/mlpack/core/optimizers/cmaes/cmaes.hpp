/**
 * @file cmaes.hpp
 * @author Marcus Edel
 * @author Kartik Nighania
 *
 * Covariance Matrix Adaptation Evolution Strategy
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_HPP
#define MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_HPP

namespace mlpack {
namespace optimization {

/**
*CMA-ES stands for Covariance Matrix Adaptation Evolution Strategy.
*Evolution strategies (ES) are stochastic, derivative-free methods for 
*numerical optimization of non-linear or non-convex continuous optimization
*problems. They belong to the class of evolutionary algorithms and 
*evolutionary computation.
*
*An evolutionary algorithm is broadly based on the principle
*of biological evolution, namely the repeated interplay
*of variation (via recombination and mutation) and selection: in each 
*generation (iteration) new individuals (candidate solutions, denoted as
*x) are generated by variation, usually in a stochastic
*way, of the current parental individuals. Then, some individuals are
*selected to become the parents in the next generation based on their
*fitness or objective function value f(x). Like
*this, over the generation sequence, individuals with better and better
*f-values are generated.
*/
class CMAES
{
 public:
/**
* Constructor for the CMAES optimizer.
* 
* @param objectDim the dimension of the object
* @param start the initial start point of the optimizer
*        double value
* @param stdDivs the intial standard deviation to choose for the
*        gaussian distribution.
* @param iters the maximum number of iterations to reach the minimum.
*        it may happen that the function gets terminated by reaching 
*        the condition and not using the remaining iterations
* @param evalDiff the change in function value to see if flat fitness
*        is matched which is the condition mostly when minima is reached.
* @param functionHistory check for minimum function value difference from
         the history of function values to check for flat fitness. 
*/ 
CMAES(int objectDim = 0,
double start = 0, double stdDivs = 0,
double iters = 0, double evalEnd = 0, double functionHistory = 0);

//! Population size. Number of samples per iteration
size_t const SampleSize() const { return lambda; }
//! modify the number of samples per iterations
int const& SampleSize() { return lambda; }
//! Number of individuals used to recompute the mean.
size_t const getMu() const { return mu; }
//! Modify number of individuals used to recompute the mean.
int const& getMu() { return mu; }
//! the covariance matrix diagonal elements
arma::mat DiagonalCovariance() const { return C.diag(); }
/**
* Optimize the given function using CMAES. The function will be given
* as a parameter along with a armadillo matrix of vector in which
* final function value coordinates will be copied. It also return
* the final objective function value in double after the complete
* optimization process.
*
* @param  function the function to be optimized
* @param  arr to put the final coordinates that are found of
*         each dimension.
* @return inal objective value obtained.
*/
template<typename funcType>
double Optimize(funcType& function, arma::mat& arr);

 private:
//! stores the fitness values of functions
  arma::vec arFunvals;
//! objective function evaluations
  double countevals;
//! Problem dimension, must stay constant.
  int N;
  //! Initial search space vector.
  arma::vec xstart;
  //! Initial standard deviations.
  arma::vec rgInitialStds;
  //! Maximal number of objective function evaluations.
  double stopMaxFunEvals;
  double facmaxeval;
  //! Maximal number of iterations.
  double stopMaxIter;
  //! Minimal fitness value. Only activated if flg is true.
  struct { bool flg; double val; } stStopFitness;
  //! Minimal value difference.
  double stopTolFun;
  //! Minimal history value difference.
  double stopTolFunHist;
  //! Minimal search space step size.
  double stopTolX;
  //! Defines the maximal condition number.
  double stopTolUpXFactor;
  //! Population size. Number of samples per iteration.
  int lambda;
  //! Number of individuals used to recompute the mean.
  int mu;
  //! variable used to recompute the mean.
  double mucov;
  //! Variance effective selection mass, should be lambda/4.
  double mueff;
  //! Weights used to recombinate the mean sum up to one.
  arma::vec weights;
  /**
  * Damping parameter for step-size adaption, d = inifinity or 0 means adaption
  * is turned off, usually close to one.
  */
  double damps;
  /**
  * cs^-1 (approx. n/3) is the backward time horizon for the evolution path
  * ps and larger than one.
  */
  double cs;
  double ccumcov;
  /**
  * ccov^-1 (approx. n/4) is the backward time horizon for the evolution path
  * pc and larger than one.
  */
  double ccov;
  double diagonalCov;
  struct { double modulo; double maxtime; } updateCmode;
  double facupdateCmode;
  //! Step size.
  double sigma;
  //! Mean x vector, "parent".
  arma::vec xmean;
  //! Best sample ever.
  arma::vec xBestEver;
  //! x-vectors, lambda offspring.
  arma::mat population;
  //! Sorting index of sample population.
  arma::vec index;
  //! History of function values.
  arma::vec funcValueHistory;
  int historySize;
  //! Lower triangular matrix: i>=j for C[i][j].
  arma::mat C;
  double chiN;
  //! Matrix with normalize eigenvectors in columns.
  arma::mat B;
  //! Axis lengths.
  arma::vec rgD;
  //! Anisotropic evolution path (for covariance).
  arma::vec pc;
  //! Isotropic evolution path (for step length).
  arma::vec ps;
  //! Last mean.
  arma::vec xold;
  //! B*D*z.
  arma::vec BDz;
  //! Temporary (random) vector used in different places.
  arma::vec tempRandom;
  //! Objective function values of the population.
  arma::vec functionValues;
  //! Generation number.
  double gen;
  //! variable for noting time sequences and delay
  clock_t lastclock = clock();
  time_t lasttime = time(NULL);
  clock_t ticclock;
  time_t tictime;
  short istic = 0;
  short isstarted = 1;
  double lastdiff = 0;
  double tictoczwischensumme = 0;
  double totaltime = 0;
  double totaltotaltime = 0;
  double tictoctime = 0;
  double lasttictoctime = 0;
  //! Algorithm state.
  enum {INITIALIZED, SAMPLED, UPDATED} state;
  //! maximum value in convariance matrix diagonal elements.
  double maxdiagC;
  double maxEW;
  //! min values in covariance matrix diagonal elements.
  double mindiagC;
  double minEW;
  double dLastMinEWgroesserNull;
  //! variable maintained for eigen value decomposition and updates.
  bool eigensysIsUptodate;
  double genOfEigensysUpdate;
  //! variable maintained for precision test
  double dMaxSignifKond;

//! eigen value decomposition and update
void updateEigensystem(bool force);
//! adapt the covariance matrix to the new distribution
void adaptC2(const int hsig);
//! initialize all the variables used in CMAES with default values
void init();
//! creates the population from a gaussian normal distribution = lambda
void samplePopulation();
//! updates the distribution according to the best fitness value selected.
void updateDistribution(arma::vec& fitnessValues);
//! test for termination of the algorithm if the condition values are reached.
bool testForTermination();
  /**
   * Calculating eigenvalues and vectors.
   * @param rgtmp (input) N+1-dimensional vector for temporal use. 
   * @param diag (output) N eigenvalues. 
   * @param Q (output) Columns are normalized eigenvectors.
   */
void eigen(arma::vec& diag, arma::mat& Q, arma::vec& rgtmp);
   /**
   * Symmetric tridiagonal QL algorithm.
   * Computes the eigensystem from a tridiagonal matrix.
   * @param d input: Diagonals of matrix. output: eigenvalues.
   * @param e input: [1..n-1], off-diagonal, output from Householder
   * @param V input: matrix output of Householder. output: basis of
   *          eigenvectors, according to d
   */
void ql(arma::vec& d, arma::vec& e, arma::mat& V);
  /**
   * Householder transformation of a symmetric matrix V into tridiagonal form.
   * @param V input: symmetric nxn-matrix. output: orthogonal transformation
   *          matrix: tridiag matrix == V* V_in* V^t.
   * @param d output: diagonal
   * @param e output: [0..n-1], off diagonal (elements 1..n-1)
   */
void householder(arma::mat& V, arma::vec& d, arma::vec& e);
double myhypot(double a, double b);
//! start timing
void tic();
//! stop timing
double toc();
//! @return time between last call of timings_*() and now
double update();
};
} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "cmaes_impl.hpp"

#endif
