/**
 * @file cmaes.hpp
 * @author Kartik Nighania (GSoC 17 mentor Marcus Edel)
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

#include "random.hpp"

namespace mlpack {
namespace optimization {

template<typename funcType>
class CMAES
{
 public:
CMAES(funcType& function, arma::mat& start, arma::mat& stdDivs,
double iters = -1.0, double evalDiff = 1e-14);

int getSampleSize(void){ return lambda;}

int getMu(void){ return mu;}

void diagonalCovariance(arma::mat& arr){ arr = C.diag();}

double Optimize(arma::mat& arr);

 private:
/* Input parameters. */
//! The instantiated function.
funcType& function;

arma::vec arFunvals;

//! Problem dimension, must stay constant.
int N;
//! Initial search space vector.
arma::vec xstart;
//! Indicates that the typical x is the initial point.
bool typicalXcase;
//! Initial standard deviations.
arma::vec rgInitialStds;
/* Termination parameters. */
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

/* internal evolution strategy parameters */
/**
 * Population size. Number of samples per iteration, at least two,
 * generally > 4.
 */
int lambda;
/**
 * Number of individuals used to recompute the mean.
 */
int mu;
double mucov;
/**
 * Variance effective selection mass, should be lambda/4.
 */
double mueff;
/**
 * Weights used to recombinate the mean sum up to one.
 */
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

Random<double> rand;

//! Step size.
double sigma;
//! Mean x vector, "parent".
arma::vec xmean;
//! Best sample ever.
arma::vec xBestEver;
//! x-vectors, lambda offspring.
arma::mat population;
//! Sorting index of sample population.
arma::uvec index;
//! History of function values.
arma::vec funcValueHistory;

double chiN;
//! Lower triangular matrix: i>=j for C[i][j].
arma::mat C;
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
//!< Public objective function value array returned by init().
arma::vec publicFitness;

//! Generation number.
double gen;
//! Algorithm state.
enum {INITIALIZED, SAMPLED, UPDATED} state;

// repeatedly used for output
double maxdiagC;
double mindiagC;
double maxEW;
double minEW;

bool eigensysIsUptodate;
bool doCheckEigen;
double genOfEigensysUpdate;

double dMaxSignifKond;

double dLastMinEWgroesserNull;

double countevals; //!< objective function evaluations

void updateEigensystem(bool force);
void sortIndex(const arma::vec rgFunVal, arma::vec& iindex, int n);
void adaptC2(const int hsig);
void addMutation(double* x, double eps = 1.0);

void init(arma::vec& func);
void samplePopulation();
void updateDistribution(const arma::vec& fitnessValues);

bool testForTermination();
int  checkEigen(arma::vec diag, arma::mat Q);
};
} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "cmaes_impl.hpp"

#endif
