/**
 * @file cmaes.h
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

#include <cmath>
#include <limits>
#include <ostream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <iostream>
#include <armadillo>
#include <iostream>

#include "random.hpp"

namespace mlpack {
namespace optimization {

/**
 * @class CMAES 
 * all the function parameters available to the user
 */
template<typename funcType, typename T>
class CMAES
{
public:

  //! constructor to initialize the algorithm parameters
    CMAES(funcType func, size_t dimension = 0, T *start = 0, T *stdDeviation = 0)
  {
    double fitToFind[dimension];
    init(fitToFind, dimension, start, stdDeviation);
  }

   /**
   * Determines the method used to initialize the weights.
   */
  enum Weights
  {
    UNINITIALIZED_WEIGHTS, LINEAR_WEIGHTS, EQUAL_WEIGHTS, LOG_WEIGHTS
  } weightMode;


  //USER FUNCTIONS TO GET PARAMETER IN VALUES AND ARRAYS

  size_t getDimension(void){ return N;}

  void getInitialStart(T *arr, size_t dimension)
  { 
    for(int i=0; i<N; i++) arr[i] = xstart[i];
  }
  
   void getInitialStandardDeviations(T *arr, size_t dimension)
  { 

    for(int i=0; i<N; i++) arr[i] = rgInitialStds[i];
  }

  void setWeights(Weights mode);


  // User defined termination criterias 
    
    void stopMaxFuncEvaluations(T evaluations)
  {
     stopMaxFunEvals = evaluations;
  }

  T getStopMaxFuncEvaluations(void)
  {
    return  stopMaxFunEvals;
  }

  void stopMaxIterations(T iterations)
  {
    stopMaxIter = iterations;
  }

  T getStopMaxIterations(void)
  {
    return stopMaxIter;
  }


  void stopMinFuntionDifference(T difference)
  {
    stopTolFun = difference;
  }

  T getStopMinFunctionDifference(void)
  {
    return stopTolFun;
  }

void stopMinFuntionHistoryDifference(T difference)
  {
    stopTolFunHist = difference;
  }

  T getStopMinFunctionHistoryDifference(void)
  {
    return stopTolFunHist;
  }

  void stopMinStepSize(T size)
  {
    stopTolX = size;
  }

  T getStopMinStepSize(void)
  {
    return stopTolX;
  }

/**
   * A message that contains a detailed description of the matched stop
   * criteria.
   */
  std::string getStopMessage(){return stopMessage;}

  //! other variable parameters

void sampleSize(T l){lambda = l;}

T getSampleSize(void){ return lambda; }

void setMu(T ind){ mu = ind;}

T getMu(void){ return mu;}

void muEffective(T ind){ mueff = ind;}

T getMuEffective(void){ return mueff;}

T axisRatio() { return maxElement(rgD,N) / minElement(rgD,N);};

T evaluation(){ return countevals; }

T fitness(){ return functionValues[index[0]];}

T fitnessBestEver(){ return xBestEver[N];}

T generation(){ return gen;}

T maxAxisLength(){ return sigma*std::sqrt(maxEW);}

T minAxisLength(){ return sigma*std::sqrt(minEW); }

T maxStdDev(){return sigma*std::sqrt(maxdiagC);}

T minStdDev(){return sigma*std::sqrt(mindiagC);}

void diagonalCovariance(T *arr, size_t N)
  {
     for(int i = 0; i < N; ++i)
          arr[i] = C[i][i];
  }

  void diagonalD(T *arr, size_t N) { for(int i = 0; i < N; ++i) arr[i] = rgD[i]; }

  void standardDeviation(T *arr, size_t N)
  {
    for(int i = 0; i < N; ++i) arr[i] = sigma*std::sqrt(C[i][i]);
  }

void XMean(T *arr, size_t N){ for(int i=0; i<N; i++) arr[i] = xmean[i]; }

  ~CMAES()
  {
    if (xstart)
      delete[] xstart;
    if (typicalX)
      delete[] typicalX;
    if (rgInitialStds)
      delete[] rgInitialStds;
    if (rgDiffMinChange)
      delete[] rgDiffMinChange;
    if (weights)
      delete[] weights;

    delete[] pc;
    delete[] ps;
    delete[] tempRandom;
    delete[] BDz;
    delete[] --xmean;
    delete[] --xold;
    delete[] --xBestEver;
    delete[] --output;
    delete[] rgD;
    for(int i = 0; i < N; ++i)
    {
      delete[] C[i];
      delete[] B[i];
    }
    for(int i = 0; i < lambda; ++i)
      delete[] --population[i];
    delete[] population;
    delete[] C;
    delete[] B;
    delete[] index;
    delete[] publicFitness;
    delete[] --functionValues;
    delete[] --funcValueHistory;
  }

private:

void init(T *arr, T dimension, T* inxstart, T* inrgsigma);

void eigen(T* diag, T** Q);

int checkEigen(T* diag, T** Q);

void sortIndex(const T* rgFunVal, int* iindex, int n);

void adaptC2(const int hsig);

void testMinStdDevs(void);

void addMutation(T* x, T eps);

 T maxElement(const T* rgd, int len);

 T minElement(const T* rgd, int len);

 T* const* samplePopulation(void);

 T* const* reSampleSingle(int i);

 T* sampleSingleInto(T* x);

 T const* reSampleSingleOld(T* x);

 T* perturbSolutionInto(T* x, T const* pxmean, T eps);

 T* updateDistribution(const T* fitnessValues);

 bool testForTermination(void);

 void updateEigensystem(bool force);

 T const* setMean(const T* newxmean);

  //! Problem dimension, must stay constant. 
  int N;
  //! Initial search space vector.
  T* xstart;
  //! A typical value for a search space vector.
  T* typicalX;
  //! Indicates that the typical x is the initial point.
  bool typicalXcase;
  //! Initial standard deviations.
  T* rgInitialStds;
  T* rgDiffMinChange;

  /* Termination parameters. */
  //! Maximal number of objective function evaluations.
  T stopMaxFunEvals;
  T facmaxeval;
  //! Maximal number of iterations.
  T stopMaxIter;
  //! Minimal value difference.
  T stopTolFun;
  //! Minimal history value difference.
  T stopTolFunHist;
  //! Minimal search space step size.
  T stopTolX;
  //! Defines the maximal condition number.
  T stopTolUpXFactor;

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
  T mucov;
  /**
   * Variance effective selection mass, should be lambda/4.
   */
  T mueff;
  /**
   * Weights used to recombinate the mean sum up to one.
   */
  T* weights;
  /**
   * Damping parameter for step-size adaption, d = inifinity or 0 means adaption
   * is turned off, usually close to one.
   */
  T damps;
  /**
   * cs^-1 (approx. n/3) is the backward time horizon for the evolution path
   * ps and larger than one.
   */
  T cs;
  T ccumcov;
  /**
   * ccov^-1 (approx. n/4) is the backward time horizon for the evolution path
   * pc and larger than one.
   */
  T ccov;
  T diagonalCov;
  struct { T modulo; T maxtime; } updateCmode;
  T facupdateCmode;

 Random<T> rand;

  //! Step size.
  T sigma;
  //! Mean x vector, "parent".
  T* xmean;
  //! Best sample ever.
  T* xBestEver;
  //! x-vectors, lambda offspring.
  T** population;
  //! Sorting index of sample population.
  int* index;
  //! History of function values.
  T* funcValueHistory;

  T chiN;
  //! Lower triangular matrix: i>=j for C[i][j].
  T** C;
  //! Matrix with normalize eigenvectors in columns.
  T** B;
  //! Axis lengths.
  T* rgD;
  //! Anisotropic evolution path (for covariance).
  T* pc;
  //! Isotropic evolution path (for step length).
  T* ps;
  //! Last mean.
  T* xold;
  //! Output vector.
  T* output;
  //! B*D*z.
  T* BDz;
  //! Temporary (random) vector used in different places.
  T* tempRandom;
  //! Objective function values of the population.
  T* functionValues;
  //!< Public objective function value array returned by init().
  T* publicFitness;

  //! Generation number.
  T gen;
  //! Algorithm state.
  enum {INITIALIZED, SAMPLED, UPDATED} state;

    //! Minimal fitness value. Only activated if flg is true.
  struct { bool flg; T val; } stStopFitness;

  // repeatedly used for output
  T maxdiagC;
  T mindiagC;
  T maxEW;
  T minEW;

  bool eigensysIsUptodate;
  bool doCheckEigen;
  T genOfEigensysUpdate;

  T dMaxSignifKond;

  T dLastMinEWgroesserNull;

  T countevals;

  std::string stopMessage; //!< A message that contains all matched stop criteria.
  
};
} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "cmaes_impl.hpp"

#endif