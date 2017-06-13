#ifndef MLPACK_CORE_OPTIMIZERS_CMAES_PARAMETERS_HPP
#define MLPACK_CORE_OPTIMIZERS_CMAES_PARAMETERS_HPP

#include <cmath>
#include <limits>
#include <ostream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <iostream>

namespace mlpack {
namespace optimization {

template<typename T> class CMAES;

/**
 * @class Parameters
 * Holds all parameters that can be adjusted by the user.
 */
template<typename T>
class Parameters
{
  friend class CMAES<T>;
public:

  /* Input parameters. */
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
  //! Minimal fitness value. Only activated if flg is true.
  struct { bool flg; T val; } stStopFitness;
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

  /**
   * Determines the method used to initialize the weights.
   */
  enum Weights
  {
    UNINITIALIZED_WEIGHTS, LINEAR_WEIGHTS, EQUAL_WEIGHTS, LOG_WEIGHTS
  } weightMode;

  //! File that contains an optimization state that should be resumed.
  std::string resumefile;

  //! Set to true to activate logging warnings.
  bool logWarnings;
  //! Output stream that is used to log warnings, usually std::cerr.
  std::ostream& logStream;

  Parameters()
      : N(-1),
        xstart(0),
        typicalX(0),
        typicalXcase(false),
        rgInitialStds(0),
        rgDiffMinChange(0),
        stopMaxFunEvals(-1),
        facmaxeval(1.0),
        stopMaxIter(-1.0),
        stopTolFun(1e-12),
        stopTolFunHist(1e-13),
        stopTolX(0), // 1e-11*insigma would also be reasonable
        stopTolUpXFactor(1e3),
        lambda(-1),
        mu(-1),
        mucov(-1),
        mueff(-1),
        weights(0),
        damps(-1),
        cs(-1),
        ccumcov(-1),
        ccov(-1),
        facupdateCmode(1),
        weightMode(UNINITIALIZED_WEIGHTS),
        resumefile(""),
        logWarnings(false),
        logStream(std::cerr)
  {
    stStopFitness.flg = false;
    stStopFitness.val = -std::numeric_limits<T>::max();
    updateCmode.modulo = -1;
    updateCmode.maxtime = -1;
  }

  Parameters(const Parameters& parameters)
  {
    assign(parameters);
  }

  ~Parameters()
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
  }

  Parameters& operator=(const Parameters& parameters)
  {
    assign(parameters);
    return *this;
  }


  void init(int dimension = 0, const T* inxstart = 0, const T* inrgsigma = 0)
  {
    if (logWarnings)
    {
      if (!(xstart || inxstart || typicalX))
        logStream << "Warning: initialX undefined. typicalX = 0.5...0.5." << std::endl;
      if (!(rgInitialStds || inrgsigma))
        logStream << "Warning: initialStandardDeviations undefined. 0.3...0.3." << std::endl;
    }

    if (dimension <= 0 && N <= 0)
      throw std::runtime_error("Problem dimension N undefined.");
    else if (dimension > 0)
      N = dimension;

    if (weightMode == UNINITIALIZED_WEIGHTS)
      weightMode = LOG_WEIGHTS;

    diagonalCov = 0; // default is 0, but this might change in future

    if (!xstart)
    {
      xstart = new T[N];
      if (inxstart)
      {
        for (int i = 0; i < N; ++i)
          xstart[i] = inxstart[i];
      }
      else if (typicalX)
      {
        typicalXcase = true;
        for (int i = 0; i < N; ++i)
          xstart[i] = typicalX[i];
      }
      else
      {
        typicalXcase = true;
        for (int i = 0; i < N; i++)
          xstart[i] = 0.5;
      }
    }

    if (!rgInitialStds)
    {
      rgInitialStds = new T[N];
      if (inrgsigma)
      {
        for (int i = 0; i < N; ++i)
          rgInitialStds[i] = inrgsigma[i];
      }
      else
      {
        for (int i = 0; i < N; ++i)
          rgInitialStds[i] = T(0.3);
      }
    }

    supplementDefaults();
  }

private:
  void assign(const Parameters& p)
  {
    N = p.N;

    if (xstart)
      delete[] xstart;
    if (p.xstart)
    {
      xstart = new T[N];
      for (int i = 0; i < N; i++)
        xstart[i] = p.xstart[i];
    }

    if (typicalX)
      delete[] typicalX;
    if (p.typicalX)
    {
      typicalX = new T[N];
      for (int i = 0; i < N; i++)
        typicalX[i] = p.typicalX[i];
    }

    typicalXcase = p.typicalXcase;

    if (rgInitialStds)
      delete[] rgInitialStds;
    if (p.rgInitialStds)
    {
      rgInitialStds = new T[N];
      for (int i = 0; i < N; i++)
        rgInitialStds[i] = p.rgInitialStds[i];
    }

    if (rgDiffMinChange)
      delete[] rgDiffMinChange;
    if (p.rgDiffMinChange)
    {
      rgDiffMinChange = new T[N];
      for (int i = 0; i < N; i++)
        rgDiffMinChange[i] = p.rgDiffMinChange[i];
    }

    stopMaxFunEvals = p.stopMaxFunEvals;
    facmaxeval = p.facmaxeval;
    stopMaxIter = p.stopMaxIter;

    stStopFitness.flg = p.stStopFitness.flg;
    stStopFitness.val = p.stStopFitness.val;

    stopTolFun = p.stopTolFun;
    stopTolFunHist = p.stopTolFunHist;
    stopTolX = p.stopTolX;
    stopTolUpXFactor = p.stopTolUpXFactor;

    lambda = p.lambda;
    mu = p.mu;
    mucov = p.mucov;
    mueff = p.mueff;

    if (weights)
      delete[] weights;
    if (p.weights)
    {
      weights = new T[mu];
      for (int i = 0; i < mu; i++)
        weights[i] = p.weights[i];
    }

    damps = p.damps;
    cs = p.cs;
    ccumcov = p.ccumcov;
    ccov = p.ccov;
    diagonalCov = p.diagonalCov;

    updateCmode.modulo = p.updateCmode.modulo;
    updateCmode.maxtime = p.updateCmode.maxtime;

    facupdateCmode = p.facupdateCmode;

    weightMode = p.weightMode;

    resumefile = p.resumefile;
  }

  /**
   * Supplements default parameter values.
   */
  void supplementDefaults()
  {
    if (lambda < 2)
      lambda = 4 + (int) (3.0*log((double) N));
    if (mu <= 0)
      mu = lambda / 2;
    if (!weights)
      setWeights(weightMode);

    if (cs > 0)
      cs *= (mueff + 2.) / (N + mueff + 3.);
    if (cs <= 0 || cs >= 1)
      cs = (mueff + 2.) / (N + mueff + 3.);

    if (ccumcov <= 0 || ccumcov > 1)
      ccumcov = 4. / (N + 4);

    if (mucov < 1)
      mucov = mueff;
    T t1 = 2. / ((N + 1.4142)*(N + 1.4142));
    T t2 = (2.* mueff - 1.) / ((N + 2.)*(N + 2.) + mueff);
    t2 = (t2 > 1) ? 1 : t2;
    t2 = (1. / mucov)* t1 + (1. - 1. / mucov)* t2;
    if (ccov >= 0)
      ccov *= t2;
    if (ccov < 0 || ccov > 1)
      ccov = t2;

    if (diagonalCov < 0)
      diagonalCov = 2 + 100. * N / sqrt((double) lambda);

    if (stopMaxFunEvals <= 0)
      stopMaxFunEvals = facmaxeval * 900 * (N + 3)*(N + 3);
    else
      stopMaxFunEvals *= facmaxeval;

    if (stopMaxIter <= 0)
      stopMaxIter = ceil((double) (stopMaxFunEvals / lambda));

    if (damps < T(0))
      damps = T(1);
    damps = damps
        * (T(1) + T(2)*std::max(T(0), std::sqrt((mueff - T(1)) / (N + T(1))) - T(1)))
        * (T) std::max(T(0.3), T(1) - // modify for short runs
          (T) N / (T(1e-6) + std::min(stopMaxIter, stopMaxFunEvals / lambda)))
        + cs;

    if (updateCmode.modulo < 0)
      updateCmode.modulo = 1. / ccov / (double) N / 10.;
    updateCmode.modulo *= facupdateCmode;
    if (updateCmode.maxtime < 0)
      updateCmode.maxtime = 0.20; // maximal 20% of CPU-time
  }

  /**
   * Initializes the offspring weights.
   */
  void setWeights(Weights mode)
  {
    if (weights)
      delete[] weights;
    weights = new T[mu];
    switch(mode)
    {
    case LINEAR_WEIGHTS:
      for (int i = 0; i < mu; ++i) weights[i] = mu - i;
      break;
    case EQUAL_WEIGHTS:
      for (int i = 0; i < mu; ++i) weights[i] = 1;
      break;
    case LOG_WEIGHTS:
    default:
      for (int i = 0; i < mu; ++i) weights[i] = log(mu + 1.) - log(i + 1.);
      break;
    }

    // normalize weights vector and set mueff
    T s1 = 0, s2 = 0;
    for (int i = 0; i < mu; ++i)
    {
      s1 += weights[i];
      s2 += weights[i]*weights[i];
    }
    mueff = s1*s1/s2;
    for (int i = 0; i < mu; ++i)
      weights[i] /= s1;

    if (mu < 1 || mu > lambda || (mu == lambda && weights[0] == weights[mu - 1]))
      throw std::runtime_error("setWeights(): invalid setting of mu or lambda");
  }
};
} //cmaes
} // optimizer
#endif