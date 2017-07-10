/**
 * @file cmaes_impl.hpp
 * @author Kartik Nighania (GSoC 17 mentor Marcus Edel)
 *
 * Covariance Matrix Adaptation Evolution Strategy
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_IMPL_HPP

#include <mlpack/core.hpp>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <armadillo>
#include <iostream>
#include <cfloat>

#include "cmaes.hpp"
#include "random.hpp"

namespace mlpack {
namespace optimization {

  template<typename funcType>
  CMAES<funcType>::CMAES(funcType& function,
arma::mat& start, arma::mat& stdDivs, 
double iters, double evalDiff)
      :
        function(function),
        N(-1),
        typicalXcase(false),
        stopMaxFunEvals(-1),
        facmaxeval(1.0),
        stopTolFunHist(1e-14),
        stopTolX(0),
        stopTolUpXFactor(1e3),
        lambda(-1),
        mu(-1),
        mucov(-1),
        mueff(-1),
        damps(-1),
        cs(-1),
        ccumcov(-1),
        ccov(-1),
        facupdateCmode(1),
        weightMode(UNINITIALIZED_WEIGHTS)
  {
    stStopFitness.flg = false;
    stStopFitness.val = -std::numeric_limits<double>::max();
    updateCmode.modulo = -1;
    updateCmode.maxtime = -1;

     N = function.NumFunctions();
    if ( N <= 0)
      throw std::runtime_error("Problem dimension N undefined.");
    bool startP  = true;
    bool initDev = true;

    for (int i=0; i<N; i++)
    {
     if (start[i]   < 1.0e-200) startP  = false;
     if (stdDivs[i] < 1.0e-200) initDev = false;
    }

    if (evalDiff =! 1e-14) stopTolFun = evalDiff;
    else
      stopTolFun = 1e-14;

if (!startP)
Log::Warn << " WARNING: initial start point undefined." <<
"Please specify if incorrect results detected."
<< "DEFAULT = 0.5...0.5." << std::endl;

if (!initDev)
Log::Warn << "WARNING: initialStandardDeviations undefined."
<< " Please specify if incorrect results detected. DEFAULT = 0.3...0.3."
<< std::endl;

    if (weightMode == UNINITIALIZED_WEIGHTS)
      weightMode = LOG_WEIGHTS;

    diagonalCov = 0; // default is 0, but this might change in future

      xstart.set_size(N);
      if (startP)
      {
        for (int i = 0; i < N; ++i) xstart[i] = start[i];
      }
     else
      {
        typicalXcase = true;
        for (int i = 0; i < N; i++) xstart[i] = 0.5;
      }

    rgInitialStds.set_size(N);
    if (initDev)
      {
        for (int i = 0; i < N; ++i) rgInitialStds[i] = stdDivs[i];
      }
      else
      {
        for (int i = 0; i < N; ++i) rgInitialStds[i] = double(0.3);
      }

    if (lambda < 2)
      lambda = 4 + (int) (3.0*log((double) N));
    if (mu <= 0)
      mu = lambda / 2;

      weights.set_size(mu);
      switch (weightMode)
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
      double s1 = 0, s2 = 0;
      for (int i = 0; i < mu; ++i)
      {
        s1 += weights[i];
        s2 += weights[i]*weights[i];
      }
      mueff = s1*s1/s2;
      for (int i = 0; i < mu; ++i)
        weights[i] /= s1;

      if (mu < 1 || mu > lambda || (mu == lambda
        && weights[0] == weights[mu - 1]))
      throw std::runtime_error("setWeights(): invalid setting of mu or lambda");

    if (cs > 0)
      cs *= (mueff + 2.) / (N + mueff + 3.);
    if (cs <= 0 || cs >= 1)
      cs = (mueff + 2.) / (N + mueff + 3.);

    if (ccumcov <= 0 || ccumcov > 1)
      ccumcov = 4. / (N + 4);

    if (mucov < 1)
      mucov = mueff;
    double t1 = 2. / ((N + 1.4142)*(N + 1.4142));
    double t2 = (2.* mueff - 1.) / ((N + 2.)*(N + 2.) + mueff);
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

    if (iters <= 0)
      stopMaxIter = ceil((double) (stopMaxFunEvals / lambda));
      else
        stopMaxIter = iters;

    if (damps < double(0))
    {
    damps = double(1); damps = damps *
    (double(1) + double(2)*std::max(double(0), std::sqrt((mueff -
    double(1)) / (N + double(1))) - double(1))) * (double)
    std::max(double(0.3), double(1) - // modify for short runs
    (double) N / (double(1e-6) + std::min(stopMaxIter, stopMaxFunEvals
    / lambda))) + cs;
    }

    if (updateCmode.modulo < 0)
      updateCmode.modulo = 1. / ccov / (double) N / 10.;
    updateCmode.modulo *= facupdateCmode;
    if (updateCmode.maxtime < 0)
      updateCmode.maxtime = 0.20;
  }

  template<typename funcType>
  double CMAES<funcType>::Optimize(arma::mat& arr)
  {
    arFunvals.set_size(lambda);
    init(arFunvals);

  while (!testForTermination())
  {
    // Generate lambda new search points, sample population
    samplePopulation();

    arma::mat x(N, 1);

    // evaluate the new search points using the given evaluate
    // function by the user
    for (int i = 0; i < lambda; ++i)
    {
      x = population.submat(i, 0, i, N-1);
      arFunvals[i] = function.Evaluate(x);
    }

    // update the search distribution used for sampleDistribution()
      updateDistribution(arFunvals);
  }

  // get best estimator for the optimum
arr = xmean;

  return function.Evaluate(xmean);
  }

template<typename funcType>
void CMAES<funcType>::sortIndex(const arma::vec rgFunVal,
arma::vec& iindex, int n)
  {
    int i, j;
    for (i = 1, iindex[0] = 0; i < n; ++i)
    {
      for (j = i; j > 0; --j)
      {
        if (rgFunVal[iindex[j - 1]] < rgFunVal[i])
          break;
        iindex[j] = iindex[j - 1];
      }
      iindex[j] = i;
    }
  }

  template<typename funcType>
  void CMAES<funcType>::adaptC2(const int hsig)
  {
    bool diag = diagonalCov == 1 || diagonalCov >= gen;

    if (ccov != double(0))
    {
      // definitions for speeding up inner-most loop
      const double mucovinv = double(1)/mucov;

      const double commonFactor = ccov *
      (diag ? (N + double(1.5)) / double(3) : double(1));

      const double ccov1 = std::min(commonFactor*
        mucovinv, double(1));

      const double ccovmu = std::min(commonFactor*
        (double(1)-mucovinv), double(1)-ccov1);

      const double sigmasquare = sigma*sigma;

      const double onemccov1ccovmu = double(1)-ccov1-ccovmu;

      const double longFactor = (double(1)-hsig)*
      ccumcov*(double(2)-ccumcov);

      eigensysIsUptodate = false;

      // update covariance matrix
      for (int i = 0; i < N; ++i)
        for (int j = diag ? i : 0; j <= i; ++j)
        {
          double& Cij = C(i, j);
          Cij = onemccov1ccovmu*Cij + ccov1 * (pc[i]*pc[j] + longFactor*Cij);
          for (int k = 0; k < mu; ++k)
          { // additional rank mu update
            Cij += ccovmu*weights[k] * (population(index[k] , i) - xold[i])
                * (population(index[k] , j) - xold[j]) / sigmasquare;
          }
        }
      // update maximal and minimal diagonal value
      maxdiagC = arma::max(C.diag());
      mindiagC = arma::min(C.diag());
    }
  }


  /**
   * Adds the mutation sigma*B*(D*z).
   * @param x Search space vector.
   * @param eps Mutation factor.
   */
  template<typename funcType>
  void CMAES<funcType>::addMutation(double* x, double eps)
  {
    for (int i = 0; i < N; ++i)
      tempRandom[i] = rgD[i] * rand.gauss();
    for (int i = 0; i < N; ++i)
    {
      double sum = 0.0;
      for (int j = 0; j < N; ++j)
        sum += B(i, j)*tempRandom[j];
      x[i] = xmean[i] + eps*sigma*sum;
    }
  }

  /**
   * Initializes the CMA-ES algorithm.
   * @param parameters The CMA-ES parameters in the parameters.h file
   * @return Array of size lambda that can be used to assign fitness values and
   *         pass them to updateDistribution()
   */
  template<typename funcType>
  void CMAES<funcType>::init(arma::vec& func)
  {
    double trace = arma::accu(arma::pow(rgInitialStds, 2));
    sigma = std::sqrt(trace/N);

    chiN = std::sqrt((double) N) * (1 - 1/(4*N) + 1/(21*N*N));
    eigensysIsUptodate = true;
    doCheckEigen = false;
    genOfEigensysUpdate = 0;

    double dtest;
    for (dtest = double(1); dtest && dtest < double(1.1)*dtest;
    dtest *= double(2))
      if (dtest == dtest + double(1))
        break;
    dMaxSignifKond = dtest / double(1000);

    gen = 0;
    countevals = 0;
    state = INITIALIZED;
    dLastMinEWgroesserNull = double(1);

    pc.set_size(N);
    ps.set_size(N);
    tempRandom.set_size(N);
    BDz.set_size(N);
    xmean.set_size(N+2);
    xold.set_size(N+2);
    xold[0] = N;
    ++xold;
    xBestEver.set_size(N+3);
    xBestEver[0] = N;
    ++xBestEver;
    xBestEver[N] = std::numeric_limits<double>::max();
    rgD.set_size(N);
    C.set_size(N, N);
    B.set_size(N, N);
    publicFitness.set_size(lambda);
    functionValues.set_size(lambda+1);
    functionValues[0] = lambda;
    ++functionValues;
    const int historySize = 10 + (int) ceil(3.*10.*N/lambda);
    funcValueHistory.set_size(historySize + 1);
    index = arma::linspace<arma::uvec>(0, lambda-1, lambda);
    population.zeros(lambda, N+2);
    functionValues.fill(DBL_MAX);
    funcValueHistory.fill(DBL_MAX);
    funcValueHistory[0] = (double) historySize;
    funcValueHistory++;
    C.zeros();
    B.zeros();
    B.diag().ones();

    rgD = rgInitialStds * std::sqrt(N / trace);
    C.diag() = rgD;
    arma::pow(C.diag(), 2);
    pc.zeros();
    ps.zeros();

    minEW = rgD.min();
    minEW = minEW*minEW;
    maxEW = rgD.max();
    maxEW = maxEW*maxEW;

    maxdiagC = arma::max(C.diag());
    mindiagC = arma::min(C.diag());

      xmean = xold;
      xmean = xstart;

    if (typicalXcase)
     xmean += sigma * rgD * rand.gauss();

    func.subvec(0, lambda - 1) = publicFitness.subvec(0, lambda - 1);
  }

  /**
   * The search space vectors have a special form: they are arrays with N+1
   * entries. Entry number -1 is the dimension of the search space N.
   * @return A pointer to a "population" of lambda N-dimensional multivariate
   * normally distributed samples.
   */
template<typename funcType>
void CMAES<funcType>::samplePopulation()
  {
    bool diag = diagonalCov == 1 || diagonalCov >= gen;

    // calculate eigensystem
    if (!eigensysIsUptodate)
    {
      if (!diag)
        updateEigensystem(false);
      else
      {
        rgD = arma::sqrt(C.diag());
        minEW = rgD.min();
        minEW *= minEW;
        maxEW = rgD.max();
        maxEW *= maxEW;
        eigensysIsUptodate = true;
      }
    }

    for (int iNk = 0; iNk < lambda; ++iNk)
    { // generate scaled random vector D*z
      for (int i = 0; i < N; ++i)
        if (diag)
          population(iNk, i) = xmean[i] + sigma*rgD[i] * rand.gauss();
        else
          tempRandom[i] = rgD[i]* rand.gauss();
      if (!diag)
      {
        for (int i = 0; i < N; ++i)
      { // add mutation sigma*B*(D*z)
        double sum = 0.0;
        {
          sum = arma::dot(B.row(i), tempRandom);
          population(iNk , i) = xmean[i] + sigma*sum;
        }
      }
    }
    }

    if (state == UPDATED || gen == 0)
      ++gen;
    state = SAMPLED;
  }

  /**    * Core procedure of the CMA-ES algorithm. Sets a new mean
value and estimates    * the new covariance matrix and a new step size
for the normal search    * distribution.    * @param fitnessValues An
array of \f$\lambda\f$ function values.    * @return Mean value of the
new distribution.    */    template<typename funcType>   void
CMAES<funcType>::updateDistribution(const arma::vec& fitnessValues)
{     bool diag = diagonalCov == 1 || diagonalCov >= gen;

    assert(state != UPDATED && "updateDistribution(): You need to call "
          "samplePopulation() before update can take place.");

    if (state == SAMPLED) // function values are delivered here
      countevals += lambda;
    else Log::Warn <<  "updateDistribution(): unexpected state" << std::endl;

    // assign function values
      population.col(N) = functionValues = fitnessValues;

    // Generate index
    index = arma::sort_index(fitnessValues);

    // Test if function values are identical, escape flat fitness
    if (fitnessValues[index[0]] == fitnessValues[index[(int) lambda / 2]])
    {
      sigma *= std::exp(double(0.2) + cs / damps);
     
        Log::Warn << "Warning: sigma increased due to equal function values"
         << std::endl << "Reconsider the formulation of the objective function";
  
    }

    // update function value history
    for (int i = (int)funcValueHistory.size() - 1; i > 0; --i)
      funcValueHistory[i] = funcValueHistory[i - 1];
    funcValueHistory[0] = fitnessValues[index[0]];

    // update xbestever
    if (xBestEver[N] > population(index[0],N) || gen == 1)
    {
        xBestEver.subvec(0,N-1) = population.submat(index[0], 0, index[0], N-1).t();
        xBestEver[N+1] = countevals;
    }

    const double sqrtmueffdivsigma = std::sqrt(mueff) / sigma;
    // calculate xmean and rgBDz~N(0,C)
    for (int i = 0; i < N; ++i)
    {
      xold[i] = xmean[i];
      xmean[i] = 0.;
      for (int iNk = 0; iNk < mu; ++iNk)
        xmean[i] += weights[iNk]*population(index[iNk] , i);
      BDz[i] = sqrtmueffdivsigma*(xmean[i]-xold[i]);
    }

    // calculate z := D^(-1)* B^(-1)* rgBDz into rgdTmp
    for (int i = 0; i < N; ++i)
    {
      double sum;
      if (diag)
        sum = BDz[i];
      else
      {
        sum = 0.;
        for (int j = 0; j < N; ++j)
          sum += B(j,i)*BDz[j];
      }
      tempRandom[i] = sum/rgD[i];
    }

    // cumulation for sigma (ps) using B*z
    const double sqrtFactor = std::sqrt(cs*(double(2)-cs));
    const double invps = double(1)-cs;
    for (int i = 0; i < N; ++i)
    {
      double sum;
      if (diag)
        sum = tempRandom[i];
      else
      {
        sum = double(0);
        for (int j = 0; j < N; ++j)
          sum += B(i,j)*tempRandom[j];
      }
      ps[i] = invps*ps[i] + sqrtFactor*sum;
    }

    // calculate norm(ps)^2
    double psxps = std::pow(arma::norm(ps),2);

    // cumulation for covariance matrix (pc) using B*D*z~N(0,C)
    int hsig = std::sqrt(psxps) / std::sqrt(double(1) - std::pow(double(1) - cs, double(2)* gen))
        / chiN < double(1.4) + double(2) / (N + 1);
    const double ccumcovinv = 1.-ccumcov;
    const double hsigFactor = hsig*std::sqrt(ccumcov*(double(2)-ccumcov));
    for (int i = 0; i < N; ++i)
      pc[i] = ccumcovinv*pc[i] + hsigFactor*BDz[i];

    // update of C
    adaptC2(hsig);

    // update of sigma
    sigma *= std::exp(((std::sqrt(psxps) / chiN) - double(1))* cs / damps);

    state = UPDATED;
  }

  /**
   * Some stopping criteria can be set in initials.par, with names starting
   * with stop... Internal stopping criteria include a maximal condition number
   * of about 10^15 for the covariance matrix and situations where the numerical
   * discretisation error in x-space becomes noticeably. You can get a message
   * that contains the matched stop criteria via getStopMessage().
   * @return Does any stop criterion match?
   */
template<typename funcType>
bool CMAES<funcType>::testForTermination()
  {
    double range, fac;
    int iAchse, iKoo;
    int diag = diagonalCov == 1 || diagonalCov >= gen;

    bool end = false;

    // function value reached
    if ((gen > 1 || state > SAMPLED) && stStopFitness.flg &&
        functionValues[(int)index[0]] <= stStopFitness.val)
    {
      Log::Info << "Fitness: function value " << functionValues[(int)index[0]]
          << " <= stopFitness (" << stStopFitness.val << ")" << std::endl;
      end = true;
    }

    // TolFun
    range = std::max(maxElement(funcValueHistory,
      (int) std::min(gen, (double)funcValueHistory.size())),
        arma::max(functionValues))-
        std::min(minElement(funcValueHistory,
        (int) std::min(gen, (double)funcValueHistory.size())),
        arma::min(functionValues));

    if (gen > 0 && range <= stopTolFun)
    {
       Log::Info << "TolFun: function value differences " << range
          << " < stopTolFun=" << stopTolFun << std::endl;
       end = true;
    }

    // TolFunHist
    if (gen > funcValueHistory.size())
    {
      range = arma::max(funcValueHistory) - arma::min(funcValueHistory);
      if (range <= stopTolFunHist)
      {
         Log::Info << "TolFunHist: history of function value changes " << range
            << " stopTolFunHist=" << stopTolFunHist << std::endl;
         end = true;
      }
    }

    // TolX
    arma::uvec x = arma::find((sigma*arma::sqrt(C.diag())) < stopTolX);
    arma::uvec y = arma::find(sigma*pc < stopTolX);
    int cTemp = x.n_rows + y.n_rows;

    if (cTemp == 2*N)
    {
       Log::Info << "TolX: object variable changes below "
       << stopTolX << std::endl;
       end = true;
    }

    // TolUpX
    for (int i = 0; i < N; ++i)
    {
      if (sigma*std::sqrt(C(i, i)) > stopTolUpXFactor*rgInitialStds[i])
      {
         Log::Info << "TolUpX: standard deviation increased by more than "
            << stopTolUpXFactor << ", larger initial standard"
            << "deviation recommended."
            << std::endl;
            end = true;
        break;
      }
    }

    // Condition of C greater than dMaxSignifKond
    if (maxEW >= minEW* dMaxSignifKond)
    {
       Log::Info << "ConditionNumber: maximal condition number " <<
       dMaxSignifKond << " reached. maxEW=" << maxEW <<  ",minEW="
       << minEW << ",maxdiagC=" << maxdiagC << ",mindiagC="
       << mindiagC << std::endl;
        end = true;
    }

    // Principal axis i has no effect on xmean
    // ie. x == x + 0.1* sigma* rgD[i]* B[i]
    if (!diag)
    {
      for (iAchse = 0; iAchse < N; ++iAchse)
      {
        fac = 0.1* sigma* rgD[iAchse];
        for (iKoo = 0; iKoo < N; ++iKoo)
        {
          if (xmean[iKoo] != xmean[iKoo] + fac* B(iKoo, iAchse))
            break;
        }
        if (iKoo == N)
        {
           Log::Info << "NoEffectAxis: standard deviation 0.1*" << (fac / 0.1)
           << " in principal axis " << iAchse << " without effect"
           << std::endl;
           end = true;
          break;
        }
      }
    }
    // Component of xmean is not changed anymore
    for (iKoo = 0; iKoo < N; ++iKoo)
    {
      if (xmean[iKoo] == xmean[iKoo] + sigma*
        std::sqrt(C(iKoo, iKoo))/double(5))
      {
         Log::Info << "NoEffectCoordinate: standard deviation 0.2*"
            << (sigma*std::sqrt(C(iKoo , iKoo))) << " in coordinate " << iKoo
            << " without effect" << std::endl;
        end = true;
        break;
      }
    }

    if (countevals >= stopMaxFunEvals)
    {
       Log::Info << "MaxFunEvals: conducted function evaluations " << countevals
          << " >= " << stopMaxFunEvals << std::endl;
       end = true;
    }
    if (gen >= stopMaxIter)
    {
       std::cout << "MaxIter: number of iterations " << gen << " >= "
          << stopMaxIter << std::endl;
      end = true;
    }

    return end;
  }

  /**
   * Conducts the eigendecomposition of C into B and D such that
   * \f$C = B \cdot D \cdot D \cdot B^T\f$ and \f$B \cdot B^T = I\f$
   * and D diagonal and positive.
   * @param force For force == true the eigendecomposion is conducted even if
   *              eigenvector and values seem to be up to date.
   */
template<typename funcType>
void CMAES<funcType>::updateEigensystem(bool force)
  {
    if (!force)
    {
      if (eigensysIsUptodate)
        return;
      // return on modulo generation number
      if (gen < genOfEigensysUpdate + updateCmode.modulo)
        return;
    }

     if (!arma::eig_sym(rgD, B, C))
        Log::Warn << "eigen decomposition failed in neuro_cmaes::eigen()";

    // find largest and smallest eigenvalue, they are
    // supposed to be sorted anyway
    minEW = rgD.min();
    maxEW = rgD.max();

    rgD = arma::sqrt(rgD);

    eigensysIsUptodate = true;
    genOfEigensysUpdate = gen;
  }



} // namespace optimization
} // namespace mlpack

#endif
