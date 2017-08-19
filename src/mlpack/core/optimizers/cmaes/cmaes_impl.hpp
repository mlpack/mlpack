/**
 * @file cmaes_impl.hpp
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
#ifndef MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_CMAES_CMAES_IMPL_HPP

#include "cmaes.hpp"

namespace mlpack {
namespace optimization {

CMAES::CMAES(const int objectDim,
             const double start,
             const double stdDivs,
             const double iters,
             const double evalEnd,
             const double functionHistory)
      :
        N(-1),
        stopMaxFunEvals(-1),
        facmaxeval(1.0),
        stopMaxIter(-1.0),
        stopTolFun(1e-12),
        stopTolFunHist(1e-13),
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
        istic(0),
        isstarted(1),
        lastdiff(0),
        tictoczwischensumme(0),
        totaltime(0),
        totaltotaltime(0),
        tictoctime(0),
        lasttictoctime(0),
        flatFitness(0)
  {
    stStopFitness.flg = false;
    stStopFitness.val = -std::numeric_limits<double>::max();
    updateCmode.modulo = -1;
    updateCmode.maxtime = -1;

     N = objectDim;
    if ( N <= 0)
    throw std::runtime_error("Problem dimension N undefined.");

    if (evalEnd != 0) stopTolFun = evalEnd;
    if (functionHistory != 0) stopTolFunHist = functionHistory;

    double start1 = start;
    double stdDivs1 = stdDivs;

    if (start1 == 0)
    { Log::Warn << " WARNING: initial start point undefined." <<
     "Please specify if incorrect results detected."
     << "DEFAULT = 0.5...0.5." << std::endl;
     start1 = 0.5;
    }

    if (stdDivs1 == 0)
    {
     Log::Warn << "WARNING: initialStandardDeviations undefined."
     << " Please specify if incorrect results detected. DEFAULT = 0.3...0.3."
     << std::endl;
     stdDivs1 = 0.3;
    }

    xstart.set_size(N);
    xstart.fill(start1);
    rgInitialStds.set_size(N);
    rgInitialStds.fill(stdDivs1);

    diagonalCov = 0;

    if (lambda < 2)
      lambda = 4 + (int) (3.0*log((double) N));
    if (mu <= 0)
      mu = lambda / 2;

    weights.set_size(mu);
    for (int i = 0; i < mu; ++i) weights[i] = log(mu + 1.) - log(i + 1.);

      // normalize weights vector and set mueff
      double s1 = arma::accu(weights);
      double s2 = arma::accu(weights % weights);

      mueff = s1*s1/s2;
      weights /= s1;

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

  template<typename FuncType>
  double CMAES::Optimize(FuncType& function, arma::mat& arr)
  {
    arFunvals.set_size(lambda);
    Init();
    int funNo = function.NumFunctions();

    arma::Col<double> x(N);

    while (!TestForTermination())
    {
      // Generate lambda new search points, sample population
      SamplePopulation();
      arFunvals.fill(0);

      // evaluate the new search points using the given evaluate
      // function by the user
      for (int i = 0; i < lambda; ++i)
      {
       x = population.submat(i, 0, i, N-1).t();

       for (int j = 0; j < funNo; j++)
       arFunvals[i] += function.Evaluate(x, j);
      }

      // update the search distribution used for sampleDistribution()
      UpdateDistribution(arFunvals);
    }

    // get best estimator for the optimum
    arr = xmean;

    double funs = 0;
    for (int j = 0; j < funNo; j++)
    funs += function.Evaluate(xmean, j);

    return funs;
  }

  void CMAES::AdaptC2(const int hsig)
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
   * Initializes the CMA-ES algorithm.
   * @param parameters The CMA-ES parameters in the parameters.h file
   * @return Array of size lambda that can be used to assign fitness values and
   * pass them to updateDistribution()
   */

  void CMAES::Init()
  {
    double trace = arma::accu(arma::pow(rgInitialStds, 2));
    sigma = std::sqrt(trace/N);

    chiN = std::sqrt((double) N) * (1 - (double)1/(4*N) + (double)1/(21*N*N));
    eigensysIsUptodate = true;
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
    xmean.set_size(N);
    xold.set_size(N);
    xBestEver.set_size(N+2);
    xBestEver[N] = std::numeric_limits<double>::max();
    rgD.set_size(N);
    C.set_size(N, N);
    B.set_size(N, N);
    functionValues.set_size(lambda);
    historySize = 10 + (int) ceil(3.*10.*N/lambda);
    funcValueHistory.set_size(historySize);
    index.set_size(lambda);
    for (int i = 0; i < lambda; ++i) index[i] = i;
    population.set_size(lambda, N+1);
    functionValues.fill(std::numeric_limits<double>::max());
    funcValueHistory.fill(std::numeric_limits<double>::max());
    B.zeros();
    B.diag().ones();
    C.zeros();


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

    xmean = xold = xstart;

    for (int i = 0; i < N; ++i)
    xmean[i] += sigma*rgD[i]*mlpack::math::RandNormal();
  }

  /**
   * The search space vectors have a special form: they are arrays with N+1
   * entries. Entry number -1 is the dimension of the search space N.
   * @return A pointer to a "population" of lambda N-dimensional multivariate
   * normally distributed samples.
   */

void CMAES::SamplePopulation()
  {
    bool diag = diagonalCov == 1 || diagonalCov >= gen;

    // calculate eigensystem
    if (!eigensysIsUptodate)
    {
      if (!diag)
        UpdateEigenSystem(false);
      else
      {
        rgD = arma::sqrt(C.diag());
        minEW = rgD.min();
        minEW *= minEW;
        maxEW = rgD.max();
        maxEW *= maxEW;
        eigensysIsUptodate = true;
        totaltime = 0;
        tictoctime = 0;
        lasttictoctime = 0;
        istic = 0;
        lastclock = clock();
        lasttime = time(NULL);
        lastdiff = 0;
        tictoczwischensumme = 0;
        isstarted = 1;;
      }
    }

      for (int iNk = 0; iNk < lambda; ++iNk)
    { // generate scaled random vector D*z
      for (int i = 0; i < N; ++i)
        if (diag)
          population(iNk, i) = xmean[i] + sigma*rgD[i]
          * mlpack::math::RandNormal();
        else
          tempRandom[i] = rgD[i]* mlpack::math::RandNormal();
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

/** 
* Core procedure of the CMA-ES algorithm. Sets a new mean
* value and estimates
* the new covariance matrix and a new step sizefor the normal search
* distribution.
* @param fitnessValues An array of \f$\lambda\f$ function values.
* @return Mean value of the new distribution. 
*/
void CMAES::UpdateDistribution(arma::vec& fitnessValues)
{
    bool diag = diagonalCov == 1 || diagonalCov >= gen;

    if (state == SAMPLED) // function values are delivered here
      countevals += lambda;
    else Log::Warn <<  "updateDistribution(): unexpected state" << std::endl;

    // assign function values
     population.col(N) = functionValues = fitnessValues;

    // Generate index
     int i, j;
    for (i = 1, index[0] = 0; i < lambda; ++i)
    {
      for (j = i; j > 0; --j)
      {
        if (fitnessValues[index[j - 1]] < fitnessValues[i])
          break;
        index[j] = index[j - 1];
      }

      index[j] = i;
    }

    // Test if function values are identical, escape flat fitness
    if (fitnessValues[index[0]] == fitnessValues[index[(int) lambda / 2]])
    {
      sigma *= std::exp(double(0.2) + cs / damps);

      Log::Warn << "Warning: sigma increased due to equal function values"
      << std::endl << "Reconsider the formulation of the objective function"
      << std::endl;

      flatFitness++;
      if (flatFitness == 3) Init();
    }

for (int i = (int)historySize - 1; i > 0; --i)
      funcValueHistory[i] = funcValueHistory[i - 1];
    funcValueHistory[0] = fitnessValues[index[0]];

    // update xbestever
    if (xBestEver[N] > population(index[0], N) || gen == 1)
    {
      for (int i = 0; i <= N; ++i)
      {
        xBestEver[i] = population(index[0], i);
        xBestEver[N+1] = countevals;
      }
    }

    const double sqrtmueffdivsigma = std::sqrt(mueff) / sigma;
    // calculate xmean and rgBDz~N(0,C)
    for (int i = 0; i < N; ++i)
    {
      xold[i] = xmean[i];
      xmean[i] = 0.;
      for (int iNk = 0; iNk < mu; ++iNk)
        xmean[i] += weights[iNk]*population(index[iNk], i);
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
          sum += B(j, i)*BDz[j];
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
          sum += B(i, j)*tempRandom[j];
      }
      ps[i] = invps*ps[i] + sqrtFactor*sum;
    }

    // calculate norm(ps)^2
    double psxps = std::pow(arma::norm(ps), 2);

    // cumulation for covariance matrix (pc) using B*D*z~N(0,C)
    int hsig = std::sqrt(psxps) / std::sqrt(double(1)
    - std::pow(double(1) - cs, double(2)* gen))
        / chiN < double(1.4) + double(2) / (N + 1);
    const double ccumcovinv = 1.-ccumcov;
    const double hsigFactor = hsig*std::sqrt(ccumcov*(double(2)-ccumcov));
    for (int i = 0; i < N; ++i)
      pc[i] = ccumcovinv*pc[i] + hsigFactor*BDz[i];

    // update of C
    AdaptC2(hsig);

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
bool CMAES::TestForTermination()
  {
    double range, fac;
    int iAchse, iKoo;
    int diag = diagonalCov == 1 || diagonalCov >= gen;

    // function value reached
    if ((gen > 1 || state > SAMPLED) && stStopFitness.flg &&
        functionValues[(int)index[0]] <= stStopFitness.val)
    {
      Log::Info << "Fitness: function value " << functionValues[(int)index[0]]
          << " <= stopFitness (" << stStopFitness.val << ")" << std::endl;
      return true;
    }

    // TolFun
    int rangeIndex = (int) std::min((int)gen, historySize-1);
    range = std::max(arma::max(funcValueHistory.subvec(0, rangeIndex)) ,
        functionValues.max()) -
        std::min(arma::min(funcValueHistory.subvec(0, rangeIndex)),
        functionValues.min());

    if (gen > 0 && range <= stopTolFun)
    {
       Log::Info << "TolFun: function value differences " << range
          << " < stopTolFun=" << stopTolFun << std::endl;
       return true;
    }

    // TolFunHist
    if (gen > historySize)
    {
      range = arma::max(funcValueHistory) - arma::min(funcValueHistory);
      if (range <= stopTolFunHist)
      {
         Log::Info << "TolFunHist: history of function value changes " << range
            << " stopTolFunHist=" << stopTolFunHist << std::endl;
         return true;
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
       return true;
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
            return true;
      }
    }

    // Condition of C greater than dMaxSignifKond
    if (maxEW >= minEW* dMaxSignifKond)
    {
       Log::Info << "ConditionNumber: maximal condition number " <<
       dMaxSignifKond << " reached. maxEW=" << maxEW <<  ",minEW="
       << minEW << ",maxdiagC=" << maxdiagC << ",mindiagC="
       << mindiagC << std::endl;
       Init();
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
           Init();

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
        return true;
      }
    }

    if (countevals >= stopMaxFunEvals)
    {
       Log::Info << "MaxFunEvals: conducted function evaluations " << countevals
          << " >= " << stopMaxFunEvals << std::endl;
       return true;
    }
    if (gen >= stopMaxIter)
    {
       Log::Info << "MaxIter: number of iterations " << gen << " >= "
          << stopMaxIter << std::endl;
      return true;
    }

    return false;
  }

  /**
   * Conducts the eigendecomposition of C into B and D such that
   * \f$C = B \cdot D \cdot D \cdot B^T\f$ and \f$B \cdot B^T = I\f$
   * and D diagonal and positive.
   * @param force For force == true the eigendecomposion is conducted even if
   *              eigenvector and values seem to be up to date.
   */
void CMAES::UpdateEigenSystem(bool force)
  {
    Update();

    if (!force)
    {
      if (eigensysIsUptodate)
        return;
      // return on modulo generation number
      if (gen < genOfEigensysUpdate + updateCmode.modulo)
        return;
      // return on time percentage
      if (updateCmode.maxtime < 1.00
          && tictoctime > updateCmode.maxtime* totaltime
          && tictoctime > 0.0002)
        {
          Log::Info << " time return happened " << std::endl;
        return;
      }
    }

    Tic();
    Eigen(rgD, B, tempRandom);
    Toc();

    // find largest and smallest eigenvalue,
    // they are supposed to be sorted anyway
    minEW = rgD.min();
    maxEW = rgD.max();

     rgD = arma::sqrt(rgD);

    eigensysIsUptodate = true;
    genOfEigensysUpdate = gen;
  }

  /**
   * Calculating eigenvalues and vectors.
   * @param rgtmp (input) N+1-dimensional vector for temporal use. 
   * @param diag (output) N eigenvalues. 
   * @param Q (output) Columns are normalized eigenvectors.
   */

  void CMAES::Eigen(arma::vec& diag, arma::mat& Q, arma::vec& rgtmp)
  {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j <= i; ++j)
          Q(i, j) = Q(j, i) = C(i, j);

    Householder(Q, diag, rgtmp);
    Ql(diag, rgtmp, Q);
  }

  /**
   * Symmetric tridiagonal QL algorithm, iterative.
   * Computes the eigensystem from a tridiagonal matrix
   * @param d input: Diagonale of tridiagonal matrix. output: eigenvalues.
   * @param e input: [1..n-1], off-diagonal, output from Householder
   * @param V input: matrix output of Householder. output: basis of
   *          eigenvectors, according to d
   */
  void CMAES::Ql(arma::vec& d, arma::vec& e, arma::mat& V)
  {
    double f(0);
    double tst1(0);
    const double eps(2.22e-16); // 2.0^-52.0 = 2.22e-16

    // shift input e
    e.subvec(0, N-2) = e.subvec(1, N-1);
    e[N-1] = 0.;

    for (int l = 0; l < N; l++)
    {
      // find small subdiagonal element
      const double smallSDElement = std::fabs(d[l]) + std::fabs(e[l]);
      if (tst1 < smallSDElement) tst1 = smallSDElement;
      const double epsTst1 = eps*tst1;
      int m = l;
      while (m < N)
      {
        if (std::fabs(e[m]) <= epsTst1) break;
        m++;
      }

      // if m == l, d[l] is an eigenvalue, otherwise, iterate.
      if (m > l)
      {
        do {
          double h, g = d[l];
          double p = (d[l+1] - g) / (double(2)*e[l]);
          double r = MyHypot(p, double(1));

          // compute implicit shift
          if (p < 0) r = -r;
          const double pr = p+r;
          d[l] = e[l]/pr;
          h = g - d[l];
          const double dl1 = e[l]*pr;
          d[l+1] = dl1;
          for (int i = l+2; i < N; i++) d[i] -= h;
          f += h;

          // implicit QL transformation.
          p = d[m];
          double c(1);
          double c2(1);
          double c3(1);
          double s(0);
          double s2(0);
          for (int i = m-1; i >= l; i--)
          {
            c3 = c2;
            c2 = c;
            s2 = s;
            g = c*e[i];
            h = c*p;
            r = MyHypot(p, e[i]);
            e[i+1] = s*r;
            s = e[i]/r;
            c = p/r;
            p = c*d[i] - s*g;
            d[i+1] = h + s*(c*g + s*d[i]);

            // accumulate transformation.
            for (int k = 0; k < N; k++)
            {
              double& Vki1 = V(k, i+1);
              h = Vki1;
              double& Vki = V(k, i);
              Vki1 = s*Vki + c*h;
              Vki *= c; Vki -= s*h;
            }
          }
          p = -s*s2*c3*e[l+1]*e[l]/dl1;
          e[l] = s*p;
          d[l] = c*p;
        } while (std::fabs(e[l]) > epsTst1);
      }
      d[l] += f;
      e[l] = 0.0;
    }
  }

  /**
   * Householder transformation of a symmetric matrix V into tridiagonal form.
   * Code slightly adapted from the Java JAMA package, function private tred2().
   * @param V input: symmetric nxn-matrix. output: orthogonal transformation
   *          matrix: tridiag matrix == V* V_in* V^t.
   * @param d output: diagonal
   * @param e output: [0..n-1], off diagonal (elements 1..n-1)
   */

  void CMAES::Householder(arma::mat& V, arma::vec& d, arma::vec& e)
  {
      d = V.submat(N-1, 0, N-1, N-1).t();

    // Householder reduction to tridiagonal form
    for (int i = N - 1; i > 0; i--)
    {
      // scale to avoid under/overflow
      double h = 0.0;
      double scale = arma::accu(arma::abs(d));
      if (scale == 0.0)
      {
        e[i] = d[i-1];
          d.subvec(0, i-1) = V.submat(i-1, 0, i-1, i-1);
          V.submat(i, 0, i, i-1).fill(0.0);
          V.submat(0, i, i-1, i).fill(0.0);
      }
      else
      {
        // generate Householder vector
          d.subvec(0, i-1) /= scale;
          h = arma::accu(d.subvec(0, i-1) % d.subvec(0, i-1));

        double& dim1 = d[i-1];
        double f = dim1;
        double g = f > 0 ? -std::sqrt(h) : std::sqrt(h);
        e[i] = scale*g;
        h = h - f* g;
        dim1 = f - g;
        e.subvec(0, i-1).fill(0.0);

        // apply similarity transformation to remaining columns
        for (int j = 0; j < i; j++)
        {
          f = d[j];
          V(j, i) = f;
          double& ej = e[j];
          g = ej + V(j, j)* f;

          for (int k = j + 1; k <= i - 1; k++)
          {
            double& Vkj = V(k, j);
            g += Vkj*d[k];
            e[k] += Vkj*f;
          }

          ej = g;
        }

        f = 0.0;
        for (int j = 0; j < i; j++)
        {
          e[j] /= h;
          f += e[j]* d[j];
        }
        double hh = f / (h + h);
        for (int j = 0; j < i; j++)
        {
          e[j] -= hh*d[j];
        }
        for (int j = 0; j < i; j++)
        {
          f = d[j];
          g = e[j];
          for (int k = j; k <= i - 1; k++)
          {
            V(k, j) -= f*e[k] + g*d[k];
          }
          d[j] = V(i-1, j);
          V(i, j) = 0.0;
        }
      }
      d[i] = h;
    }

    // accumulate transformations
    const int nm1 = N-1;
    for (int i = 0; i < nm1; i++)
    {
      double h;
      double& Vii = V(i, i);
      V(N-1, i) = Vii;
      Vii = 1.0;
      h = d[i+1];
      if (h != 0.0)
      {
        d.subvec(0, i) = V.submat(0, i+1, i, i+1) / h;
        for (int j = 0; j <= i; j++)
        {
          double g = arma::accu(V.submat(0, i+1, i, i+1)
            % V.submat(0, j, i, j));
          V.submat(0, j, i, j) -= g*d.subvec(0, i);
        }
      }
      V.submat(0, i+1, i, i+1).fill(0.0);
    }
      d.subvec(0, N-1) = V.submat(N-1, 0,  N-1, N-1).t();
      V.submat(N-1, 0,  N-1, N-1).fill(0.0);

     V(N-1, N-1) = 1.0;
     e[0] = 0.0;
  }

  double CMAES::MyHypot(double a, double b)
{
  const register double fabsa = std::fabs(a), fabsb = std::fabs(b);
  if (fabsa > fabsb)
  {
    const register double r = b / a;
    return fabsa*std::sqrt(double(1)+r*r);
  }
  else if (b != double(0))
  {
    const register double r = a / b;
    return fabsb*std::sqrt(double(1)+r*r);
  }
  else
    return double(0);
}

  double CMAES::Update()
  {
    double diffc, difft;
    clock_t lc = lastclock;
    time_t lt = lasttime;
    lastclock = clock();
    lasttime = time(NULL);
    diffc = (double) (lastclock - lc) / CLOCKS_PER_SEC;
    difft = difftime(lasttime, lt);
    lastdiff = difft;

    if (diffc > 0 && difft < 1000) lastdiff = diffc;
    totaltime += lastdiff;
    totaltotaltime += lastdiff;
    if (istic)
    {
      tictoczwischensumme += lastdiff;
      tictoctime += lastdiff;
    }
    return lastdiff;
  }

  void CMAES::Tic()
  {
    Update();
    istic = 1;
  }

  double CMAES::Toc()
  {
    Update();
    lasttictoctime = tictoczwischensumme;
    tictoczwischensumme = 0;
    istic = 0;
    return lasttictoctime;
  }

} // namespace optimization
} // namespace mlpack

#endif
