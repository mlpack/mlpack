#ifndef MLPACK_METHODS_NEURO_CMAES_NEURO_CMAES_HPP
#define MLPACK_METHODS_NEURO_CMAES_NEURO_CMAES_HPP

 /**
 * @file neuro_cmaes.hpp
 * @author www.github.com/Kartik-Nighania
 *
 * implementation of CMAES with feed forward neural networks
 */

#include <cstddef>
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

#include "genome.hpp"
#include "neuron_gene.hpp"
#include "link_gene.hpp"
#include "parameters.hpp"


namespace mlpack {
namespace neuro_cmaes {

template<typename T> class CMAES;


template<typename T>
class CMAES{
public:

//USER FUNCTIONS TO GET PARAMETER IN VALUES AND ARRAYS

T axisRatio()
{
 return maxElement(rgD, params.N) / minElement(rgD, params.N);
};

T evaluation(){ return countevals; }

T fitness(){ return functionValues[index[0]];}

T fitnessBestEver(){ return xBestEver[params.N];}

T generation(){ return gen;}

T maxEvaluation(){ return params.stopMaxFunEvals;}

T maxIteration(){ return std::ceil(params.stopMaxIter); }

T maxAxisLength(){ return sigma*std::sqrt(maxEW);}

T minAxisLength(){ return sigma*std::sqrt(minEW); }

T maxStdDev(){return sigma*std::sqrt(maxdiagC);}

T minStdDev(){return sigma*std::sqrt(mindiagC);}

T dimension(){return params.N;}

T sampleSize(){return params.lambda;}

T sigmaValue(){return sigma;}

  T* diagonalCovariance()
  {
     for(int i = 0; i < params.N; ++i)
          output[i] = C[i][i];
        return output;
  }

  T* diagonalD(){ return rgD; }

  T* standardDeviation()
  {
    for(int i = 0; i < params.N; ++i)
          output[i] = sigma*std::sqrt(C[i][i]);
        return output;
  }

 T* XBestEver(){ return xBestEver;}

T* XBest(){return population[index[0]];}

T* XMean(){return xmean;}

private:

  //!< CMA-ES parameters.
  Parameters<T> params;

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

  // repeatedly used for output
  T maxdiagC;
  T mindiagC;
  T maxEW;
  T minEW;

  bool eigensysIsUptodate;
  bool doCheckEigen; //!< control via signals.par
  T genOfEigensysUpdate;

  T dMaxSignifKond;
  T dLastMinEWgroesserNull;

  std::string stopMessage; //!< A message that contains all matched stop criteria.

  /**
   * Calculating eigenvalues and vectors.
   * Also checks for successful eigen decomposition.
   * @param diag (output) N eigenvalues.
   * @param Q (output) Columns are normalized eigenvectors.
   */
  void eigen(T* diag, T** Q)
  { 

     arma::vec eV;
     arma::mat eigMat;

     arma::mat cov(params.N,params.N);
     for(int i=0; i<params.N; i++)
      for(int j=0; j<=i; j++) cov(i,j)=cov(j,i)=C[i][j];


   if(!arma::eig_sym(eV, eigMat, cov))
        assert("eigen decomposition failed in neuro_cmaes::eigen()");

     for(int i=0; i<params.N; i++)
     {
      diag[i]=eV(i);

        for(int j=0; j<params.N; j++)
        Q[i][j]=eigMat(i,j);
      
     }
  
  }
 
  //index sort.

  void sortIndex(const T* rgFunVal, int* iindex, int n)
  {
    int i, j;
    for(i = 1, iindex[0] = 0; i < n; ++i)
    {
      for(j = i; j > 0; --j)
      {
        if(rgFunVal[iindex[j - 1]] < rgFunVal[i])
          break;
        iindex[j] = iindex[j - 1]; 
      }
      iindex[j] = i;
    }
  }

  void adaptC2(const int hsig)
  {
    const int N = params.N;
    bool diag = params.diagonalCov == 1 || params.diagonalCov >= gen;

    if(params.ccov != T(0))
    {
      // definitions for speeding up inner-most loop
      const T mucovinv = T(1)/params.mucov;
      const T commonFactor = params.ccov * (diag ? (N + T(1.5)) / T(3) : T(1));
      const T ccov1 = std::min(commonFactor*mucovinv, T(1));
      const T ccovmu = std::min(commonFactor*(T(1)-mucovinv), T(1)-ccov1);
      const T sigmasquare = sigma*sigma;
      const T onemccov1ccovmu = T(1)-ccov1-ccovmu;
      const T longFactor = (T(1)-hsig)*params.ccumcov*(T(2)-params.ccumcov);

      eigensysIsUptodate = false;

      // update covariance matrix
      for(int i = 0; i < N; ++i)
        for(int j = diag ? i : 0; j <= i; ++j)
        {
          T& Cij = C[i][j];
          Cij = onemccov1ccovmu*Cij + ccov1 * (pc[i]*pc[j] + longFactor*Cij);
          for(int k = 0; k < params.mu; ++k)
          { // additional rank mu update
            const T* rgrgxindexk = population[index[k]];
            Cij += ccovmu*params.weights[k] * (rgrgxindexk[i] - xold[i])
                * (rgrgxindexk[j] - xold[j]) / sigmasquare;
          }
        }
      // update maximal and minimal diagonal value
      maxdiagC = mindiagC = C[0][0];
      for(int i = 1; i < N; ++i)
      {
        const T& Cii = C[i][i];
        if(maxdiagC < Cii)
          maxdiagC = Cii;
        else if(mindiagC > Cii)
          mindiagC = Cii;
      }
    }
  }

  /**
   * Treats minimal standard deviations and numeric problems. Increases sigma.
   */
  void testMinStdDevs(void)
  {
    if(!this->params.rgDiffMinChange)
      return;

    for(int i = 0; i < params.N; ++i)
      while(this->sigma*std::sqrt(this->C[i][i]) < this->params.rgDiffMinChange[i])
        this->sigma *= std::exp(T(0.05) + this->params.cs / this->params.damps);
  }

  /**
   * Adds the mutation sigma*B*(D*z).
   * @param x Search space vector.
   * @param eps Mutation factor.
   */
  void addMutation(T* x, T eps = 1.0)
  {
    for(int i = 0; i < params.N; ++i)
      tempRandom[i] = rgD[i]*(arma::randu(1));
    for(int i = 0; i < params.N; ++i)
    {
      T sum = 0.0;
      for(int j = 0; j < params.N; ++j)
        sum += B[i][j]*tempRandom[j];
      x[i] = xmean[i] + eps*sigma*sum;
    }
  }

public:

  T countevals;

  CMAES()
  {
  }

  /**
   * Releases the dynamically allocated memory, including that of the return
   * value of init().
   */
  ~CMAES()
  {
    delete[] pc;
    delete[] ps;
    delete[] tempRandom;
    delete[] BDz;
    delete[] --xmean;
    delete[] --xold;
    delete[] --xBestEver;
    delete[] --output;
    delete[] rgD;
    for(int i = 0; i < params.N; ++i)
    {
      delete[] C[i];
      delete[] B[i];
    }
    for(int i = 0; i < params.lambda; ++i)
      delete[] --population[i];
    delete[] population;
    delete[] C;
    delete[] B;
    delete[] index;
    delete[] publicFitness;
    delete[] --functionValues;
    delete[] --funcValueHistory;
  }

  /**
   * Initializes the CMA-ES algorithm.
   * @param parameters The CMA-ES parameters.
   * @return Array of size lambda that can be used to assign fitness values and
   *         pass them to updateDistribution(). Not that after the desctructor
   *         was called, the array is deleted.
   */
  T* init(const Parameters<T>& parameters)
  {
    params = parameters;

    stopMessage = "";

    T trace(0);
    for(int i = 0; i < params.N; ++i)
      trace += params.rgInitialStds[i]*params.rgInitialStds[i];
    sigma = std::sqrt(trace/params.N);

    chiN = std::sqrt((T) params.N) * (T(1) - T(1)/(T(4)*params.N) + T(1)/(T(21)*params.N*params.N));
    eigensysIsUptodate = true;
    doCheckEigen = false;
    genOfEigensysUpdate = 0;

    T dtest;
    for(dtest = T(1); dtest && dtest < T(1.1)*dtest; dtest *= T(2))
      if(dtest == dtest + T(1))
        break;
    dMaxSignifKond = dtest / T(1000); // not sure whether this is really safe, 100 does not work well enough

    gen = 0;
    countevals = 0;
    state = INITIALIZED;
    dLastMinEWgroesserNull = T(1);

    pc = new T[params.N];
    ps = new T[params.N];
    tempRandom = new T[params.N+1];
    BDz = new T[params.N];
    xmean = new T[params.N+2];
    xmean[0] = params.N;
    ++xmean;
    xold = new T[params.N+2];
    xold[0] = params.N;
    ++xold;
    xBestEver = new T[params.N+3];
    xBestEver[0] = params.N;
    ++xBestEver;
    xBestEver[params.N] = std::numeric_limits<T>::max();
    output = new T[params.N+2];
    output[0] = params.N;
    ++output;
    rgD = new T[params.N];
    C = new T*[params.N];
    B = new T*[params.N];
    publicFitness = new T[params.lambda];
    functionValues = new T[params.lambda+1];
    functionValues[0] = params.lambda;
    ++functionValues;
    const int historySize = 10 + (int) ceil(3.*10.*params.N/params.lambda);
    funcValueHistory = new T[historySize + 1];
    funcValueHistory[0] = (T) historySize;
    funcValueHistory++;

    for(int i = 0; i < params.N; ++i)
    {
      C[i] = new T[i+1];
      B[i] = new T[params.N];
    }
    index = new int[params.lambda];
    for(int i = 0; i < params.lambda; ++i)
        index[i] = i;
    population = new T*[params.lambda];
    for(int i = 0; i < params.lambda; ++i)
    {
      population[i] = new T[params.N+2];
      population[i][0] = params.N;
      population[i]++;
      for(int j = 0; j < params.N; j++)
        population[i][j] = 0.0;
    }

    // initialize newed space
    for(int i = 0; i < params.lambda; i++)
    {
      functionValues[i] = std::numeric_limits<T>::max();
    }
    for(int i = 0; i < historySize; i++)
    {
      funcValueHistory[i] = std::numeric_limits<T>::max();
    }
    for(int i = 0; i < params.N; ++i)
      for(int j = 0; j < i; ++j)
        C[i][j] = B[i][j] = B[j][i] = 0.;

    for(int i = 0; i < params.N; ++i)
    {
      B[i][i] = T(1);
      C[i][i] = rgD[i] = params.rgInitialStds[i]*std::sqrt(params.N/trace);
      C[i][i] *= C[i][i];
      pc[i] = ps[i] = T(0);
    }
    minEW = minElement(rgD, params.N);
    minEW = minEW*minEW;
    maxEW = maxElement(rgD, params.N);
    maxEW = maxEW*maxEW;

    maxdiagC = C[0][0];
    for(int i = 1; i < params.N; ++i) if(maxdiagC < C[i][i]) maxdiagC = C[i][i];
    mindiagC = C[0][0];
    for(int i = 1; i < params.N; ++i) if(mindiagC > C[i][i]) mindiagC = C[i][i];

    for(int i = 0; i < params.N; ++i)
      xmean[i] = xold[i] = params.xstart[i];
    // use in case xstart as typicalX
    if(params.typicalXcase)
      for(int i = 0; i < params.N; ++i)
        xmean[i] += sigma*rgD[i]*(arma::randu(1));

    return publicFitness;
  }


  /**
   * The search space vectors have a special form: they are arrays with N+1
   * entries. Entry number -1 is the dimension of the search space N.
   * @return A pointer to a "population" of lambda N-dimensional multivariate
   * normally distributed samples.
   */
  T* const* samplePopulation()
  {
    bool diag = params.diagonalCov == 1 || params.diagonalCov >= gen;

    // calculate eigensystem
    if(!eigensysIsUptodate)
    {
      if(!diag)
        updateEigensystem(false);
      else
      {
        for(int i = 0; i < params.N; ++i)
          rgD[i] = std::sqrt(C[i][i]);
        minEW = square(minElement(rgD, params.N));
        maxEW = square(maxElement(rgD, params.N));
        eigensysIsUptodate = true;

      }
    }

    testMinStdDevs();

    for(int iNk = 0; iNk < params.lambda; ++iNk)
    { // generate scaled random vector D*z
      T* rgrgxink = population[iNk];
      for(int i = 0; i < params.N; ++i)
        if(diag)
          rgrgxink[i] = xmean[i] + sigma*rgD[i]*(arma::randu(1));
        else
          tempRandom[i] = rgD[i]*(arma::randu(1));
      if(!diag)
        for(int i = 0; i < params.N; ++i) // add mutation sigma*B*(D*z)
        {
          T sum = 0.0;
          for(int j = 0; j < params.N; ++j)
            sum += B[i][j]*tempRandom[j];
          rgrgxink[i] = xmean[i] + sigma*sum;
        }
    }

    if(state == UPDATED || gen == 0)
      ++gen;
    state = SAMPLED;

    return population;
  }

  /**
   * Can be called after samplePopulation() to resample single solutions of the
   * population as often as desired. Useful to implement a box constraints
   * (boundary) handling.
   * @param i Index to an element of the returned value of samplePopulation()
   * @return A pointer to the resampled "population".
   */
  T* const* reSampleSingle(int i)
  {
    T* x;
    assert(i >= 0 && i < params.lambda &&
        "reSampleSingle(): index must be between 0 and sp.lambda");
    x = population[i];
    addMutation(x);
    return population;
  }

  /**
   * Can be called after samplePopulation() to resample single solutions. In
   * general, the function can be used to sample as many independent
   * mean+sigma*Normal(0,C) distributed vectors as desired.
   *
   * Input x can be a pointer to an element of the vector returned by
   * samplePopulation() but this is inconsistent with the const qualifier of the
   * returned value and therefore rather reSampleSingle() should be used.
   * @param x Solution vector that gets sampled a new value. If x == NULL new
   *          memory is allocated and must be released by the user using
   *          delete[].
   * @return A pointer to the resampled solution vector, equals input x for
   *         x != NULL on input.
   */
  T* sampleSingleInto(T* x)
  {
    if(!x)
      x = new T[params.N];
    addMutation(x);
    return x;
  }

  /**
   * Can be called after samplePopulation() to resample single solutions. In
   * general, the function can be used to sample as many independent
   * mean+sigma*Normal(0,C) distributed vectors as desired.
   * @param x Element of the return value of samplePopulation(), that is
   *          pop[0..\f$\lambda\f$]. This solution vector of the population gets
   *          sampled a new value.
   * @return A pointer to the resampled "population" member.
   */
  T const* reSampleSingleOld(T* x)
  {
    assert(x && "reSampleSingleOld(): Missing input x");
    addMutation(x);
    return x;
  }

  /**
   * Used to reevaluate a slightly disturbed solution for an uncertaintly
   * measurement. In case if x == NULL on input, the memory of the returned x
   * must be released.
   * @param x Solution vector that gets sampled a new value. If x == NULL new
   *          memory is allocated and must be released by the user using
   *          delete[] x.
   * @param pxmean Mean vector for perturbation.
   * @param eps Scale factor for perturbation:
   * @return A pointer to the perturbed solution vector, equals input x for
   *         x != NULL.
   */
  T* perturbSolutionInto(T* x, T const* pxmean, T eps)
  {
    if(!x)
      x = new T[params.N];
    assert(pxmean && "perturbSolutionInto(): pxmean was not given");
    addMutation(x, eps);
    return x;
  }

  /**
   * Core procedure of the CMA-ES algorithm. Sets a new mean value and estimates
   * the new covariance matrix and a new step size for the normal search
   * distribution.
   * @param fitnessValues An array of lambda function values.
   * @return Mean value of the new distribution.
   */
  T* updateDistribution(const T* fitnessValues)
  {
    const int N = params.N;
    bool diag = params.diagonalCov == 1 || params.diagonalCov >= gen;

    assert(state != UPDATED && "updateDistribution(): You need to call "
          "samplePopulation() before update can take place.");
    assert(fitnessValues && "updateDistribution(): No fitness function value array input.");

    if(state == SAMPLED) // function values are delivered here
      countevals += params.lambda;
    else if(params.logWarnings)
      params.logStream <<  "updateDistribution(): unexpected state" << std::endl;

    // assign function values
    for(int i = 0; i < params.lambda; ++i)
      population[i][N] = functionValues[i] = fitnessValues[i];

    // Generate index
    sortIndex(fitnessValues, index, params.lambda);

    // Test if function values are identical, escape flat fitness
    if(fitnessValues[index[0]] == fitnessValues[index[(int) params.lambda / 2]])
    {
      sigma *= std::exp(T(0.2) + params.cs / params.damps);
      if(params.logWarnings)
      {
        params.logStream << "Warning: sigma increased due to equal function values"
            << std::endl << "   Reconsider the formulation of the objective function";
      }
    }

    // update function value history
    for(int i = (int) *(funcValueHistory - 1) - 1; i > 0; --i)
      funcValueHistory[i] = funcValueHistory[i - 1];
    funcValueHistory[0] = fitnessValues[index[0]];

    // update xbestever
    if(xBestEver[N] > population[index[0]][N] || gen == 1)
      for(int i = 0; i <= N; ++i)
      {
        xBestEver[i] = population[index[0]][i];
        xBestEver[N+1] = countevals;
      }

    const T sqrtmueffdivsigma = std::sqrt(params.mueff) / sigma;
    // calculate xmean and rgBDz~N(0,C)
    for(int i = 0; i < N; ++i)
    {
      xold[i] = xmean[i];
      xmean[i] = 0.;
      for(int iNk = 0; iNk < params.mu; ++iNk)
        xmean[i] += params.weights[iNk]*population[index[iNk]][i];
      BDz[i] = sqrtmueffdivsigma*(xmean[i]-xold[i]);
    }

    // calculate z := D^(-1)* B^(-1)* rgBDz into rgdTmp
    for(int i = 0; i < N; ++i)
    {
      T sum;
      if(diag)
        sum = BDz[i];
      else
      {
        sum = 0.;
        for(int j = 0; j < N; ++j)
          sum += B[j][i]*BDz[j];
      }
      tempRandom[i] = sum/rgD[i];
    }

    // cumulation for sigma (ps) using B*z
    const T sqrtFactor = std::sqrt(params.cs*(T(2)-params.cs));
    const T invps = T(1)-params.cs;
    for(int i = 0; i < N; ++i)
    {
      T sum;
      if(diag)
        sum = tempRandom[i];
      else
      {
        sum = T(0);
        T* Bi = B[i];
        for(int j = 0; j < N; ++j)
          sum += Bi[j]*tempRandom[j];
      }
      ps[i] = invps*ps[i] + sqrtFactor*sum;
    }

    // calculate norm(ps)^2
    T psxps(0);
    for(int i = 0; i < N; ++i)
    {
      const T& rgpsi = ps[i];
      psxps += rgpsi*rgpsi;
    }

    // cumulation for covariance matrix (pc) using B*D*z~N(0,C)
    int hsig = std::sqrt(psxps) / std::sqrt(T(1) - std::pow(T(1) - params.cs, T(2)* gen))
        / chiN < T(1.4) + T(2) / (N + 1);
    const T ccumcovinv = 1.-params.ccumcov;
    const T hsigFactor = hsig*std::sqrt(params.ccumcov*(T(2)-params.ccumcov));
    for(int i = 0; i < N; ++i)
      pc[i] = ccumcovinv*pc[i] + hsigFactor*BDz[i];

    // update of C
    adaptC2(hsig);

    // update of sigma
    sigma *= std::exp(((std::sqrt(psxps) / chiN) - T(1))* params.cs / params.damps);

    state = UPDATED;
    return xmean;
  }



  bool testForTermination()
  {
    T range, fac;
    int iAchse, iKoo;
    int diag = params.diagonalCov == 1 || params.diagonalCov >= gen;
    int N = params.N;
    std::stringstream message;
    std::string msg;

    if(stopMessage != "")
    {
      message << stopMessage << std::endl;

      msg = message.str();
      std::cout << msg <<"\n";
    }

    // function value reached
    if((gen > 1 || state > SAMPLED) && params.stStopFitness.flg &&
        functionValues[index[0]] <= params.stStopFitness.val)
    {
      message << "Fitness: function value " << functionValues[index[0]]
          << " <= stopFitness (" << params.stStopFitness.val << ")" << std::endl;
      
    
      msg = message.str();
      std::cout << msg <<"\n";

    }

    // TolFun
    range = std::max(maxElement(funcValueHistory, (int) std::min(gen, *(funcValueHistory - 1))),
        maxElement(functionValues, params.lambda)) -
        std::min(minElement(funcValueHistory, (int) std::min(gen, *(funcValueHistory - 1))),
        minElement(functionValues, params.lambda));

    if(gen > 0 && range <= params.stopTolFun)
    {
      message << "TolFun: function value differences " << range
          << " < stopTolFun=" << params.stopTolFun << std::endl;

      msg = message.str();
      std::cout << msg <<"\n";    
    }

    // TolFunHist
    if(gen > *(funcValueHistory - 1))
    {
      range = maxElement(funcValueHistory, (int) *(funcValueHistory - 1))
          - minElement(funcValueHistory, (int) *(funcValueHistory - 1));
      if(range <= params.stopTolFunHist)
        message << "TolFunHist: history of function value changes " << range
            << " stopTolFunHist=" << params.stopTolFunHist << std::endl;

      msg = message.str();
      std::cout << msg <<"\n";   
       }

    // TolX
    int cTemp = 0;
    for(int i = 0; i < N; ++i)
    {
      cTemp += (sigma*std::sqrt(C[i][i]) < params.stopTolX) ? 1 : 0;
      cTemp += (sigma*pc[i] < params.stopTolX) ? 1 : 0;
    }
    if(cTemp == 2*N)
    {
      message << "TolX: object variable changes below " << params.stopTolX << std::endl;

      msg = message.str();
      std::cout << msg <<"\n";   
       }

    // TolUpX
    for(int i = 0; i < N; ++i)
    {
      if(sigma*std::sqrt(C[i][i]) > params.stopTolUpXFactor*params.rgInitialStds[i])
      {
        message << "TolUpX: standard deviation increased by more than "
            << params.stopTolUpXFactor << ", larger initial standard deviation recommended."
            << std::endl;

      msg = message.str();
      std::cout << msg <<"\n";        
      break;
      }
    }

    // Condition of C greater than dMaxSignifKond
    if(maxEW >= minEW* dMaxSignifKond)
    {
      message << "ConditionNumber: maximal condition number " << dMaxSignifKond
          << " reached. maxEW=" << maxEW <<  ",minEW=" << minEW << ",maxdiagC="
          << maxdiagC << ",mindiagC=" << mindiagC << std::endl;

      msg = message.str();
      std::cout << msg <<"\n";   
       }

    // Principal axis i has no effect on xmean, ie. x == x + 0.1* sigma* rgD[i]* B[i]
    if(!diag)
    {
      for(iAchse = 0; iAchse < N; ++iAchse)
      {
        fac = 0.1* sigma* rgD[iAchse];
        for(iKoo = 0; iKoo < N; ++iKoo)
        {
          if(xmean[iKoo] != xmean[iKoo] + fac* B[iKoo][iAchse])
            break;
        }
        if(iKoo == N)
        {
          message << "NoEffectAxis: standard deviation 0.1*" << (fac / 0.1)
              << " in principal axis " << iAchse << " without effect" << std::endl;

      msg = message.str();
      std::cout << msg <<"\n";          
      break;

        }
      }
    }
    // Component of xmean is not changed anymore
    for(iKoo = 0; iKoo < N; ++iKoo)
    {
      if(xmean[iKoo] == xmean[iKoo] + sigma*std::sqrt(C[iKoo][iKoo])/T(5))
      {
        message << "NoEffectCoordinate: standard deviation 0.2*"
            << (sigma*std::sqrt(C[iKoo][iKoo])) << " in coordinate " << iKoo
            << " without effect" << std::endl;

      msg = message.str();
      std::cout << msg <<"\n";        
      break;
      }
    }

    if(countevals >= params.stopMaxFunEvals)
    {
      message << "MaxFunEvals: conducted function evaluations " << countevals
          << " >= " << params.stopMaxFunEvals << std::endl;

      msg = message.str();
      std::cout << msg <<"\n";

    }

    if(gen >= params.stopMaxIter)
    {
      message << "MaxIter: number of iterations " << gen << " >= "
          << params.stopMaxIter << std::endl;

      msg = message.str();
      std::cout << msg <<"\n";    
    }

    stopMessage = message.str();
    return stopMessage != "";
  }

  /**
   * A message that contains a detailed description of the matched stop
   * criteria.
   */
  std::string getStopMessage()
  {
    return stopMessage;
  }

  void updateEigensystem(bool force)
  {

    if(!force)
    {
      if(eigensysIsUptodate)
        return;
      // return on modulo generation number
      if(gen < genOfEigensysUpdate + params.updateCmode.modulo)
        return;

    }

    eigen(rgD, B);

    // find largest and smallest eigenvalue, they are supposed to be sorted anyway
    minEW = minElement(rgD, params.N);
    maxEW = maxElement(rgD, params.N);

    for(int i = 0; i < params.N; ++i)
      rgD[i] = std::sqrt(rgD[i]);

    eigensysIsUptodate = true;
    genOfEigensysUpdate = gen;
  }

  /**
   * Distribution mean could be changed before samplePopulation(). This might
   * lead to unexpected behaviour if done repeatedly.
   * @param newxmean new mean, if it is NULL, it will be set to the current mean
   * @return new mean
   */
  T const* setMean(const T* newxmean)
  {
    assert(state != SAMPLED && "setMean: mean cannot be set inbetween the calls"
        "of samplePopulation and updateDistribution");

    if(newxmean && newxmean != xmean)
      for(int i = 0; i < params.N; ++i)
        xmean[i] = newxmean[i];
    else
      newxmean = xmean;

    return newxmean;
  }
};


}  // namespace neuro_cmaes
}  // namespace mlpack

#endif  // MLPACK_METHODS_NEURO_CMAES_NEURO_CMAES_HPP
