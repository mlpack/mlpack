/**
 * @file lshmodel.hpp
 * @author Yannis Mentekidis
 *
 * Defines the LSHModel class, which models the Locality Sensitive Hashing
 * algorithm. The model identifies parameter sets that produce satisfactory
 * results while keeping execution time low.
 *
 * The model was proposed by Dong et al in the following paper.
 *
 * @code
 * @article{Dong2008LSHModel,
 *  author = {Dong, Wei and Wang, Zhe and Josephson, William and Charikar,
 *      Moses and Li, Kai},
 *  title = {{Modeling LSH for performance tuning}},
 *  journal = {Proceeding of the 17th ACM conference on Information and
 *      knowledge mining - CIKM '08},
 *  pages = {669},
 *  url = {http://portal.acm.org/citation.cfm?doid=1458082.1458172},
 *  year = {2008}
 * }
 * @endcode
 *
 * We use a different method to fit Gamma Distributions to pairwise distances.
 * Instead of the MLE method proposed in the paper above, we use the mlpack
 * class GammaDistribution, which implements fitting according to Thomas Minka's
 * work.
 *
 * @code
 * @techreport{minka2002estimating,
 *   title={Estimating a {G}amma distribution},
 *   author={Minka, Thomas P.},
 *   institution={Microsoft Research},
 *   address={Cambridge, U.K.},
 *   year={2002}
 * }
 * @endcode
 */

#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_MODEL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_MODEL_HPP

// For returning LSHSearch objects.
#include <mlpack/methods/lsh/lsh_search.hpp>
// For template parameters and kNN search (if nescessary).
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/neighbor_search/sort_policies/nearest_neighbor_sort.hpp>
// For curve fitting.
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>
// Default objective function.
#include "objectivefunction.hpp"

namespace mlpack {
namespace neighbor {

template <
    typename SortPolicy = NearestNeighborSort,
    typename ObjectiveFunction = DefaultObjectiveFunction
      >
class LSHModel
{
 public:

   //! Empty Constructor. Do nothing
   LSHModel(){ /* Do nothing. */ };

   /** Parameterized Constructor. This function initializes the object and
    * trains it with the provided reference set.
    *
    * @param referenceSet The data that will be used as a reference set for LSH
    *     to run queries against. We will fit distributions based on this data
    *     and produce good parameters for it.
    * @param minRecall The minimum recall we want to guarantee. The parameters
    *     we will estimate will try to keep average recall of LSH above this.
    *     Must be in [0, 1).
    * @param sampleSize The percentage of the reference set to sample for the
    *     estimation. Naive all-kNN will be run on this sample, so if it is too
    *     big, training will be very slow. Must be in [0, 1)
    * @param k The number of nearest neighbors wanted for each query.
    */
   LSHModel(
       const arma::mat &referenceSet,
       const double minRecall,
       const double sampleSize,
       const size_t k);

   //! Destructor. If we own any memory, free it.
   ~LSHModel();

   /**
    * Trains the LSHModel. Fits distributions using referenceSet and then looks
    * for LSH parameters that would return recalls larger than minRecall in the
    * lowest cost (selectivity) possible.
    *
    * The model can estimate good values for the parameters:
    *   * numProj: Number of projections per projection table.
    *   * numTables: Number of projection tables.
    *   * hashWidth: Hash width of the LSH hash.
    *   * numProbes: Number of probes for multiprobe LSH.
    *
    * Train stores the computed parameters in the LSHModel object's variables.
    *
    * @param referenceSet The data that will be used as a reference set for LSH
    *     to run queries against. We will fit distributions based on this data
    *     and produce good parameters for it.
    * @param minRecall The minimum recall we want to guarantee. The parameters
    *     we will estimate will try to keep average recall of LSH above this.
    *     Must be in [0, 1).
    * @param sampleSize The percentage of the reference set to sample for the
    *     estimation. Naive all-kNN will be run on this sample, so if it is too
    *     big, training will be very slow. Must be in [0, 1)
    * @param k The number of nearest neighbors wanted for each query.
    */
   void Train(
       const arma::mat &referenceSet,
       const double minRecall,
       const double sampleSize,
       const size_t k);

   /**
    * This function returns an LSHSearch object trained with the parameters
    * calculated when the LSHModel was trained.
    * If any of the parameters we trained for (numProj, numTables, hashWidth)
    * are specified, we will not used the trained but the provided parameters.
    * If these are left to default (0), the estimated parameters will be used.
    *
    * @param numProjIn The number of projections per table.
    * @param numTablesIn The number of projection tables.
    * @param hashWidthIn The first level hash width.
    * @param secondHashSize The second level hash width.
    * @param bucketSize The second level bucket size.
    */
   LSHSearch<SortPolicy>* LSHObject(
       const size_t numProjIn = 0,
       const size_t numTablesIn = 0,
       const double hashWidthIn = 0.0,
       const size_t secondHashSize = 99901,
       const size_t bucketSize = 500);

   //! Return the number of projections calculated.
   size_t NumProj(void) const { return numProj; };

   //! Return the number of tables calculated.
   size_t NumTables(void) const { return numTables; };

   //! Return the calculated hash width.
   double HashWidth(void) const { return hashWidth; };

   //! Return the calculated number of probes.
   double NumProbes(void) const { return numProbes; };

   //! Return the reference set.
   const arma::mat ReferenceSet(void) const {return *referenceSet; };

   //! Serialize the LSHModel object.
   template<typename Archive>
   void Serialize(Archive& ar);

 private:

   /**
    * This is a helper class that uses the function a * k^b * N^c for some
    * parameters a, b, c that have been fit to either predict the arithmetic or
    * geometric mean of the squared distance of a point to its k-nearest
    * neighbor, given some dataset size N and its k-nearest neighbor.
    */
   class DistanceStatisticPredictor
   {
    public:
      //! Empty constructor.
      DistanceStatisticPredictor() { };

      /** 
       * Function to construct with training set.
       *
       * @param inputSize A vector of input sizes. The first input variable of 
       *     the regression.
       * @param kValues A vector of k values. The second input variable of the
       *     regression.
       * @param statistic A vector of responses - the value of the statistic for
       *     each given inputSize.
       */
      DistanceStatisticPredictor(const arma::Col<size_t>& inputSize, 
                                 const arma::Col<size_t>& kValues,
                                 const arma::mat& statistic) 
      { Train(inputSize, kValues, statistic); };
      
      //! Default destructor.
      ~DistanceStatisticPredictor() { };

      /**
       * Function that fits the alpha, beta and gamma parameters.
       *
       * @param inputSize A vector of input sizes. The first input variable of 
       *     the regression.
       * @param kValues A vector of k values. The second input variable of the
       *     regression.
       * @param statistic A vector of responses - the value of the statistic for
       *     each given inputSize.
       */
      double Train(const arma::Col<size_t>& inputSize, 
                 const arma::Col<size_t>& kValues,
                 const arma::mat& statistic);

      /** 
       * Evaluate the statistic for a given dataset size.
       *
       * @param N - a new input size for which to evaluate the expected
       *     statistic.
       */
      double Predict(size_t N, size_t k) 
      { return alpha * std::pow(k, beta) * std::pow(N, gamma); };

      //! Set the alpha parameter.
      void Alpha(double a) { alpha = a; };

      //! Get the alpha parameter.
      double Alpha(void) { return alpha; };
      
      //! Set the beta parameter.
      void Beta(double b) { beta = b; };

      //! Get the beta parameter.
      double Beta(void) { return beta; };

      //! Set the gamma parameter.
      void Gamma(double c) { gamma = c; };

      //! Get the gamma parameter.
      double Gamma(void) { return gamma; };


    private:
      double alpha;
      double beta;
      double gamma;
   };

   //! DistanceStatisticPredictor for arithmetic mean.
   DistanceStatisticPredictor aMeanPredictor;

   //! DistanceStatisticPredictor for geometric mean.
   DistanceStatisticPredictor gMeanPredictor;
   
   /**
    * Function that fits two DistanceStatisticPredictors - one
    * to predict arithmetic mean and one to preduct geometric mean.
    *
    * @param referenceSizes The number of reference points for each kNN search.
    * @param kValues The rank of the neighbors used for the statistic, for
    *     example k = 5 means Ek is the arithmetic mean of the 5th-nearest
    *     neighbor for different sample sizes.
    * @param Ek The arithmetic mean of the squared distances of a point and its
    *      k-nearest neighbor. One column per k.
    * @param Gk The geometric mean of the squared distances of a point and its
    *      k-nearest neighbor. One column per k.
    */
   void ApproximateKNNStatistics(const arma::Col<size_t>& referenceSizes, 
                                 const arma::Col<size_t>& kValues,
                                 const arma::mat& Ek, 
                                 const arma::mat& Gk);

   //! Flag that tracks if we own the reference set.
   bool ownsSet;

   //! Flag that tracks if we own an LSHSearch object.
   bool ownsLSHObject;

   //! Number of projections per table.
   size_t numProj;

   //! Number of projection tables.
   size_t numTables;

   //! First-level hash width.
   double hashWidth;

   //! Number of probes for multiprobe LSH.
   size_t numProbes;

   //! Reference dataset.
   const arma::mat* referenceSet;

   //! Vector of LSHSearch objects.
   std::vector< LSHSearch<SortPolicy> > lshObjectVector;

   //! Statistic: average squared distance of points.
   double meanDist;

   //! Statistic: logarithm of squared distance of points.
   double logMeanDist;

   //! Statisitc: average of logarithm of squared distances of points.
   double meanLogDist;


}; // class LSHModel.

} // namespace neighbor.
} // namespace mlpack.

// Include the class implementation.
#include "lshmodel_impl.hpp"

#endif
