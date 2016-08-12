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
// Gamma distribution for modeling squared distances.
#include <mlpack/core/dists/gamma_distribution.hpp>

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
   LSHModel(){ referenceSet = NULL; };

   /** Parameterized Constructor. This function initializes the object and
    * trains it with the provided reference set.
    *
    * @param referenceSet The data that will be used as a reference set for LSH
    *     to run queries against. We will fit distributions based on this data
    *     and produce good parameters for it.
    * @param sampleSize The percentage of the reference set to sample for the
    *     estimation. Naive all-kNN will be run on this sample, so if it is too
    *     big, training will be very slow. Must be in [0, 1)
    * @param k The number of nearest neighbors wanted for each query.
    */
   LSHModel(
       const arma::mat &referenceSet,
       const double sampleSize,
       const size_t k);

   //! Destructor. If we own any memory, free it.
   ~LSHModel();

   /**
    * Trains the LSHModel. Train() uses a sample that is sampleRate * |N| to
    * estimate parameters of the dataset. The estimated parameters are:
    *   * Arithmetic mean of pairwise distances of random points in the sample.
    *   * Geometric mean for the pairwise distnaces
    *   * Arithmetic mean of distance random point to its k-th nearest neighbor 
    *       as a function of |N|, the number of points.
    *   * Geometric mean of the same distance.
    *
    * Train() does not find LSH Parameters - it only estimates the dataset
    * parameters. You have to call Predict() to find LSH Parameters.
    *
    * @param referenceSet The data that will be used as a reference set for LSH
    *     to run queries against. We will fit distributions based on this data
    *     and produce good parameters for it.
    * @param sampleRate The percentage of the reference set to sample for the
    *     estimation. Naive all-kNN will be run on this sample, so if it is too
    *     big, training will be very slow. Must be in [0, 1)
    * @param maxKValue The maximum number of nearest neighbors for each query to
    *     train for.
    */
   void Train(const arma::mat& referenceSet, 
              const double sampleRate = 0.1,
              const size_t maxKValue = 32);

   /**
    * Predict() finds LSH parameters that should work well for the dataset the 
    * LSHModel was trained for. 
    * Warning: If the k specified is larger than the maxKValue passed to
    * Train(), Train() will be called again. This might have adverse effects to
    * performance.
    *
    * @param datasetSize The size of the dataset that will be used.
    * @param k The number of k-nearest neighbors LSH must find.
    * @param minRecall The minimum acceptable recall we want to tune for.
    */
   void Predict(const size_t datasetSize, 
                const size_t k, 
                const double minRecall);

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
   LSHSearch<SortPolicy> LSHObject(
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
   * Returns the score of a perturbation vector generated by perturbation set A.
   * The score of a pertubation set (vector) is the sum of scores of the
   * participating actions.
   * @param A perturbation set to compute the score of.
   * @param scores vector containing score of each perturbation.
  */
  double PerturbationScore(const std::vector<bool>& A,
                                  const arma::vec& scores) const;
  /**
   * Inline function used by GetAdditionalProbingBins. The vector shift operation
   * replaces the largest element of a vector A with (largest element) + 1.
   * Returns true if resulting vector is valid, otherwise false.
   * @param A perturbation set to shift.
  */
  bool PerturbationShift(std::vector<bool>& A) const;

  /**
   * Inline function used by GetAdditionalProbingBins. The vector expansion
   * operation adds the element [1 + (largest_element)] to a vector A, where
   * largest_element is the largest element of A. Returns true if resulting vector
   * is valid, otherwise false.
   * @param A perturbation set to expand.
  */
  bool PerturbationExpand(std::vector<bool>& A) const;

  /**
   * Return true if perturbation set A is valid. A perturbation set is invalid if
   * it contains two (or more) actions for the same dimension or dimensions that
   * are larger than the queryCode's dimensions.
   * @param A perturbation set to validate.
   * @param numProj The number of projections for the sequence under validation.
  */
  bool PerturbationValid(const std::vector<bool>& A, size_t numProj) const;
   /**
    * Function that creates a template perturbation sequence given a value for
    * an M and a W. The template perturbation sequence is based on the
    * statistical properties of multi-probe LSH and uses those, instead of
    * specific points, to generate scores.
    * See mlpack/methods/lsh/lsh_search_impl.hpp for more details about how
    * perturbation sequences are generated from specific points.
    *
    * @param numProj The number of projections for the LSH scheme for which we
    *     want to compute the template perturbation sequence.
    * @param hashWidth The hash width for the LSH scheme.
    * @param numProbes The number of probes to generate.
    */
   void GenerateTemplateSequence(size_t numProj, 
                                 double hashWidth, 
                                 size_t numProbes);

   /** Matrix that stores, in each column, the "direction" of the perturbation:
    * 0 means no perturbation on that dimension, -1 means reduce dimension value
    * by 1, and +1 means increase dimension value by 1.
    */
   
   arma::Mat<short int> templateSequence;

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

   //! (k+1)-dimensional gamma distribution for predicting squared distances.
   mlpack::distribution::GammaDistribution distancesDistribution;

   //! Flag that tracks if we own the reference set.
   bool ownsSet;

   //! Maximum k value the object is trained for.
   size_t maxKValue;

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

   //! LSHSearch Object Vector.
   std::vector<LSHSearch<SortPolicy>> lshObjectVector;

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
