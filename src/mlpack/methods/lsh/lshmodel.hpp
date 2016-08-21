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
// Gamma distribution for modeling squared distances.
#include <mlpack/core/dists/gamma_distribution.hpp>
// For fitting distance statistic regressors.
#include "distance_statistic_predictor.hpp"

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
    * This function uses the trained model to predict recall / selectivity
    * values for a given parameter set.
    *
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
                const size_t numTables,
                const size_t numProj,
                const size_t numProbes,
                const double hashWidth,
                double& predictedRecall,
                double& predictedSelect);

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
    * specific points, to generate scores. The template sequence is also
    * independent of the hashWidth, and depends only on numProj and numProbes.
    *
    * See mlpack/methods/lsh/lsh_search_impl.hpp for more details about how
    * perturbation sequences are generated from specific points.
    *
    * @param numProj The number of projections for the LSH scheme for which we
    *     want to compute the template perturbation sequence.
    * @param numProbes The number of probes to generate.
    */
   void GenerateTemplateSequence(size_t numProj,
                                 size_t numProbes);

   /**
    * This function evaluates the probability that two points that are at
    * distance chi from each other will be neighbors when we use LSH with a
    * specific number of projections, probing bins, and tables for a given hash
    * width.
    *
    * @param chi The distance of two points.
    * @param hashWidth The first-level hash width.
    * @param numTables The number of random projection tables used by LSH.
    * @param numProj The number of projections per hash table (dimensionality of
    *     new space).
    * @param numProbes The number of additional probing bins of Multiprobe LSH.
    */
   //TODO: inline?
   double Rho(double chi,
              double hashWidth,
              size_t numTables,
              size_t numProj,
              size_t numProbes) const;
   /**
    * This is a helper function that is called by Rho() and returns the inner
    * value of the product used in the calculation of the probability that Rho
    * calculates.
    *
    * @param chi The distance of two points.
    * @param hashWidth The first-level hash width.
    * @param delta The perturbation to evaluate for.
    * @param proj The projection we evaluate for ( 0 <= proj < numProj).
    * @param numProj The total number of projections.
    */
   inline double SameBucketProbability(double chi,
                                       double hashWidth,
                                       short delta,
                                       size_t proj,
                                       size_t numProj) const;

   /**
    * This function calculates the recall of LSH for a given set of parameters.
    * It uses the function
    *
    * r = \frac{1}{K} \sum_{1}^{K} \int_{0}^{\infty}(Rho(\sqrt{x}) * f_k(x)) dx
    *
    * as proposed in the paper.
    *
    */
   double Recall(size_t maxK,
                 size_t numTables,
                 size_t numProj,
                 size_t numProbes,
                 double hashWidth);

   /**
    * This function calculates the selectivity of LSH for a given set of parameters.
    * It uses the function
    *
    * s = \int_{0}^{\infty}(Rho(\sqrt{x}) * f(x)) dx
    *
    * as proposed in the paper.
    *
    */
   double Selectivity(size_t numTables,
                      size_t numProj,
                      size_t numProbes,
                      double hashWidth);

   /**
    * Helper class for boost::integration.
    */
   class IntegralObjective
   {
    public:
     // Initialize everything.
     IntegralObjective(const size_t k, 
                       const size_t numTables,
                       const size_t numProj,
                       const size_t numProbes,
                       const double hashWidth,
                       const mlpack::distribution::GammaDistribution* gamma,
                       const LSHModel* model)
     : k(k), numTables(numTables), numProj(numProj), 
     numProbes(numProbes), hashWidth(hashWidth), gamma(gamma), model(model)
     { /* do nothing */};

     ~IntegralObjective() { };

     // Use as function with the operator () and one argument.
     double operator()(const double& chi) const
     {
       return 
         (model->Rho(std::sqrt(chi), hashWidth, numTables, numProj, numProbes)) 
         * (gamma->Probability(chi, k));
     }

    private:  
     const size_t k;
     const size_t numTables;
     const size_t numProj;
     const size_t numProbes;
     const double hashWidth;

     const mlpack::distribution::GammaDistribution* gamma;
     const LSHModel* model;

   };

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
    * Matrix that stores, in each column, the "direction" of the perturbation:
    * 0 means no perturbation on that dimension, -1 means reduce dimension value
    * by 1, and +1 means increase dimension value by 1.
    */
   arma::Mat<short int> templateSequence;

   //! DistanceStatisticPredictor for arithmetic mean.
   DistanceStatisticPredictor<ObjectiveFunction> aMeanPredictor;

   //! DistanceStatisticPredictor for geometric mean.
   DistanceStatisticPredictor<ObjectiveFunction> gMeanPredictor;

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

   //! LSHSearch Object
   LSHSearch<SortPolicy> trainedLSHObject;

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
