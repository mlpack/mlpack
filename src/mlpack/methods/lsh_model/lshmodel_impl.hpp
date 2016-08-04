/**
 * @file lshmodel_impl.hpp
 * @author Yannis Mentekidis
 *
 * Implementation of the LSHModel functions.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_MODEL_IMPL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_MODEL_IMPL_HPP

#include "lshmodel.hpp"


//TODO: remove
using std::cout;
using std::flush;
using std::endl;

namespace mlpack {
namespace neighbor {

// Constructor sets variables and trains the object.
template <typename SortPolicy, typename ObjectiveFunction>
LSHModel<SortPolicy, ObjectiveFunction>::LSHModel(const arma::mat &referenceSet,
                   const double minRecall,
                   const double sampleSize,
                   const size_t k)
{
  // We don't own the set - we just point to it.
  ownsSet = false;
  this->referenceSet = &referenceSet;

  Train(referenceSet, minRecall, sampleSize, k);
}

// Destructor must de-allocate any referenceSet and LSHSearch objects we own.
template <typename SortPolicy, typename ObjectiveFunction>
LSHModel<SortPolicy, ObjectiveFunction>::~LSHModel()
{
  if (ownsSet)
    delete referenceSet;
};

// Trains the object.
template <typename SortPolicy, typename ObjectiveFunction>
void LSHModel<SortPolicy, ObjectiveFunction>::Train(const arma::mat &referenceSet,
                     const double minRecall,
                     const double sampleSize,
                     const size_t k)
{
  // TODO: Implement

  // Sanity Check: Verify that recall and sampleSize are in [0, 1).
  if (minRecall >= 1 || minRecall < 0)
    throw std::runtime_error("Recall must be floating point number in [0, 1)");

  if (sampleSize > 1 || sampleSize <= 0)
    throw std::runtime_error(
        "Sampling rate must be floating point number in (0, 1]");

  const size_t numPoints = referenceSet.n_cols; // Points in original set.

  // Step 1. Select a random sample of the dataset. We will work with only that
  // sample.
  arma::vec sampleHelper(referenceSet.n_cols, arma::fill::randu);

  // Keep a sample of the dataset. Shuffle to be impartial (in case reference
  // set is sorted).
  arma::mat sampleSet = arma::shuffle(referenceSet.cols(
        // We have uniformly random numbers in [0, 1], so we expect about
        // N*sampleSize of them to be in [0, sampleSize).
        arma::find(sampleHelper < sampleSize)
        ));
  const size_t numSamples = sampleSet.n_cols; // Points in sampled set.

  Log::Info << "Sampled " << numSamples << " points to train with." << std::endl;

  // Step 2. Compute all-vs-all distances of points in the sample.
  // The distance matrix is symmetric, so we only compute elements above the
  // diagonal. There are (N * (N - 1)) / 2 such elements.
  Timer::Start("pairwise_distances");
  arma::vec distances(numSamples * (numSamples - 1) / 2);
  size_t d = 0; // Index of where to store next.
  for (size_t i = 0; i < numSamples; ++i)
    for (size_t j = i + 1; j < numSamples; ++j)
      distances(d++) = metric::EuclideanDistance::Evaluate(
          sampleSet.unsafe_col(i), sampleSet.unsafe_col(j));
  Log::Info << "Computed " << d << " pointwise distances." << std::endl;
  Timer::Stop("pairwise_distances");

  // Step 3. Estimate statistics of these distances: log(mean(d)), mean(log(d)),
  // mean(d).
  distances = arma::pow(distances, 2);
  meanDist = arma::mean(distances);
  logMeanDist = std::log(meanDist);
  meanLogDist = arma::mean(arma::log(distances));

  // Step 4. Select a small part of the sample as 'anchor points'. Use the rest
  // of the sample as the reference set. Find the k-Nearest Neighbors' distances
  // from the anchor points for increasing portion of the reference set. Compute
  // the arithmetic and geometric mean of distances from each anchor to its
  // k-Nearest Neighbor.
  // The geometric mean of N numbers is the Nth root of the product of the
  // numbers. Through logarithmic properties though, this becomes computable
  // through exponentiating the mean of the logarithms of x:
  // mean(log(x)) = geometricmean(x).

  // Number of samples to create for modeling the Gamma Distributions
  size_t regressionExamples = 50; // TODO: parameter?

  // Number of points to use as queries.
  size_t numAnchors = (size_t) std::round(0.1 * numSamples);
  arma::mat queryMat = sampleSet.cols(0, numAnchors - 1);
  // Evenly spaced sample sizes.
  arma::Col<size_t> referenceSizes = arma::conv_to< arma::Col<size_t> >::from(
    arma::linspace(numAnchors, numSamples - numAnchors - 1,
      regressionExamples));

  // Statistics - Arithmetic and geometric means for growing reference set.
  // Compute one of each for each k.
  arma::mat Ek(regressionExamples, k);
  arma::mat Gk(regressionExamples, k);

  Timer::Start("neighbors_distances");
  // For each referenceSize, calculate the kNN of the anchors
  for (size_t i = 0; i < regressionExamples; ++i)
  {
    // TODO: Since we've already computed this, avoid calling kNN?
    // Reference set for kNN
    arma::mat refMat = sampleSet.cols(numAnchors, numAnchors + referenceSizes(i) );

    arma::Mat<size_t> neighbors; // Not going to be used but required.
    arma::mat kNNDistances; // What we need.
    KNN naive(refMat, true); // true: train and use naive kNN.
    naive.Search(queryMat, k, neighbors, kNNDistances);
    kNNDistances = arma::pow(kNNDistances, 2);

    // Compute Arithmetic and Geometric mean of the distances.
    Ek.row(i) = arma::mean(kNNDistances.t());
    Gk.row(i) = arma::exp(arma::mean(arma::log(kNNDistances.t()), 0));
  }
  Timer::Stop("neighbors_distances");

  // Step 5. Model the arithmetic and geometric mean according to the paper.
  // This will produce 6 parameters (aE, bE, cE, aG, bG, cG).
  // Vector of k values.
  arma::Col<size_t> kValues = arma::linspace<arma::Col<size_t>>(1, k, k);
  ApproximateKNNStatistics(referenceSizes, kValues, Ek, Gk);

  // Step 6. Fit Gamma distributions to pairwise distances and kNN distances,
  // generated or estimated in steps 3 and 5.

  // Step 7. Run Binary search on parameter space to minimize selectivity while
  // keeping recall above minimum.
}

// Fit two predictors, one for arithmetic mean E and one for geometric mean G.
template <typename SortPolicy, typename ObjectiveFunction>
void LSHModel<SortPolicy, ObjectiveFunction>::
ApproximateKNNStatistics(const arma::Col<size_t>& referenceSizes,
                         const arma::Col<size_t>& kValues,
                         const arma::mat& Ek,
                         const arma::mat& Gk)
{
  double aError = aMeanPredictor.Train(referenceSizes, kValues, Ek);
  Log::Info << "L_BFGS Converged for arithmetic mean with error "
    << aError << "." << std::endl;
  double gError = gMeanPredictor.Train(referenceSizes, kValues, Gk);
  Log::Info << "L_BFGS Converged for geometric mean with error "
    << gError << "." << std::endl;
}

// Construct and return an LSH object.
template <typename SortPolicy, typename ObjectiveFunction>
LSHSearch<SortPolicy> LSHModel<SortPolicy, ObjectiveFunction>::
LSHObject(const size_t numProjIn,
          const size_t numTablesIn,
          const double hashWidthIn,
          const size_t secondHashSize,
          const size_t bucketSize)
{
  // Values for the object to be created with (specified by user or default).
  size_t numProjOut = numProjIn;
  size_t numTablesOut = numTablesIn;
  double hashWidthOut = hashWidthIn;

  // If not specified by user, set these to the ones we trained for.
  if (numProjIn == 0)
    numProjOut = this->numProj;

  if (numTablesIn == 0)
    numTablesOut = this->numTables;

  if (hashWidthOut == 0.0)
    hashWidthOut = this->hashWidth;

  LSHSearch<> lsh(*referenceSet, numProjOut, numTablesOut, hashWidthOut,
      secondHashSize, bucketSize);

  lshObjectVector.push_back(lsh);

  return lshObjectVector[lshObjectVector.size() - 1];
}

// Fit a curve to the data provided.
template<typename SortPolicy, typename ObjectiveFunction>
double LSHModel<SortPolicy, ObjectiveFunction>::DistanceStatisticPredictor::Train(
    const arma::Col<size_t>& inputSize,
    const arma::Col<size_t>& kValues,
    const arma::mat& statistic)
{
  // Objective function for fitting the E(x, k) curve to the statistic.
  ObjectiveFunction f(inputSize, kValues, statistic);

  // Optimizer. Use L_BFGS (TODO: Make this a template parameter?)
  mlpack::optimization::L_BFGS<ObjectiveFunction> opt(f);

  // Get an initial point from the optimizer.
  arma::mat currentPoint = f.GetInitialPoint();
  double result = opt.Optimize(currentPoint);

  // Optimizer is done - set alpha, beta, gamma.
  this->alpha = currentPoint(0, 0);
  this->beta = currentPoint(1, 0);
  this->gamma = currentPoint(2, 0);

  return result;
}

// Serialize the object and save to a file.
template <typename SortPolicy, typename ObjectiveFunction>
template<typename Archive>
void LSHModel<SortPolicy, ObjectiveFunction>::Serialize(Archive& ar)
{
  //TODO: implement this.
}
} // Namespace neighbor.
} // Namespace mlpack.

#endif
