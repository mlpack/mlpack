/**
 * @file lshmodel_impl.hpp
 * @author Yannis Mentekidis
 *
 * Implementation of the LSHModel functions.
 */
#ifndef MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_MODEL_IMPL_HPP
#define MLPACK_METHODS_NEIGHBOR_SEARCH_LSH_MODEL_IMPL_HPP

#include "lshmodel.hpp"
#include <boost/math/distributions/normal.hpp> // pdf and cdf needed


//TODO: remove
using std::cout;
using std::flush;
using std::endl;

namespace mlpack {
namespace neighbor {

// Constructor sets variables and trains the object.
template <typename SortPolicy, typename ObjectiveFunction>
LSHModel<SortPolicy, ObjectiveFunction>::
LSHModel(const arma::mat &referenceSet,
         const double sampleSize,
         const size_t k)
{
  // We don't own the set - we just point to it.
  ownsSet = false;
  this->referenceSet = &referenceSet;

  Train(referenceSet, sampleSize, k);
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
void LSHModel<SortPolicy, ObjectiveFunction>::Train(
    const arma::mat &referenceSet,
    const double sampleRate,
    const size_t k)
{
  // Sanity check - sample rate must be in (0, 1].
  if (sampleRate > 1 || sampleRate <= 0)
    Log::Fatal << "Sampling rate must be floating point number in (0, 1]"
        << std::endl;

  // Update the object's max K value information.
  maxKValue = k;

  // Save pointer to training set.
  this->referenceSet = &referenceSet;

  // Step 1. Select a random sample of the dataset. We will work with only that
  // sample.
  arma::vec sampleHelper(referenceSet.n_cols, arma::fill::randu);

  // Keep a sample of the dataset: We have uniformly random numbers in [0, 1],
  // so we expect about N*sampleRate of them to be in [0, sampleRate).
  arma::mat sampleSet = referenceSet.cols(
        arma::find(sampleHelper < sampleRate));
  // Shuffle to be impartial (in case dataset is sorted in some way).
  sampleSet = arma::shuffle(sampleSet);
  const size_t numSamples = sampleSet.n_cols; // Points in sampled set.

  Log::Info << "Training model with " << numSamples << " points in sample set."
    << std::endl;

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
  this->meanDist = arma::mean(distances);
  this->logMeanDist = std::log(meanDist);
  this->meanLogDist = arma::mean(arma::log(distances));

  // Step 4. Select a small part of the sample as 'anchor points'. Use the rest
  // of the sample as the reference set. Find the k-Nearest Neighbors' distances
  // from the anchor points for increasing portion of the reference set. Compute
  // the arithmetic and geometric mean of distances from each anchor to its
  // k-Nearest Neighbor.
  // The geometric mean of N numbers is the Nth root of the product of the
  // numbers. Through logarithmic properties though, this becomes computable
  // through exponentiating the mean of the logarithms of x:
  // exp(mean(log(x))) = geometricmean(x).

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

  // For each referenceSize, calculate the kNN of the anchors
  Log::Info.ignoreInput = true; // Ignore kNN output.
  for (size_t i = 0; i < regressionExamples; ++i)
  {
    // TODO: Since we've already computed this, avoid calling kNN?

    // Reference set for kNN
    arma::mat refMat = sampleSet.cols(numAnchors,
        numAnchors + referenceSizes(i));

    arma::Mat<size_t> neighbors; // Not going to be used but required.
    arma::mat kNNDistances; // What we need.
    KNN naive(refMat, true); // true: train and use naive kNN.
    naive.Search(queryMat, k, neighbors, kNNDistances);
    kNNDistances = arma::pow(kNNDistances, 2);

    // Compute Arithmetic and Geometric mean of the distances.
    Ek.row(i) = arma::mean(kNNDistances.t());
    Gk.row(i) = arma::exp(arma::mean(arma::log(kNNDistances.t()), 0));
  }
  Log::Info.ignoreInput = false; // Keep giving normal output.

  // Step 5. Model the arithmetic and geometric mean according to the paper.
  // This will produce 6 parameters (aE, bE, cE, aG, bG, cG).
  // Vector of k values.
  Timer::Start("neighbor_statistic_regression");
  arma::Col<size_t> kValues = arma::linspace<arma::Col<size_t>>(1, k, k);
  ApproximateKNNStatistics(referenceSizes, kValues, Ek, Gk);
  Timer::Stop("neighbor_statistic_regression");
}


// Predict recall / selectivity for the given parameters.
template <typename SortPolicy, typename ObjectiveFunction>
void LSHModel<SortPolicy, ObjectiveFunction>::Predict(const size_t datasetSize,
                                                      const size_t k,
                                                      const size_t numTables,
                                                      const size_t numProj,
                                                      const size_t numProbes,
                                                      const double hashWidth,
                                                      double& predictedRecall,
                                                      double& predictedSelect)
{
  // If the object wasn't trained, die here.
  if (referenceSet == NULL)
    Log::Fatal << "Attempt to use Predict() on untrained Object. Exiting."
        << std::endl;

  // Before proceeding, if requested k is larger than the k we trained with,
  // re-train the object.
  if (k > maxKValue)
  {

    // Otherwise, warn the user of the re-training and re-train.
    Log::Warn << "Larger k requested; Re-training the LSHModel "
      "with default sampling rate and new k." << std::endl;
    Train(*referenceSet, 0.1, k); // Default sampling rate.
  }

  // Note: Steps 1 - 5 happen in Train().

  // Step 6. Fit Gamma distributions to pairwise distances and kNN distances,
  // generated or estimated in steps 3 and 5.
  // Gamma distribution for pairwise distances.
  arma::vec logMeanVec(k + 1), meanLogVec(k + 1), meanVec(k + 1);
  // Statistics were computed in Train()
  meanVec(0) = this->meanDist;
  logMeanVec(0) = this->logMeanDist;
  meanLogVec(0) = this->meanLogDist;
  // Train gamma and put in gammaDists[0].

  Timer::Start("fitting_distributions");
  for (size_t i = 1; i <= k; ++i)
  {
    // Use the arithmetic and geometric mean predictors that were trained in
    // Train() to estimate the statistics for the given datasetSize and k.
    meanVec(i) = aMeanPredictor.Predict(datasetSize, k);
    logMeanVec(i) = std::log(meanVec(i));
    // log(geometricMean) = \frac{1}{n} \sum(lnx_i) = mean(lnx) = meanLog
    meanLogVec(i) = std::log(gMeanPredictor.Predict(datasetSize, k));
  }
  // Fit the distribution.
  distancesDistribution.Train(logMeanVec, meanLogVec, meanVec);
  Timer::Stop("fitting_distributions");

  // Step 7. Generate the Template Probing Sequence using the maximum number of
  // projections and the maximum number of probes.
  GenerateTemplateSequence(numProj, numProbes);

  // Step 8. Use formulas (19) and (20) from the paper to predict recall and
  // selectivity, using LSHModel::Rho() and the distribution functions of the
  // gammas we fit back in Step 6.
  predictedRecall = Recall(k, numTables, numProj, numProbes, hashWidth);
  predictedSelect = Selectivity(numTables, numProj, numProbes, hashWidth);
}

// Uses paper's formula (19) to predict recall.
template <typename SortPolicy, typename ObjectiveFunction>
double LSHModel<SortPolicy, ObjectiveFunction>::Recall(size_t maxK,
                                                       size_t numTables,
                                                       size_t numProj,
                                                       size_t numProbes,
                                                       double hashWidth)
{
  double recall = 0;

  // Loop over k values, accumulating the probabilities. Then take average.
  // k starts from one because distancesDistribution(0) is the "simple" pairwise
  // distances distribution.
  for (size_t k = 1; k < maxK + 1; k++)
  {
    // Create a helper object for this value of k.
    IntegralObjective f(k, numTables, numProj, numProbes, hashWidth,
       &distancesDistribution, this);

    // TODO: change with boost integration.
    double from = 0;
    double to = 1000;
    double step = 0.001;
    double integralSum = 0;
    for (double i = from+step; i < to; i+=step)
    {
      double temp = f(i);
      if (temp > 0)
        integralSum += temp; // Use as function thanks to operator().
      else
        break; // Gamma distribution == 0 means we're past the tail.
    }
    recall += integralSum * step ;
  }
  return recall / double(maxK);
}

// Uses paper's formula (20) to compute selectivity
template <typename SortPolicy, typename ObjectiveFunction>
double LSHModel<SortPolicy, ObjectiveFunction>::Selectivity(size_t numTables,
                                                            size_t numProj,
                                                            size_t numProbes,
                                                            double hashWidth)
{

  // Create a helper object for k = 0 (pairwise distances).
  IntegralObjective f(0, numTables, numProj, numProbes, hashWidth,
      &distancesDistribution, this);

  // TODO: change with boost integration.
  double from = 0;
  double to = 1000;
  double step = 0.001;
  double integralSum = 0;
  for (double i = from+step; i < to; i+=step)
  {
    double temp = f(i);
    if (temp > 0)
      integralSum += temp; // Use as function thanks to operator().
    else
      break; // Gamma distribution == 0 means we're past the tail.
  }

  return integralSum * step ;
}

/*
 * Based on the LSHKIT implementation, not my understanding of the paper.
 */
// Probability of two points being neighbors if they are at distance chi.
template <typename SortPolicy, typename ObjectiveFunction>
double LSHModel<SortPolicy, ObjectiveFunction>::Rho(double chi,
                                                    double hashWidth,
                                                    size_t numTables,
                                                    size_t numProj,
                                                    size_t numProbes) const
{
  double rho = 0;

  for (size_t probe = 0; probe < numProbes; ++probe)
  {
    double rTemp = 1;
    for (size_t proj = 0; proj < numProj; ++proj)
    {
      rTemp *= SameBucketProbability(chi, hashWidth, 
          templateSequence(proj, probe), proj, numProj);
    }
    rho += rTemp;
  }

  return 1 - std::exp(std::log(1.0 - rho) * numTables);
}

// Probability of two points being neighbors if they are at distance chi.
template <typename SortPolicy, typename ObjectiveFunction>
double LSHModel<SortPolicy, ObjectiveFunction>::
SameBucketProbability(double chi, double hashWidth, short delta, size_t proj,
                      size_t numProj) const
{
  boost::math::normal_distribution<> phi;
  if (delta == 0)
  {
    // No perturbation - probability of two queries sharing the same bin.
    return 2 * pdf(phi, hashWidth / chi) - 1
      + std::sqrt(2 / M_PI) 
      * (std::exp(-pow((hashWidth / chi), 2) / 2.0 - 1.0)) / (hashWidth / chi);
  }
  else
  {
    // +1/-1 perturbation - probability of two queries being in adjacent bins.
    double deltaI = (proj + 1.0) / (2.0 * (numProj + 2.0));
    
    // Negative perturbation - flip deltaI.
    if (delta == -1)
      deltaI = 1 - deltaI;

    return cdf(phi, hashWidth / chi * (1 + deltaI)) 
        - cdf(phi, hashWidth / chi * deltaI);
  }
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

  trainedLSHObject = lsh;

  return trainedLSHObject;
}

// Helper function to generate perturbations.
template<typename SortPolicy, typename ObjectiveFunction>
inline force_inline
double LSHModel<SortPolicy, ObjectiveFunction>::PerturbationScore(
    const std::vector<bool>& A,
    const arma::vec& scores) const
{
  double score = 0.0;
  for (size_t i = 0; i < A.size(); ++i)
    if (A[i])
      score += scores(i); // add scores of non-zero indices
  return score;
}

// Helper function to generate perturbations.
template<typename SortPolicy, typename ObjectiveFunction>
inline force_inline
bool LSHModel<SortPolicy, ObjectiveFunction>::PerturbationShift(
    std::vector<bool>& A) const
{
  size_t maxPos = 0;
  for (size_t i = 0; i < A.size(); ++i)
    if (A[i] == 1) // Marked true.
      maxPos = i;

  if (maxPos + 1 < A.size()) // Otherwise, this is an invalid vector.
  {
    A[maxPos] = 0;
    A[maxPos + 1] = 1;
    return true; // valid
  }
  return false; // invalid
}

// Helper function to generate perturbations.
template<typename SortPolicy, typename ObjectiveFunction>
inline force_inline
bool LSHModel<SortPolicy, ObjectiveFunction>::PerturbationExpand(
    std::vector<bool>& A) const
{
  // Find the last '1' in A.
  size_t maxPos = 0;
  for (size_t i = 0; i < A.size(); ++i)
    if (A[i]) // Marked true.
      maxPos = i;

  if (maxPos + 1 < A.size()) // Otherwise, this is an invalid vector.
  {
    A[maxPos + 1] = 1;
    return true;
  }
  return false;
}

// Helper function to generate perturbations.
template<typename SortPolicy, typename ObjectiveFunction>
inline force_inline
bool LSHModel<SortPolicy, ObjectiveFunction>::PerturbationValid(
    const std::vector<bool>& A,
    size_t numProj) const
{
  // Use check to mark dimensions we have seen before in A. If a dimension is
  // seen twice (or more), A is not a valid perturbation.
  std::vector<bool> check(numProj);

  if (A.size() > 2 * numProj)
    return false; // This should never happen.

  // Check that we only see each dimension once. If not, vector is not valid.
  for (size_t i = 0; i < A.size(); ++i)
  {
    // Only check dimensions that were included.
    if (!A[i])
      continue;

    // If dimesnion is unseen thus far, mark it as seen.
    if (check[i % numProj] == false)
      check[i % numProj] = true;
    else
      return false; // If dimension was seen before, set is not valid.
  }
  // If we didn't fail, set is valid.
  return true;
}

// Generate a probing sequence for a given M, W and T.
template <typename SortPolicy, typename ObjectiveFunction>
void LSHModel<SortPolicy, ObjectiveFunction>::GenerateTemplateSequence(
    size_t numProj,
    size_t numProbes)
{
  // If no probes requested, stop here.
  if (numProbes == 0)
  {
    Log::Warn << "GenerateTemplateSequence called with numProbes = 0"
      << std::endl;
    return;
  }

  // If number of additional probes exceeds possible, set to max possible.
  if (numProbes > pow(3, numProj))
    numProbes = pow(3, numProj); // {-1, 0, 1} for each probe.

  // Calculate the expected scores based on Multi-probe LSH paper.
  arma::vec scores(2 * numProj);
  double M = (double) numProj;

  // Generate expected scores in sorted order.
  for (size_t i = 0; i < numProj; ++i)
  {
    // Everything is double to avoid integer division headache.
    double left = double(i);
    double right = 2 * M - left - 1;

    // Expected score - left boundary.
    scores[left] = (left + 1) * (left + 2) / (2 * (M + 1) * (M + 2));
    scores[right] = 1 - (left + 1)/(M + 1) + scores[left];
  }

  // A "+1" signifies a positive perturbation, a "-1" a negative one.
  arma::Col<short int> actions(2 * numProj); // will be [1 ... -1 ...]
  actions.rows(0, numProj - 1) = // First numProj rows.
    arma::ones< arma::Col<short int> > (numProj); // 1s
  actions.rows(numProj, (2 * numProj) - 1) = // Last numProj rows.
    -1 * arma::ones< arma::Col<short int> > (numProj); // -1s

  // The "acting dimension", or which of the numProj dimension to increase or
  // reduce according to the "actions".
  arma::Col<size_t> positions(2 * numProj); // Will be [0 1 2 ... 0 1 2 ...].
  positions.rows(0, numProj - 1) =
    arma::linspace< arma::Col<size_t> >(0, numProj - 1, numProj);
  positions.rows(numProj, 2 * numProj - 1) =
    arma::linspace< arma::Col<size_t> >(0, numProj - 1, numProj);

  // Perturbation sets (A) mark with 1 the (score, action, dimension) positions
  // included in a given perturbation vector. Other spaces are 0.
  std::vector<bool> Ao(2 * numProj);
  Ao[0] = 1; // Smallest vector includes only smallest score.

  std::vector< std::vector<bool> > perturbationSets;
  perturbationSets.push_back(Ao); // Storage of perturbation sets.

  std::priority_queue<
    std::pair<double, size_t>,        // contents: pairs of (score, index)
    std::vector<                      // container: vector of pairs
      std::pair<double, size_t>
      >,
    std::greater< std::pair<double, size_t> > // comparator of pairs
  > minHeap; // our minheap

  // Start by adding the lowest scoring set to the minheap.
  minHeap.push( std::make_pair(PerturbationScore(Ao, scores), 0) );

  // Allocate 1 column per perturbed "code".
  templateSequence.zeros(numProj, numProbes);

  // Column 0 is all 0s. Fill columns 1:numProbes using Lv's algorithm.
  for (size_t pvec = 1; pvec < numProbes; ++pvec)
  {
    std::vector<bool> Ai;
    do
    {
      // Get the perturbation set corresponding to the minimum score.
      Ai = perturbationSets[ minHeap.top().second ];
      minHeap.pop(); // .top() returns, .pop() removes

      // Shift operation on Ai (replace max with max+1).
      std::vector<bool> As = Ai;
      if (PerturbationShift(As) && PerturbationValid(As, numProj))
        // Don't add invalid sets.
      {
        perturbationSets.push_back(As); // add shifted set to sets
        minHeap.push(
            std::make_pair(PerturbationScore(As, scores),
            perturbationSets.size() - 1));
      }

      // Expand operation on Ai (add max+1 to set).
      std::vector<bool> Ae = Ai;
      if (PerturbationExpand(Ae) && PerturbationValid(Ae, numProj))
        // Don't add invalid sets.
      {
        perturbationSets.push_back(Ae); // add expanded set to sets
        minHeap.push(
            std::make_pair(PerturbationScore(Ae, scores),
            perturbationSets.size() - 1));
      }

    } while (!PerturbationValid(Ai, numProj));//Discard invalid perturbations

    // Found valid perturbation set Ai. Construct perturbation vector from set.
    for (size_t pos = 0; pos < Ai.size(); ++pos)
    {
      // If Ai[pos] is marked, set template to +/- 1.
      if (Ai[pos] == 1)
        templateSequence(positions(pos), pvec) = actions(pos);
    }
  }
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
