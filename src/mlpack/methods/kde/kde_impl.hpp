/**
 * @file kde_impl.hpp
 * @author Roberto Hueso
 *
 * Implementation of Kernel Density Estimation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "kde.hpp"
#include "kde_rules.hpp"

namespace mlpack {
namespace kde {

//! Construct tree that rearranges the dataset.
template<typename TreeType, typename MatType>
TreeType* BuildTree(
    MatType&& dataset,
    std::vector<size_t>& oldFromNew,
    const typename std::enable_if<
        tree::TreeTraits<TreeType>::RearrangesDataset>::type* = 0)
{
  return new TreeType(std::forward<MatType>(dataset), oldFromNew);
}

//! Construct tree that doesn't rearrange the dataset.
template<typename TreeType, typename MatType>
TreeType* BuildTree(
    MatType&& dataset,
    const std::vector<size_t>& /* oldFromNew */,
    const typename std::enable_if<
        !tree::TreeTraits<TreeType>::RearrangesDataset>::type* = 0)
{
  return new TreeType(std::forward<MatType>(dataset));
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
KDE<KernelType,
    MetricType,
    MatType,
    TreeType,
    DualTreeTraversalType,
    SingleTreeTraversalType>::
KDE(const double relError,
    const double absError,
    KernelType kernel,
    const KDEMode mode,
    MetricType metric,
    const bool monteCarlo,
    const double mcProb,
    const size_t initialSampleSize,
    const double mcEntryCoef,
    const double mcBreakCoef) :
    kernel(kernel),
    metric(metric),
    referenceTree(nullptr),
    oldFromNewReferences(nullptr),
    relError(relError),
    absError(absError),
    ownsReferenceTree(false),
    trained(false),
    mode(mode),
    monteCarlo(monteCarlo),
    initialSampleSize(initialSampleSize)
{
  CheckErrorValues(relError, absError);
  MCProb(mcProb);
  MCEntryCoef(mcEntryCoef);
  MCBreakCoef(mcBreakCoef);
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
KDE<KernelType,
    MetricType,
    MatType,
    TreeType,
    DualTreeTraversalType,
    SingleTreeTraversalType>::
KDE(const KDE& other) :
    kernel(KernelType(other.kernel)),
    metric(MetricType(other.metric)),
    relError(other.relError),
    absError(other.absError),
    ownsReferenceTree(other.ownsReferenceTree),
    trained(other.trained),
    mode(other.mode),
    monteCarlo(other.monteCarlo),
    mcProb(other.mcProb),
    initialSampleSize(other.initialSampleSize),
    mcEntryCoef(other.mcEntryCoef),
    mcBreakCoef(other.mcBreakCoef)
{
  if (trained)
  {
    if (ownsReferenceTree)
    {
      oldFromNewReferences =
          new std::vector<size_t>(*other.oldFromNewReferences);
      referenceTree = new Tree(*other.referenceTree);
    }
    else
    {
      oldFromNewReferences = other.oldFromNewReferences;
      referenceTree = other.referenceTree;
    }
  }
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
KDE<KernelType,
    MetricType,
    MatType,
    TreeType,
    DualTreeTraversalType,
    SingleTreeTraversalType>::
KDE(KDE&& other) :
    kernel(std::move(other.kernel)),
    metric(std::move(other.metric)),
    referenceTree(other.referenceTree),
    oldFromNewReferences(other.oldFromNewReferences),
    relError(other.relError),
    absError(other.absError),
    ownsReferenceTree(other.ownsReferenceTree),
    trained(other.trained),
    mode(other.mode),
    monteCarlo(other.monteCarlo),
    mcProb(other.mcProb),
    initialSampleSize(other.initialSampleSize),
    mcEntryCoef(other.mcEntryCoef),
    mcBreakCoef(other.mcBreakCoef)
{
  other.kernel = std::move(KernelType());
  other.metric = std::move(MetricType());
  other.referenceTree = nullptr;
  other.oldFromNewReferences = nullptr;
  other.relError = KDEDefaultParams::relError;
  other.absError = KDEDefaultParams::absError;
  other.ownsReferenceTree = false;
  other.trained = false;
  other.mode = KDEDefaultParams::mode;
  other.monteCarlo = KDEDefaultParams::monteCarlo;
  other.mcProb = KDEDefaultParams::mcProb;
  other.initialSampleSize = KDEDefaultParams::initialSampleSize;
  other.mcEntryCoef = KDEDefaultParams::mcEntryCoef;
  other.mcBreakCoef = KDEDefaultParams::mcBreakCoef;
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
KDE<KernelType,
    MetricType,
    MatType,
    TreeType,
    DualTreeTraversalType,
    SingleTreeTraversalType>&
KDE<KernelType,
    MetricType,
    MatType,
    TreeType,
    DualTreeTraversalType,
    SingleTreeTraversalType>::
operator=(KDE other)
{
  // Clean memory.
  if (ownsReferenceTree)
  {
    delete referenceTree;
    delete oldFromNewReferences;
  }

  // Move the other object.
  this->kernel = std::move(other.kernel);
  this->metric = std::move(other.metric);
  this->referenceTree = std::move(other.referenceTree);
  this->oldFromNewReferences = std::move(other.oldFromNewReferences);
  this->relError = other.relError;
  this->absError = other.absError;
  this->ownsReferenceTree = other.ownsReferenceTree;
  this->trained = other.trained;
  this->mode = other.mode;
  this->monteCarlo = other.monteCarlo;
  this->mcProb = other.mcProb;
  this->initialSampleSize = other.initialSampleSize;
  this->mcEntryCoef = other.mcEntryCoef;
  this->mcBreakCoef = other.mcBreakCoef;

  return *this;
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
KDE<KernelType,
    MetricType,
    MatType,
    TreeType,
    DualTreeTraversalType,
    SingleTreeTraversalType>::
~KDE()
{
  if (ownsReferenceTree)
  {
    delete referenceTree;
    delete oldFromNewReferences;
  }
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void KDE<KernelType,
         MetricType,
         MatType,
         TreeType,
         DualTreeTraversalType,
         SingleTreeTraversalType>::
Train(MatType referenceSet)
{
  // Check if referenceSet is not an empty set.
  if (referenceSet.n_cols == 0)
  {
    throw std::invalid_argument("cannot train KDE model with an empty "
                                "reference set");
  }

  if (ownsReferenceTree)
  {
    delete referenceTree;
    delete oldFromNewReferences;
  }

  this->ownsReferenceTree = true;
  Timer::Start("building_reference_tree");
  this->oldFromNewReferences = new std::vector<size_t>;
  this->referenceTree = BuildTree<Tree>(std::move(referenceSet),
                                        *oldFromNewReferences);
  Timer::Stop("building_reference_tree");
  this->trained = true;
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void KDE<KernelType,
         MetricType,
         MatType,
         TreeType,
         DualTreeTraversalType,
         SingleTreeTraversalType>::
Train(Tree* referenceTree, std::vector<size_t>* oldFromNewReferences)
{
  // Check if referenceTree dataset is not an empty set.
  if (referenceTree->Dataset().n_cols == 0)
  {
    throw std::invalid_argument("cannot train KDE model with an empty "
                                "reference set");
  }

  if (ownsReferenceTree == true)
  {
    delete this->referenceTree;
    delete this->oldFromNewReferences;
  }

  this->ownsReferenceTree = false;
  this->referenceTree = referenceTree;
  this->oldFromNewReferences = oldFromNewReferences;
  this->trained = true;
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void KDE<KernelType,
         MetricType,
         MatType,
         TreeType,
         DualTreeTraversalType,
         SingleTreeTraversalType>::
Evaluate(MatType querySet, arma::vec& estimations)
{
  if (mode == DUAL_TREE_MODE)
  {
    Timer::Start("building_query_tree");
    std::vector<size_t> oldFromNewQueries;
    Tree* queryTree = BuildTree<Tree>(std::move(querySet), oldFromNewQueries);
    Timer::Stop("building_query_tree");
    this->Evaluate(queryTree, oldFromNewQueries, estimations);
    delete queryTree;
  }
  else if (mode == SINGLE_TREE_MODE)
  {
    // Get estimations vector ready.
    estimations.clear();
    estimations.set_size(querySet.n_cols);
    estimations.fill(arma::fill::zeros);

    // Check whether has already been trained.
    if (!trained)
    {
      throw std::runtime_error("cannot evaluate KDE model: model needs to be "
                               "trained before evaluation");
    }

    // Check querySet has at least 1 element to evaluate.
    if (querySet.n_cols == 0)
    {
      Log::Warn << "KDE::Evaluate(): querySet is empty, no predictions will "
                << "be returned" << std::endl;
      return;
    }

    // Check whether dimensions match.
    if (querySet.n_rows != referenceTree->Dataset().n_rows)
    {
      throw std::invalid_argument("cannot evaluate KDE model: querySet and "
                                  "referenceSet dimensions don't match");
    }

    Timer::Start("computing_kde");

    // Evaluate.
    typedef KDERules<MetricType, KernelType, Tree> RuleType;
    RuleType rules = RuleType(referenceTree->Dataset(),
                              querySet,
                              estimations,
                              relError,
                              absError,
                              mcProb,
                              initialSampleSize,
                              mcEntryCoef,
                              mcBreakCoef,
                              metric,
                              kernel,
                              monteCarlo,
                              false);

    // Create traverser.
    SingleTreeTraversalType<RuleType> traverser(rules);

    // Traverse for each point.
    for (size_t i = 0; i < querySet.n_cols; ++i)
      traverser.Traverse(i, *referenceTree);

    estimations /= referenceTree->Dataset().n_cols;
    Timer::Stop("computing_kde");

    Log::Info << rules.Scores() << " node combinations were scored."
              << std::endl;
    Log::Info << rules.BaseCases() << " base cases were calculated."
              << std::endl;
  }
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void KDE<KernelType,
         MetricType,
         MatType,
         TreeType,
         DualTreeTraversalType,
         SingleTreeTraversalType>::
Evaluate(Tree* queryTree,
         const std::vector<size_t>& oldFromNewQueries,
         arma::vec& estimations)
{
  // Get estimations vector ready.
  estimations.clear();
  estimations.set_size(queryTree->Dataset().n_cols);
  estimations.fill(arma::fill::zeros);

  // Check whether has already been trained.
  if (!trained)
  {
    throw std::runtime_error("cannot evaluate KDE model: model needs to be "
                             "trained before evaluation");
  }

  // Check querySet has at least 1 element to evaluate.
  if (queryTree->Dataset().n_cols == 0)
  {
    Log::Warn << "KDE::Evaluate(): querySet is empty, no predictions will "
              << "be returned" << std::endl;
    return;
  }

  // Check whether dimensions match.
  if (queryTree->Dataset().n_rows != referenceTree->Dataset().n_rows)
  {
    throw std::invalid_argument("cannot evaluate KDE model: querySet and "
                                "referenceSet dimensions don't match");
  }

  // Check the mode is correct.
  if (mode != DUAL_TREE_MODE)
  {
    throw std::invalid_argument("cannot evaluate KDE model: cannot use "
                                "a query tree when mode is different from "
                                "dual-tree");
  }

  // Clean accumulated alpha if Monte Carlo estimations are available.
  if (monteCarlo && std::is_same<KernelType, kernel::GaussianKernel>::value)
  {
    Timer::Start("cleaning_query_tree");
    KDECleanRules<Tree> cleanRules;
    SingleTreeTraversalType<KDECleanRules<Tree>> cleanTraverser(cleanRules);
    cleanTraverser.Traverse(0, *queryTree);
    Timer::Stop("cleaning_query_tree");
  }

  Timer::Start("computing_kde");

  // Evaluate.
  typedef KDERules<MetricType, KernelType, Tree> RuleType;
  RuleType rules = RuleType(referenceTree->Dataset(),
                            queryTree->Dataset(),
                            estimations,
                            relError,
                            absError,
                            mcProb,
                            initialSampleSize,
                            mcEntryCoef,
                            mcBreakCoef,
                            metric,
                            kernel,
                            monteCarlo,
                            false);

  // Create traverser.
  DualTreeTraversalType<RuleType> traverser(rules);
  traverser.Traverse(*queryTree, *referenceTree);
  estimations /= referenceTree->Dataset().n_cols;
  Timer::Stop("computing_kde");

  // Rearrange if necessary.
  RearrangeEstimations(oldFromNewQueries, estimations);

  Log::Info << rules.Scores() << " node combinations were scored." << std::endl;
  Log::Info << rules.BaseCases() << " base cases were calculated." << std::endl;
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void KDE<KernelType,
         MetricType,
         MatType,
         TreeType,
         DualTreeTraversalType,
         SingleTreeTraversalType>::
Evaluate(arma::vec& estimations)
{
  // Check whether has already been trained.
  if (!trained)
  {
    throw std::runtime_error("cannot evaluate KDE model: model needs to be "
                             "trained before evaluation");
  }

  // Get estimations vector ready.
  estimations.clear();
  estimations.set_size(referenceTree->Dataset().n_cols);
  estimations.fill(arma::fill::zeros);

  // Clean accumulated alpha if Monte Carlo estimations are available.
  if (monteCarlo && std::is_same<KernelType, kernel::GaussianKernel>::value)
  {
    Timer::Start("cleaning_query_tree");
    KDECleanRules<Tree> cleanRules;
    SingleTreeTraversalType<KDECleanRules<Tree>> cleanTraverser(cleanRules);
    cleanTraverser.Traverse(0, *referenceTree);
    Timer::Stop("cleaning_query_tree");
  }

  Timer::Start("computing_kde");

  // Evaluate.
  typedef KDERules<MetricType, KernelType, Tree> RuleType;
  RuleType rules = RuleType(referenceTree->Dataset(),
                            referenceTree->Dataset(),
                            estimations,
                            relError,
                            absError,
                            mcProb,
                            initialSampleSize,
                            mcEntryCoef,
                            mcBreakCoef,
                            metric,
                            kernel,
                            monteCarlo,
                            true);

  if (mode == DUAL_TREE_MODE)
  {
    // Create traverser.
    DualTreeTraversalType<RuleType> traverser(rules);
    traverser.Traverse(*referenceTree, *referenceTree);
  }
  else if (mode == SINGLE_TREE_MODE)
  {
    SingleTreeTraversalType<RuleType> traverser(rules);
    for (size_t i = 0; i < referenceTree->Dataset().n_cols; ++i)
      traverser.Traverse(i, *referenceTree);
  }

  estimations /= referenceTree->Dataset().n_cols;
  // Rearrange if necessary.
  RearrangeEstimations(*oldFromNewReferences, estimations);
  Timer::Stop("computing_kde");

  Log::Info << rules.Scores() << " node combinations were scored." << std::endl;
  Log::Info << rules.BaseCases() << " base cases were calculated." << std::endl;
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void KDE<KernelType,
         MetricType,
         MatType,
         TreeType,
         DualTreeTraversalType,
         SingleTreeTraversalType>::
RelativeError(const double newError)
{
  CheckErrorValues(newError, absError);
  relError = newError;
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void KDE<KernelType,
         MetricType,
         MatType,
         TreeType,
         DualTreeTraversalType,
         SingleTreeTraversalType>::
AbsoluteError(const double newError)
{
  CheckErrorValues(relError, newError);
  absError = newError;
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void KDE<KernelType,
         MetricType,
         MatType,
         TreeType,
         DualTreeTraversalType,
         SingleTreeTraversalType>::
MCProb(const double newProb)
{
  if (newProb < 0 || newProb >= 1)
  {
    throw std::invalid_argument("Monte Carlo probability must be a value "
                                "greater than or equal to 0 and smaller than"
                                "1");
  }
  mcProb = newProb;
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void KDE<KernelType,
         MetricType,
         MatType,
         TreeType,
         DualTreeTraversalType,
         SingleTreeTraversalType>::
MCEntryCoef(const double newCoef)
{
  if (newCoef < 1)
  {
    throw std::invalid_argument("Monte Carlo entry coefficient must be a value "
                                "greater than or equal to 1");
  }
  mcEntryCoef = newCoef;
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void KDE<KernelType,
         MetricType,
         MatType,
         TreeType,
         DualTreeTraversalType,
         SingleTreeTraversalType>::
MCBreakCoef(const double newCoef)
{
  if (newCoef <= 0 || newCoef > 1)
  {
    throw std::invalid_argument("Monte Carlo break coefficient must be a value "
                                "greater than 0 and less than or equal to 1");
  }
  mcBreakCoef = newCoef;
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
template<typename Archive>
void KDE<KernelType,
         MetricType,
         MatType,
         TreeType,
         DualTreeTraversalType,
         SingleTreeTraversalType>::
serialize(Archive& ar, const unsigned int version)
{
  // Serialize preferences.
  ar & BOOST_SERIALIZATION_NVP(relError);
  ar & BOOST_SERIALIZATION_NVP(absError);
  ar & BOOST_SERIALIZATION_NVP(trained);
  ar & BOOST_SERIALIZATION_NVP(mode);

  // Backward compatibility: Old versions of KDE did not need to handle Monte
  // Carlo parameters.
  if (version > 0)
  {
    ar & BOOST_SERIALIZATION_NVP(monteCarlo);
    ar & BOOST_SERIALIZATION_NVP(mcProb);
    ar & BOOST_SERIALIZATION_NVP(initialSampleSize);
    ar & BOOST_SERIALIZATION_NVP(mcEntryCoef);
    ar & BOOST_SERIALIZATION_NVP(mcBreakCoef);
  }
  else if (Archive::is_loading::value)
  {
    monteCarlo = KDEDefaultParams::monteCarlo;
    mcProb = KDEDefaultParams::mcProb;
    initialSampleSize = KDEDefaultParams::initialSampleSize;
    mcEntryCoef = KDEDefaultParams::mcEntryCoef;
    mcBreakCoef = KDEDefaultParams::mcBreakCoef;
  }

  // If we are loading, clean up memory if necessary.
  if (Archive::is_loading::value)
  {
    if (ownsReferenceTree && referenceTree)
    {
      delete referenceTree;
      delete oldFromNewReferences;
    }
    // After loading tree, we own it.
    ownsReferenceTree = true;
  }

  // Serialize the rest of values.
  ar & BOOST_SERIALIZATION_NVP(kernel);
  ar & BOOST_SERIALIZATION_NVP(metric);
  ar & BOOST_SERIALIZATION_NVP(referenceTree);
  ar & BOOST_SERIALIZATION_NVP(oldFromNewReferences);
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void KDE<KernelType,
         MetricType,
         MatType,
         TreeType,
         DualTreeTraversalType,
         SingleTreeTraversalType>::
CheckErrorValues(const double relError, const double absError)
{
  if (relError < 0 || relError > 1)
  {
    throw std::invalid_argument("Relative error tolerance must be a value "
                                "between 0 and 1");
  }
  if (absError < 0)
  {
    throw std::invalid_argument("Absolute error tolerance must be a value "
                                "greater than or equal to 0");
  }
}

template<typename KernelType,
         typename MetricType,
         typename MatType,
         template<typename TreeMetricType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType,
         template<typename> class DualTreeTraversalType,
         template<typename> class SingleTreeTraversalType>
void KDE<KernelType,
         MetricType,
         MatType,
         TreeType,
         DualTreeTraversalType,
         SingleTreeTraversalType>::
RearrangeEstimations(const std::vector<size_t>& oldFromNew,
                     arma::vec& estimations)
{
  if (tree::TreeTraits<Tree>::RearrangesDataset)
  {
    const size_t nQueries = oldFromNew.size();
    arma::vec rearrangedEstimations(nQueries);

    // Remap vector.
    for (size_t i = 0; i < nQueries; ++i)
      rearrangedEstimations(oldFromNew.at(i)) = estimations(i);

    estimations = std::move(rearrangedEstimations);
  }
}

} // namespace kde
} // namespace mlpack
