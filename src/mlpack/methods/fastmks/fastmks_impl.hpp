/**
 * @file methods/fastmks/fastmks_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the FastMKS class (fast max-kernel search).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_FASTMKS_FASTMKS_IMPL_HPP
#define MLPACK_METHODS_FASTMKS_FASTMKS_IMPL_HPP

// In case it hasn't yet been included.
#include "fastmks.hpp"

#include "fastmks_rules.hpp"

namespace mlpack {

// No data; create a model on an empty dataset.
template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>::FastMKS(const bool singleMode,
                                                const bool naive) :
    referenceSet(new MatType()),
    referenceTree(NULL),
    treeOwner(true),
    setOwner(true),
    singleMode(singleMode),
    naive(naive)
{
  if (!naive)
    referenceTree = new Tree(*referenceSet);
}

// No instantiated kernel.
template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>::FastMKS(
    const MatType& referenceSet,
    const bool singleMode,
    const bool naive) :
    referenceSet(&referenceSet),
    referenceTree(NULL),
    treeOwner(true),
    setOwner(false),
    singleMode(singleMode),
    naive(naive)
{
  if (!naive)
    referenceTree = new Tree(referenceSet);
}

// Instantiated kernel.
template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>::FastMKS(const MatType& referenceSet,
                                                KernelType& kernel,
                                                const bool singleMode,
                                                const bool naive) :
    referenceSet(&referenceSet),
    referenceTree(NULL),
    treeOwner(true),
    setOwner(false),
    singleMode(singleMode),
    naive(naive),
    distance(kernel)
{
  // If necessary, the reference tree should be built.  There is no query tree.
  if (!naive)
    referenceTree = new Tree(referenceSet, distance);
}

// No instantiated kernel.
template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>::FastMKS(
    MatType&& referenceSet,
    const bool singleMode,
    const bool naive) :
    referenceSet(naive ? new MatType(std::move(referenceSet)) : NULL),
    referenceTree(NULL),
    treeOwner(true),
    setOwner(naive),
    singleMode(singleMode),
    naive(naive)
{
  if (!naive)
  {
    referenceTree = new Tree(std::move(referenceSet));
    referenceSet = &referenceTree->Dataset();
  }
}

// Instantiated kernel.
template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>::FastMKS(MatType&& referenceSet,
                                                KernelType& kernel,
                                                const bool singleMode,
                                                const bool naive) :
    referenceSet(naive ? new MatType(std::move(referenceSet)) : NULL),
    referenceTree(NULL),
    treeOwner(true),
    setOwner(naive),
    singleMode(singleMode),
    naive(naive),
    distance(kernel)
{
  // If necessary, the reference tree should be built.  There is no query tree.
  if (!naive)
  {
    referenceTree = new Tree(referenceSet, distance);
    referenceSet = &referenceTree->Dataset();
  }
}

// One dataset, pre-built tree.
template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>::FastMKS(Tree* referenceTree,
                                                const bool singleMode) :
    referenceSet(&referenceTree->Dataset()),
    referenceTree(referenceTree),
    treeOwner(false),
    setOwner(false),
    singleMode(singleMode),
    naive(false),
    distance(referenceTree->Distance())
{
  // Nothing to do.
}

template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>::FastMKS(const FastMKS& other) :
    referenceSet(NULL),
    referenceTree(other.referenceTree ? new Tree(*other.referenceTree) : NULL),
    treeOwner(other.referenceTree != NULL),
    setOwner(other.referenceTree == NULL),
    singleMode(other.singleMode),
    naive(other.naive),
    distance(other.distance)
{
  // Set reference set correctly.
  if (referenceTree)
    referenceSet = &referenceTree->Dataset();
  else
    referenceSet = new MatType(*other.referenceSet);
}

template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>::FastMKS(FastMKS&& other) :
    referenceSet(other.referenceSet),
    referenceTree(other.referenceTree),
    treeOwner(other.treeOwner),
    setOwner(other.setOwner),
    singleMode(other.singleMode),
    naive(other.naive),
    distance(std::move(other.distance))
{
  // Clear information from the other.
  other.referenceSet = NULL;
  other.referenceTree = NULL;
  other.treeOwner = false;
  other.setOwner = false;
  other.singleMode = false;
  other.naive = false;
}

template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>&
FastMKS<KernelType, MatType, TreeType>::operator=(const FastMKS& other)
{
  if (this == &other)
    return *this;

  // Clear anything we currently have.
  if (treeOwner)
    delete referenceTree;
  if (setOwner)
    delete referenceSet;

  referenceTree = NULL;
  referenceSet = NULL;

  if (other.referenceTree)
  {
    referenceTree = new Tree(*other.referenceTree);
    referenceSet = &referenceTree->Dataset();
    treeOwner = true;
    setOwner = false;
  }
  else
  {
    referenceSet = new MatType(*other.referenceSet);
    treeOwner = false;
    setOwner = true;
  }

  singleMode = other.singleMode;
  naive = other.naive;
}

template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>&
FastMKS<KernelType, MatType, TreeType>::operator=(FastMKS&& other)
{
  if (this != &other)
  {
    referenceSet = other.referenceSet;
    referenceTree = other.referenceTree;
    treeOwner = other.treeOwner;
    setOwner = other.setOwner;
    singleMode = other.singleMode;
    naive = other.naive;
    distance = std::move(other.distance);

    // Clear information from the other.
    other.referenceSet = nullptr;
    other.referenceTree = nullptr;
    other.treeOwner = false;
    other.setOwner = false;
    other.singleMode = false;
    other.naive = false;
  }
  return *this;
}

template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
FastMKS<KernelType, MatType, TreeType>::~FastMKS()
{
  // If we created the trees, we must delete them.
  if (treeOwner && referenceTree)
    delete referenceTree;
  if (setOwner)
    delete referenceSet;
}

template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Train(const MatType& referenceSet)
{
  if (setOwner)
    delete this->referenceSet;

  this->referenceSet = &referenceSet;
  this->setOwner = false;

  if (!naive)
  {
    if (treeOwner && referenceTree)
      delete referenceTree;
    referenceTree = new Tree(referenceSet, distance);
    treeOwner = true;
  }
}

template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Train(const MatType& referenceSet,
                                                   KernelType& kernel)
{
  if (setOwner)
    delete this->referenceSet;

  this->referenceSet = &referenceSet;
  this->distance = IPMetric<KernelType>(kernel);
  this->setOwner = false;

  if (!naive)
  {
    if (treeOwner && referenceTree)
      delete referenceTree;
    referenceTree = new Tree(referenceSet, distance);
    treeOwner = true;
  }
}

template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Train(MatType&& referenceSet)
{
  if (setOwner)
    delete this->referenceSet;

  if (!naive)
  {
    if (treeOwner && referenceTree)
      delete referenceTree;
    referenceTree = new Tree(std::move(referenceSet), distance);
    referenceSet = referenceTree->Dataset();
    treeOwner = true;
    setOwner = false;
  }
  else
  {
    this->referenceSet = new MatType(std::move(referenceSet));
    this->setOwner = true;
  }
}

template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Train(MatType&& referenceSet,
                                                   KernelType& kernel)
{
  if (setOwner)
    delete this->referenceSet;

  this->distance = IPMetric<KernelType>(kernel);

  if (!naive)
  {
    if (treeOwner && referenceTree)
      delete referenceTree;
    referenceTree = new Tree(std::move(referenceSet), distance);
    treeOwner = true;
    setOwner = false;
  }
  else
  {
    this->referenceSet = new MatType(std::move(referenceSet));
    this->setOwner = true;
  }
}

template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Train(Tree* tree)
{
  if (naive)
    throw std::invalid_argument("cannot call FastMKS::Train() with a tree when "
        "in naive search mode");

  if (setOwner)
    delete this->referenceSet;

  this->referenceSet = &tree->Dataset();
  this->distance = IPMetric<KernelType>(tree->Distance().Kernel());
  this->setOwner = false;

  if (treeOwner && referenceTree)
    delete referenceTree;

  this->referenceTree = tree;
  this->treeOwner = true;
}

template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Search(
    const MatType& querySet,
    const size_t k,
    arma::Mat<size_t>& indices,
    arma::mat& kernels)
{
  if (k > referenceSet->n_cols)
  {
    std::stringstream ss;
    ss << "requested value of k (" << k << ") is greater than the number of "
        << "points in the reference set (" << referenceSet->n_cols << ")";
    throw std::invalid_argument(ss.str());
  }

  if (querySet.n_rows != referenceSet->n_rows)
  {
    std::stringstream ss;
    ss << "The number of dimensions in the query set (" << querySet.n_rows
        << ") must be equal to the number of dimensions in the reference set ("
        << referenceSet->n_rows << ")!";
    throw std::invalid_argument(ss.str());
  }

  // No remapping will be necessary because we are using the cover tree.
  indices.set_size(k, querySet.n_cols);
  kernels.set_size(k, querySet.n_cols);

  // Naive implementation.
  if (naive)
  {
    // Simple double loop.  Stupid, slow, but a good benchmark.
    for (size_t q = 0; q < querySet.n_cols; ++q)
    {
      const Candidate def = std::make_pair(-DBL_MAX, size_t() - 1);
      std::vector<Candidate> cList(k, def);
      CandidateList pqueue(CandidateCmp(), std::move(cList));

      for (size_t r = 0; r < referenceSet->n_cols; ++r)
      {
        const double eval = distance.Kernel().Evaluate(querySet.col(q),
                                                       referenceSet->col(r));

        if (eval > pqueue.top().first)
        {
          Candidate c = std::make_pair(eval, r);
          pqueue.pop();
          pqueue.push(c);
        }
      }

      for (size_t j = 1; j <= k; ++j)
      {
        indices(k - j, q) = pqueue.top().second;
        kernels(k - j, q) = pqueue.top().first;
        pqueue.pop();
      }
    }

    return;
  }

  // Single-tree implementation.
  if (singleMode)
  {
    // Create rules object (this will store the results).  This constructor
    // precalculates each self-kernel value.
    using RuleType = FastMKSRules<KernelType, Tree>;
    RuleType rules(*referenceSet, querySet, k, distance.Kernel());

    typename Tree::template SingleTreeTraverser<RuleType> traverser(rules);

    for (size_t i = 0; i < querySet.n_cols; ++i)
      traverser.Traverse(i, *referenceTree);

    Log::Info << rules.BaseCases() << " base cases." << std::endl;
    Log::Info << rules.Scores() << " scores." << std::endl;

    rules.GetResults(indices, kernels);

    return;
  }

  // Dual-tree implementation.  First, we need to build the query tree.  We are
  // assuming it doesn't map anything...
  Tree queryTree(querySet);

  Search(&queryTree, k, indices, kernels);
}

template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Search(
    Tree* queryTree,
    const size_t k,
    arma::Mat<size_t>& indices,
    arma::mat& kernels)
{
  if (k > referenceSet->n_cols)
  {
    std::stringstream ss;
    ss << "requested value of k (" << k << ") is greater than the number of "
        << "points in the reference set (" << referenceSet->n_cols << ")";
    throw std::invalid_argument(ss.str());
  }
  if (queryTree->Dataset().n_rows != referenceSet->n_rows)
  {
    std::stringstream ss;
    ss << "The number of dimensions in the query set ("
        << queryTree->Dataset().n_rows << ") must be equal to the number of "
        << "dimensions in the reference set (" << referenceSet->n_rows << ")!";
    throw std::invalid_argument(ss.str());
  }

  // If either naive mode or single mode is specified, this must fail.
  if (naive || singleMode)
  {
    throw std::invalid_argument("can't call Search() with a query tree when "
        "single mode or naive search is enabled");
  }

  // No remapping will be necessary because we are using the cover tree.
  indices.set_size(k, queryTree->Dataset().n_cols);
  kernels.set_size(k, queryTree->Dataset().n_cols);

  using RuleType = FastMKSRules<KernelType, Tree>;
  RuleType rules(*referenceSet, queryTree->Dataset(), k, distance.Kernel());

  typename Tree::template DualTreeTraverser<RuleType> traverser(rules);

  traverser.Traverse(*queryTree, *referenceTree);

  Log::Info << rules.BaseCases() << " base cases." << std::endl;
  Log::Info << rules.Scores() << " scores." << std::endl;

  rules.GetResults(indices, kernels);
}

template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
void FastMKS<KernelType, MatType, TreeType>::Search(
    const size_t k,
    arma::Mat<size_t>& indices,
    arma::mat& kernels)
{
  // No remapping will be necessary because we are using the cover tree.
  indices.set_size(k, referenceSet->n_cols);
  kernels.set_size(k, referenceSet->n_cols);

  // Naive implementation.
  if (naive)
  {
    // Simple double loop.  Stupid, slow, but a good benchmark.
    for (size_t q = 0; q < referenceSet->n_cols; ++q)
    {
      const Candidate def = std::make_pair(-DBL_MAX, size_t() - 1);
      std::vector<Candidate> cList(k, def);
      CandidateList pqueue(CandidateCmp(), std::move(cList));

      for (size_t r = 0; r < referenceSet->n_cols; ++r)
      {
        if (q == r)
          continue; // Don't return the point as its own candidate.

        const double eval = distance.Kernel().Evaluate(referenceSet->col(q),
                                                       referenceSet->col(r));

        if (eval > pqueue.top().first)
        {
          Candidate c = std::make_pair(eval, r);
          pqueue.pop();
          pqueue.push(c);
        }
      }

      for (size_t j = 1; j <= k; ++j)
      {
        indices(k - j, q) = pqueue.top().second;
        kernels(k - j, q) = pqueue.top().first;
        pqueue.pop();
      }
    }

    return;
  }

  // Single-tree implementation.
  if (singleMode)
  {
    // Create rules object (this will store the results).  This constructor
    // precalculates each self-kernel value.
    using RuleType = FastMKSRules<KernelType, Tree>;
    RuleType rules(*referenceSet, *referenceSet, k, distance.Kernel());

    typename Tree::template SingleTreeTraverser<RuleType> traverser(rules);

    for (size_t i = 0; i < referenceSet->n_cols; ++i)
      traverser.Traverse(i, *referenceTree);

    // Save the number of pruned nodes.
    const size_t numPrunes = traverser.NumPrunes();

    Log::Info << "Pruned " << numPrunes << " nodes." << std::endl;

    Log::Info << rules.BaseCases() << " base cases." << std::endl;
    Log::Info << rules.Scores() << " scores." << std::endl;

    rules.GetResults(indices, kernels);

    return;
  }

  // Dual-tree implementation.
  Search(referenceTree, k, indices, kernels);
}

//! Serialize the model.
template<typename KernelType,
         typename MatType,
         template<typename TreeDistanceType,
                  typename TreeStatType,
                  typename TreeMatType> class TreeType>
template<typename Archive>
void FastMKS<KernelType, MatType, TreeType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  // Serialize preferences for search.
  ar(CEREAL_NVP(naive));
  ar(CEREAL_NVP(singleMode));

  // If we are doing naive search, serialize the dataset.  Otherwise we
  // serialize the tree.
  if (naive)
  {
    if (cereal::is_loading<Archive>())
    {
      if (setOwner && referenceSet)
        delete referenceSet;

      setOwner = true;
    }

    ar(CEREAL_POINTER(const_cast<MatType*&>(referenceSet)));
    ar(CEREAL_NVP(distance));
  }
  else
  {
    // Delete the current reference tree, if necessary.
    if (cereal::is_loading<Archive>())
    {
      if (treeOwner && referenceTree)
        delete referenceTree;

      treeOwner = true;
    }

    ar(CEREAL_POINTER(referenceTree));

    if (cereal::is_loading<Archive>())
    {
      if (setOwner && referenceSet)
        delete referenceSet;

      referenceSet = &referenceTree->Dataset();
      distance = IPMetric<KernelType>(referenceTree->Distance().Kernel());
      setOwner = false;
    }
  }
}

} // namespace mlpack

#endif
