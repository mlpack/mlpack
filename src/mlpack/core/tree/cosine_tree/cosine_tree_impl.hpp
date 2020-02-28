/**
 * @file cosine_tree_impl.hpp
 * @author Sriram S K
 *
 * Implementation of cosine tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_IMPL_HPP
#define MLPACK_CORE_TREE_COSINE_TREE_COSINE_TREE_IMPL_HPP

#include "cosine_tree.hpp"
#include <mlpack/core/util/log.hpp>

namespace mlpack {
namespace tree {

/**
 * Serialize the tree.
 */
template<typename Archive>
void CosineTree::serialize(Archive& ar, const unsigned int /* version */)
{
  // If we're loading, and we have children, they need to be deleted.
  if (Archive::is_loading::value)
  {
    if (left)
      delete left;
    if (right)
      delete right;
    if (!parent)
      delete dataset;

    parent = NULL;
    left = NULL;
    right = NULL;
  }

  ar & BOOST_SERIALIZATION_NVP(delta);
  ar & BOOST_SERIALIZATION_NVP(basis);
  ar & BOOST_SERIALIZATION_NVP(indices);
  ar & BOOST_SERIALIZATION_NVP(l2NormsSquared);
  ar & BOOST_SERIALIZATION_NVP(centroid);
  ar & BOOST_SERIALIZATION_NVP(basisVector);
  ar & BOOST_SERIALIZATION_NVP(splitPointIndex);
  ar & BOOST_SERIALIZATION_NVP(numColumns);
  ar & BOOST_SERIALIZATION_NVP(l2Error);
  ar & BOOST_SERIALIZATION_NVP(frobNormSquared);
  ar & BOOST_SERIALIZATION_NVP(localDataset);
  ar & BOOST_SERIALIZATION_NVP(dataset);

  // Save children last; otherwise boost::serialization gets confused.
  bool hasLeft = (left != NULL);
  bool hasRight = (right != NULL);

  ar & BOOST_SERIALIZATION_NVP(hasLeft);
  ar & BOOST_SERIALIZATION_NVP(hasRight);
  if (hasLeft)
    ar & BOOST_SERIALIZATION_NVP(left);
  if (hasRight)
    ar & BOOST_SERIALIZATION_NVP(right);

  if (Archive::is_loading::value)
  {
    if (left)
      left->parent = this;
    if (right)
      right->parent = this;
  }
}

/**
 * Initialize the tree from an archive.
 */
template<typename Archive>
CosineTree::CosineTree(
    Archive& ar,
    const typename std::enable_if_t<Archive::is_loading::value>*) :
    CosineTree() // Create an empty CosineTree.
{
  // We've delegated to the constructor which gives us an empty tree, and now we
  // can serialize from it.
  ar >> BOOST_SERIALIZATION_NVP(*this);
}

} // namespace tree
} // namespace mlpack

#endif
