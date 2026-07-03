/**
 * @file methods/decision_tree/decision_tree_model.hpp
 * @author Ryan Curtin
 * @author Dirk Eddelbuettel
 *
 * A serializable decision tree model, used by the decision tree binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_DECISION_TREE_MODEL_HPP
#define MLPACK_METHODS_DECISION_TREE_DECISION_TREE_MODEL_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * This is the class that we will serialize.  It is a pretty simple wrapper
 * around DecisionTree<>.
 */
class DecisionTreeModel
{
 public:
  // The tree itself, left public for direct access by this program.
  DecisionTree<> tree;
  DatasetInfo info;

  // Create the model.
  DecisionTreeModel() { /* Nothing to do. */ }

  // Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(tree));
    ar(CEREAL_NVP(info));
  }
};

} // namespace mlpack

#endif
