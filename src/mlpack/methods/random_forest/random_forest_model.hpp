/**
 * @file methods/random_forest/random_forest_model.hpp
 * @author Ryan Curtin
 * @author Dirk Eddelbuettel
 *
 * A serializable Random Forest model, used by the random_forest binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_MODEL_HPP
#define MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_MODEL_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * This is the class that we will serialize.  It is a pretty simple wrapper
 * around DecisionTree<>.  In order to support categoricals, it will need to
 * also hold and serialize a DatasetInfo.
 */
class RandomForestModel
{
 public:
  // The tree itself, left public for direct access by this binding.
  RandomForest<> rf;

  // Create the model.
  RandomForestModel() { /* Nothing to do. */ }

  // Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(rf));
  }
};

} // namespace mlpack

#endif
