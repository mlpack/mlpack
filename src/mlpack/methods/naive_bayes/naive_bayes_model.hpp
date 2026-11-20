/**
 * @file mmethods/naive_bayes/naive_bayes_model.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @author Dirk Eddelbuettel
 *
 * A serializable Naive Bayes model, used by the Naive Bayes Classifier.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_MODEL_HPP
#define MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_MODEL_HPP

#include <mlpack/core.hpp>

namespace mlpack {

// A struct for saving the model with mappings.
struct NBCModel
{
  //! The model itself.
  NaiveBayesClassifier<> nbc;
  //! The mappings for labels.
  arma::Col<size_t> mappings;

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(nbc));
    ar(CEREAL_NVP(mappings));
  }
};

} // namespace mlpack

#endif
