/**
 * @file methods/linear_svm/linear_svm_model.hpp
 * @author Yashwant Singh Parihar
 * @author Dirk Eddelbuettel
 *
 * A serializable decision tree model, used by the decision tree binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_MODEL_HPP
#define MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_MODEL_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * This is the class that we will serialize.  It is a pretty simple wrapper
 * around LinearSVM<>.
 */
class LinearSVMModel
{
 public:
  arma::Col<size_t> mappings;
  LinearSVM<> svm;

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(mappings));
    ar(CEREAL_NVP(svm));
  }
};

} // namespace mlpack

#endif
