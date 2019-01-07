/**
 * @file sparse_svm.hpp
 * @author Ayush Chamoli
 *
 * An implementation of softmax regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_HPP
#define MLPACK_METHODS_SPARSE_SVM_SPARSE_SVM_HPP

#include <mlpack/prereqs.hpp>
#include <ensmallen.hpp>

#include "sparse_svm_function.hpp"

namespace mlpack{
namespace regression{

class SparseSVM
{
 public:
  template<typename OptimizerType = ens::L_BFGS>
  SparseSVM(const arma::mat& data,
            const arma::Row<size_t>& labels);


};

} // namespace regression
} // namespace mlpack

#include "sparse_svm.cpp"

#endif
