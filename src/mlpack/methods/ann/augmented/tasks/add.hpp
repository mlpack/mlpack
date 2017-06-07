/**
 * @file add.hpp
 * @author Konstantin Sidorov
 *
 * Definition of the AddTask class, which implements a generator of
 * instances of sequence addition task.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_AUGMENTED_TASKS_ADD_HPP
#define MLPACK_METHODS_AUGMENTED_TASKS_ADD_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
namespace tasks /* Task utilities for augmented */ {
class AddTask
{
public:
  AddTask(size_t bitLen);

  void GenerateData(arma::field<arma::irowvec>& input,
                    arma::field<arma::irowvec>& labels,
                    size_t batchSize);
private:
  size_t bitLen;
};
} // namespace tasks 
} // namespace augmented
} // namespace ann
} // namespace mlpack 

#include "add_impl.hpp"
#endif



