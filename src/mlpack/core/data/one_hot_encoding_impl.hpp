/**
 * @file one_hot_encoding_impl.hpp
 * @author Jeffin Sam
 *
 * Implementation of one hot encoding functions; categorical variables as binary
 * vectors.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_ONE_HOT_ENCODING_IMPL_HPP
#define MLPACK_CORE_DATA_ONE_HOT_ENCODING_IMPL_HPP

// In case it hasn't been included yet.
#include "one_hot_encoding.hpp"


namespace mlpack {
namespace data {

/**
 * Given a set of labels of a particular datatype, convert them to binary
 * vector. The categorical values be mapped to integer values.
 * Then, each integer value is represented as a binary vector that is
 * all zero values except the index of the integer, which is marked
 * with a 1.
 *
 * @param labelsIn Input labels of arbitrary datatype.
 * @param output Binary matrix.
 */
template<typename eT, typename RowType>
void OneHotEncoding(const RowType& labelsIn,
                    arma::Mat<eT>& output)
{
  RowType labels = arma::unique(labelsIn);

  // Resize output matrix to necessary size.
  // Fill it with zero
  output.zeros(labelsIn.n_elem, labels.n_elem);

  // Filling one at required place
  for (size_t i = 0; i < labelsIn.n_elem; ++i)
    output(i, std::lower_bound(labels.begin(), labels.end(),
        labelsIn[i]) - labels.begin()) = 1;

}

} // namespace data
} // namespace mlpack

#endif
