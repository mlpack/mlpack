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
template<class RowType, template <typename> class MatType, class eT>
void OneHotEncoding(const RowType& labelsIn,
                    MatType<eT>& output)
{
  arma::Row<size_t> labels;
  labels.set_size(labelsIn.n_elem);

  // Loop over the input labels, and develop the mapping.
  std::unordered_map<eT, size_t> labelMap; // Map for labelsIn to labels.
  size_t curLabel = 0;
  for (size_t i = 0; i < labelsIn.n_elem; ++i)
  {
    // If labelsIn[i] is already in the map, use the existing label.
    if (labelMap.count(labelsIn[i]) != 0)
    {
      labels[i] = labelMap[labelsIn[i]] - 1;
    }
    else
    {
      // If labelsIn[i] not there then add it to the map.
      labelMap[labelsIn[i]] = curLabel + 1;
      labels[i] = curLabel;
      ++curLabel;
    }
  }
  // Resize output matrix to necessary size, and fill it with zeros.
  output.zeros(curLabel, labelsIn.n_elem);
  // Fill ones in at the required places.
  for (size_t i = 0; i < labelsIn.n_elem; ++i)
  {
    output(labels[i], i) = 1;
  }
  labelMap.clear();
}

/**
 * Overloaded function for the above function, which takes a matrix as input
 * and also a vector of indices to encode and outputs a matrix.
 *
 * @param input Input dataset to be encoded.
 * @param indices Index of rows to be encoded.
 * @param output Encoded matrix.
 */
template<typename eT>
void OneHotEncoding(arma::Mat<eT>& input,
                    const arma::ucolvec indices,
                    arma::Mat<eT>& output)
{
  output = input;
  for (size_t i = 0; i < indices.n_elem; i++)
    output.shed_rows(indices.at(i));

  std::vector<arma::Mat<eT>> oheOutput(indices.n_elem);
  for (size_t i = 0; i < indices.n_elem; i++)
  {
    // call OneHotEncoding() for each of the indices.
    OneHotEncoding(input.row(indices.at(i)), oheOutput[i]);
  }
  size_t row = 0;
  for (size_t i = 0; i < indices.n_elem; i++)
  {
    // calculating index at which the rows shoudl be inserted.
    output.insert_rows(row + indices.at(i), oheOutput[i]);
    row += oheOutput[i].n_rows - 1;
  }
}

} // namespace data
} // namespace mlpack

#endif
