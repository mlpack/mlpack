/**
 * @file core/data/one_hot_encoding_impl.hpp
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
template<typename RowType, typename MatType>
void OneHotEncoding(const RowType& labelsIn,
                    MatType& output)
{
  arma::Row<size_t> labels;
  labels.set_size(labelsIn.n_elem);

  // Loop over the input labels, and develop the mapping.
  // Map for labelsIn to labels.
  std::unordered_map<typename MatType::elem_type, size_t> labelMap;
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
 * Indices represent the IDs of the dimensions to be one-hot encoded.
 *
 * @param input Input dataset to be encoded.
 * @param indices Index of rows to be encoded.
 * @param output Encoded matrix.
 */
template<typename eT>
void OneHotEncoding(const arma::Mat<eT>& input,
                    const arma::Col<size_t>& indices,
                    arma::Mat<eT>& output)
{
  // Handle the edge case where there is nothing to encode.
  if (indices.n_elem == 0)
  {
    output = input;
    return;
  }

  // First, we need to compute the size of the output matrix.

  // This vector will eventually hold the offsets for each dimension in the
  // one-hot encoded matrix, but first it will just hold the counts of
  // dimensions for each dimension.
  arma::Col<size_t> dimensionOffsets(input.n_rows, arma::fill::ones);
  // This will hold the mappings from a value that should be one-hot encoded to
  // the index of the dimension it should take.
  std::unordered_map<size_t, std::unordered_map<eT, size_t>> mappings;
  for (size_t i = 0; i < indices.n_elem; ++i)
  {
    dimensionOffsets[indices[i]] = 0;
    mappings.insert(
        std::make_pair(indices[i], std::unordered_map<eT, size_t>()));
  }

  for (size_t col = 0; col < input.n_cols; ++col)
  {
    for (size_t row = 0; row < input.n_rows; ++row)
    {
      if (mappings.count(row) != 0)
      {
        // We have to one-hot encode this point.
        if (mappings[row].count(input(row, col)) == 0)
          mappings[row][input(row, col)] = dimensionOffsets[row]++;
      }
    }
  }

  // Turn the dimension counts into offsets.  Note that the last element is the
  // total number of dimensions, and the first element is the offset for
  // dimension *2* (not 1).
  for (size_t i = 1; i < dimensionOffsets.n_elem; ++i)
    dimensionOffsets[i] += dimensionOffsets[i - 1];

  // Now, initialize the output matrix to the right size.
  output.zeros(dimensionOffsets[dimensionOffsets.n_elem - 1], input.n_cols);

  // Finally, one-hot encode the matrix.
  for (size_t col = 0; col < input.n_cols; ++col)
  {
    for (size_t row = 0; row < input.n_rows; ++row)
    {
      const size_t dimOffset = (row == 0) ? 0 : dimensionOffsets[row - 1];
      if (mappings.count(row) != 0)
      {
        output(dimOffset + mappings[row][input(row, col)], col) = eT(1);
      }
      else
      {
        // No need for one-hot encoding.
        output(dimOffset, col) = input(row, col);
      }
    }
  }
}

/**
 * Overloaded function for the above function, which takes a matrix as input
 * and also a DatasetInfo object and outputs a matrix.
 * This function encodes all the dimensions marked `Datatype::categorical`
 * in the data::DatasetInfo.
 *
 * @param input Input dataset to be encoded.
 * @param output Encoded matrix.
 * @param datasetInfo DatasetInfo object that has information about data.
 */
template<typename eT>
void OneHotEncoding(const arma::Mat<eT>& input,
                    arma::Mat<eT>& output,
                    const data::DatasetInfo& datasetInfo)
{
  std::vector<size_t> indices;
  for (size_t i = 0; i < datasetInfo.Dimensionality(); ++i)
  {
    if (datasetInfo.Type(i) == data::Datatype::categorical)
    {
      indices.push_back(i);
    }
  }
  OneHotEncoding(input, arma::Col<size_t>(indices), output);
}

} // namespace data
} // namespace mlpack

#endif
