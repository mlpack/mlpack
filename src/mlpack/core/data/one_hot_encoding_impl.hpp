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
  std::unordered_map<size_t, size_t> mapIndices;
  size_t newRows = input.n_rows - indices.n_elem;
  for (size_t i = 0; i < indices.n_elem; i++)
  {
    mapIndices[indices.at(i)] = numofMapping(input.row(indices.at(i)));
    newRows += mapIndices[indices.at(i)];
  }

  // find the size of output mat 
  output.zeros(newRows, input.n_cols);
  size_t row = 0;

  for (size_t i = 0; i < input.n_rows; i++)
  {
    if (mapIndices.count(i) == 0)
    {
      // Copy exactly as required
      for (size_t j = 0; j < input.n_cols; j++)
      {
        output(row, j) = input(i, j);
      }
      row++;
    }
    else
    {
      arma::Row<size_t> labels;
      labels.set_size(input.n_cols);

      // Loop over the input labels, and develop the mapping.
      // Map for labelsIn to labels.
      std::unordered_map<eT, size_t> labelMap;
      size_t curLabel = 0;
      for (size_t k = 0; k < input.n_cols; ++k)
      {
        // If labelsIn[i] is already in the map, use the existing label.
        if (labelMap.count(input(i, k)) != 0)
        {
          labels[k] = labelMap[input(i, k)] - 1;
        }
        else
        {
          // If labelsIn[i] not there then add it to the map.
          labelMap[input(i, k)] = curLabel + 1;
          labels[k] = curLabel;
          ++curLabel;
        }
      }
      // Fill ones in at the required places.
      for (size_t l = 0; l < input.n_cols; ++l)
      {
        output(row + labels[l], l) = 1;
      }
      labelMap.clear();
      row +=  mapIndices[i];
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
 * @param datasetInfo DatasetInfod object that has information about data.
 */
template<typename eT>
void OneHotEncoding(const arma::Mat<eT>& input,
                    arma::Mat<eT>& output,
                    const data::DatasetInfo& datasetInfo)
{
  std::vector<size_t>indices;
  for (size_t i = 0; i < datasetInfo.Dimensionality(); ++i)
  {
    if (datasetInfo.Type(i) == data::Datatype::categorical)
    {
      indices.push_back(i);
    }
  }
  OneHotEncoding(input, (arma::Col<size_t>)(indices), output);
}

/**
 * Returns the count of unique categorical values in a row.
 * 
 * @param labelsIn Input labels of arbitrary datatype.
 */
template<typename RowType>
size_t numofMapping(const RowType& labelsIn)
{
  // Loop over the input labels, and develop the mapping.
  size_t curLabel = 0;
  std::unordered_map<typename RowType::elem_type, size_t> labelMap;
  for (size_t i = 0; i < labelsIn.n_elem; ++i)
  {
    // If labelsIn[i] is already in the map, use the existing label.
    if (labelMap.count(labelsIn[i]) != 0)
    {
      continue;
    }
    else
    {
      // If labelsIn[i] not there then add it to the map.
      labelMap[labelsIn[i]] = curLabel + 1;
      ++curLabel;
    }
  }
  return labelMap.size();
}

} // namespace data
} // namespace mlpack

#endif
