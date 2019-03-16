/**
 * @file normalize_labels_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of label normalization functions; these are useful for mapping
 * labels to the range [0, n).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_NORMALIZE_LABELS_IMPL_HPP
#define MLPACK_CORE_DATA_NORMALIZE_LABELS_IMPL_HPP

// In case it hasn't been included yet.
#include "normalize_labels.hpp"

namespace mlpack {
namespace data {

/**
 * Given a set of labels of a particular datatype, convert them to unsigned
 * labels in the range [0, n) where n is the number of different labels.  Also,
 * a reverse mapping from the new label to the old value is stored in the
 * 'mapping' vector.
 *
 * @param labelsIn Input labels of arbitrary datatype.
 * @param labels Vector that unsigned labels will be stored in.
 * @param mapping Reverse mapping to convert new labels back to old labels.
 */
template<typename eT, typename RowType>
void NormalizeLabels(const RowType& labelsIn,
                     arma::Row<size_t>& labels,
                     arma::Col<eT>& mapping)
{
  // Loop over the input labels, and develop the mapping.  We'll first naively
  // resize the mapping to the maximum possible size, and then when we fill it,
  // we'll resize it back down to its actual size.
  mapping.set_size(labelsIn.n_elem);
  labels.set_size(labelsIn.n_elem);
  size_t curLabel = 0;
  for (size_t i = 0; i < labelsIn.n_elem; ++i)
  {
    bool found = false;
    for (size_t j = 0; j < curLabel; ++j)
    {
      // Is the label already in the list of labels we have seen?
      if (labelsIn[i] == mapping[j])
      {
        labels[i] = j;
        found = true;
        break;
      }
    }

    // Do we need to add this new label?
    if (!found)
    {
      mapping[curLabel] = labelsIn[i];
      labels[i] = curLabel;
      ++curLabel;
    }
  }

  // Resize mapping back down to necessary size.
  mapping.resize(curLabel);
}

/**
 * Given a set of labels that have been mapped to the range [0, n), map them
 * back to the original labels given by the 'mapping' vector.
 *
 * @param labels Set of normalized labels to convert.
 * @param mapping Mapping to use to convert labels.
 * @param labelsOut Vector to store new labels in.
 */
template<typename eT>
void RevertLabels(const arma::Row<size_t>& labels,
                  const arma::Col<eT>& mapping,
                  arma::Row<eT>& labelsOut)
{
  // We already have the mapping, so we just need to loop over each element.
  labelsOut.set_size(labels.n_elem);

  for (size_t i = 0; i < labels.n_elem; ++i)
    labelsOut[i] = mapping[labels[i]];
}

} // namespace data
} // namespace mlpack

#endif
