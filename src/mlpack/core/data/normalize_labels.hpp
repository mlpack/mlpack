/**
 * @file normalize_labels.hpp
 * @author Ryan Curtin
 *
 * Often labels are not given as {0, 1, 2, ...} but instead {1, 2, ...} or even
 * {-1, 1} or otherwise.  The purpose of this function is to normalize labels to
 * {0, 1, 2, ...} and provide a mapping back to those labels.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_DATA_NORMALIZE_LABELS_HPP
#define __MLPACK_CORE_DATA_NORMALIZE_LABELS_HPP

#include <mlpack/prereqs.hpp>

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
template<typename eT>
void NormalizeLabels(const arma::Col<eT>& labelsIn,
                     arma::Col<size_t>& labels,
                     arma::Col<eT>& mapping);

/**
 * Given a set of labels that have been mapped to the range [0, n), map them
 * back to the original labels given by the 'mapping' vector.
 *
 * @param labels Set of normalized labels to convert.
 * @param mapping Mapping to use to convert labels.
 * @param labelsOut Vector to store new labels in.
 */
template<typename eT>
void RevertLabels(const arma::Col<size_t>& labels,
                  const arma::Col<eT>& mapping,
                  arma::Col<eT>& labelsOut);

}; // namespace data
}; // namespace mlpack

// Include implementation.
#include "normalize_labels_impl.hpp"

#endif
