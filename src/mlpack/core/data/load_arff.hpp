/**
 * @file load_arff.hpp
 * @author Ryan Curtin
 *
 * Load an ARFF dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_LOAD_ARFF_HPP
#define MLPACK_CORE_DATA_LOAD_ARFF_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace data {

/**
 * A utility function to load an ARFF dataset as numeric features (that is, as
 * an Armadillo matrix without any modification).  An exception will be thrown
 * if any features are non-numeric.
 */
template<typename eT>
void LoadARFF(const std::string& filename, arma::Mat<eT>& matrix);

/**
 * A utility function to load an ARFF dataset as numeric and categorical
 * features, using the DatasetInfo structure for mapping.  An exception will be
 * thrown upon failure.
 *
 * A pre-existing DatasetInfo object can be passed in, but if the dimensionality
 * of the given DatasetInfo object (info.Dimensionality()) does not match the
 * dimensionality of the data, a std::invalid_argument exception will be thrown.
 * If an empty DatasetInfo object is given (constructed with the default
 * constructor or otherwise, so that info.Dimensionality() is 0), it will be set
 * to the right dimensionality.
 *
 * This ability to pass in pre-existing DatasetInfo objects is very necessary
 * when, e.g., loading a test set after training.  If the same DatasetInfo from
 * loading the training set is not used, then the test set may be loaded with
 * different mappings---which can cause horrible problems!
 *
 * @param filename Name of ARFF file to load.
 * @param matrix Matrix to load data into.
 * @param info DatasetInfo object; can be default-constructed or pre-existing
 *     from another call to LoadARFF().
 */
template<typename eT, typename PolicyType>
void LoadARFF(const std::string& filename,
              arma::Mat<eT>& matrix,
              DatasetMapper<PolicyType>& info);

} // namespace data
} // namespace mlpack

// Include implementation.
#include "load_arff_impl.hpp"

#endif
