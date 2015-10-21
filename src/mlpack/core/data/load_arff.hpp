/**
 * @file load_arff.hpp
 * @author Ryan Curtin
 *
 * Load an ARFF dataset.
 */
#ifndef __MLPACK_CORE_DATA_LOAD_ARFF_HPP
#define __MLPACK_CORE_DATA_LOAD_ARFF_HPP

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
 */
template<typename eT>
void LoadARFF(const std::string& filename,
              arma::Mat<eT>& matrix,
              DatasetInfo& info);

} // namespace data
} // namespace mlpack

// Include implementation.
#include "load_arff_impl.hpp"

#endif
