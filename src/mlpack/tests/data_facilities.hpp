/**
 * @file data_facilities.hpp
 * @author Ryan Curtin, Kirill Mishchenko
 *
 * This file declares some useful data facilities.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_TESTS_DATA_FACILITIES_HPP
#define MLPACK_TESTS_DATA_FACILITIES_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * Create a mock categorical dataset for testing.
 */
void MockCategoricalData(arma::mat& d,
                         arma::Row<size_t>& l,
                         data::DatasetInfo& datasetInfo);

} // namespace mlpack

#endif
