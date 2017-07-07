/**
 * @file test_function.cpp
 * @author Kartik Nighania (Mentor Marcus Edel)
 *
 * Implementation of very simple test for 
 * COVARIANCE MATRIX ADAPTATION EVOLUTION STRATEGY
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "test_function.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

size_t cmaesTestFunction::NumFunctions(){ return 3; }

double camesTestFunction::Evaluate(arma::mat& coordinates)
{
 	return -std::exp(-std::abs(coordinates[0])) + 
            std::pow(coordinates[1], 2) + 
            std::pow(coordinates[2], 4) + 3 * std::pow(coordinates[2], 2);
}
