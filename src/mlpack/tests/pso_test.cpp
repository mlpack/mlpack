/**
 * @file pso_test.cpp
 *
 * Test file for PSO.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/pso/pso.hpp>

using namespace arma;
using namespace mlpack::optimization;

int main() {
    PSO<> optimizer();
}