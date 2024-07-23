/**
 * @file tests/main.cpp
 *
 * Main file for Catch testing.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

// #define CATCH_CONFIG_MAIN  // catch.hpp will define main()
#define CATCH_CONFIG_RUNNER  // we will define main()
#include "catch.hpp"

int main(int argc, char** argv)
{
  Catch::Session session;
  const int returnCode = session.applyCommandLine(argc, argv);
  // Check for a command line error.
  if (returnCode != 0)
    return returnCode;

  std::cout << "mlpack version: " << mlpack::util::GetVersion() << std::endl;
  std::cout << "armadillo version: " << arma::arma_version::as_string()
      << std::endl;

  // Use Catch2 command-line to set the random seed.
  // -rng-seed <'time'|number>
  // If a number is provided this is used directly as the seed. Alternatively
  // if the keyword 'time' is provided then the result of calling std::time(0)
  // is used.
  const size_t seed = session.config().rngSeed();
  std::cout << "random seed: " << seed << std::endl;
  mlpack::RandomSeed(seed);

  return session.run();
}
