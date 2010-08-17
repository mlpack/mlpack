/** @file gp_regression.cc
 *
 *  @brief The main driver for the Gaussian process regression.
 *
 *  @author Dongryeol Lee
 */

#include <vector>
#include <string>
#include "sg_gp_regression_dev.h"

int main(int argc, char *argv[]) {

  srand(time(NULL));

  // Convert C input to C++; skip executable name for Boost
  std::vector<std::string> args(argv + 1, argv + argc);
  try {
    return ml::SparseGreedyGpr::Main(args);
  }
  catch (const std::exception &exception) {
    return EXIT_FAILURE;
  }
}
