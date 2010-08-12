/** @file clusterwise_regression.cc
 *
 *  @brief The main driver for the clusterwise regression using the
 *         mixture of experts.
 *
 *  @author Dongryeol Lee
 */

#include <vector>
#include <string>
#include "clusterwise_regression_dev.h"

int main(int argc, char *argv[]) {

  srand(time(NULL));

  // Convert C input to C++; skip executable name for Boost
  std::vector<std::string> args(argv + 1, argv + argc);
  try {
    return fl::ml::ClusterwiseRegression::Main(args);
  }
  catch (const std::exception &exception) {
    return EXIT_FAILURE;
  }
}
