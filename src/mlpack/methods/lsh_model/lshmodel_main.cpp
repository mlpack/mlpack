#include <mlpack/core.hpp>
#include <mlpack/methods/lsh/lsh_search.hpp>

#include "lshmodel.hpp"

using namespace mlpack;
using namespace mlpack::neighbor;

PROGRAM_INFO("LSH Model (TODO: Complete this)", "");

PARAM_STRING_IN("reference_file", "File containing the dataset", "r", "");
PARAM_STRING_OUT("output_model_file", "File to save trained LSH model to", "m");

int main(int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);

  // Generate a random point set.
  size_t N = 5000;
  size_t d = 10;
  arma::mat A(d, N, arma::fill::randu);
  LSHModel<> model(A, 0.7, 0.25, 2);

  return 0;
}
