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
  size_t k = 5;
  double sampleSize = 0.25;
  double recall, selectivity;
  arma::mat rdata(d, N, arma::fill::randu);
  LSHModel<> model(rdata, sampleSize, k);
  model.Predict(N, k, 16, 4, 4, 1.0, recall, selectivity);

  Log::Info << "Model predicts " << recall*100 << "\% recall and "
    << selectivity*100 << "\% selectivity." << std::endl;

  arma::mat qdata(d, 1, arma::fill::randu);
  arma::Mat<size_t> neighbors;
  arma::mat distances;
  LSHSearch<> lsh = model.LSHObject(1, 1, 1.0, 99901, 500);
  lsh.Search(qdata, 1, neighbors, distances);

  return 0;
}
