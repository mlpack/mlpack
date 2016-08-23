#include <mlpack/core.hpp>
#include <mlpack/methods/lsh/lsh_search.hpp>

#include "lshmodel.hpp"

using std::string; using std::endl; using std::cout;
using namespace mlpack;
using namespace mlpack::neighbor;

PROGRAM_INFO("LSH Modeling and Tuning", 
    "This program can help tune parameters for the LSH algorithm for"
    " approximate nearest neighbor search. Currently, the only option is to"
    " specify a number of the four parameters (numTables, numProj, numProbes,"
    " hashWidth) and receive an estimate of LSH's recall and selectivity for a"
    " given dataset."
    );

PARAM_STRING_IN("reference_file", "File containing the dataset", "r", "");
PARAM_DOUBLE_IN("sample_percentage", "Sample size percentage. Must be in (0, 1]", "p", 0.0) 

PARAM_INT_IN("neighbors", "The number of nearest neighbors LSH will search for", "k", 1);
PARAM_INT_IN("tables", "The number of tables for LSH", "L", 30);
PARAM_INT_IN("projections", "The number of projections per table for LSH", "K", 10);
PARAM_INT_IN("probes", "The number of probes for multiprobe LSH", "T", 0);
PARAM_DOUBLE_IN("hash_width", "The hash width for the first level hashing", "H", 1.0);

//PARAM_STRING_OUT("output_model_file", "File to save trained LSH model to", "m");

int main(int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);

  // If no input file was specified, die here.
  if (!CLI::HasParam("reference_file"))
    Log::Fatal << "You need to specify the reference file." << endl;
  // Read input file name
  string rfile = CLI::GetParam<string>("reference_file");
  // Attempt to read file.
  arma::mat rdata;
  data::Load(rfile, rdata, true); // true: if you can't open file, die.
  size_t N = rdata.n_cols; // Dataset size.

  // Parse rest of command line input.
  size_t k = CLI::GetParam<int>("neighbors");
  size_t numTables = CLI::GetParam<int>("tables");
  size_t numProj = CLI::GetParam<int>("projections");
  size_t numProbes = CLI::GetParam<int>("probes");
  double hashWidth = CLI::GetParam<double>("hash_width");
  double sampleSize = CLI::GetParam<double>("sample_percentage");
  if (sampleSize == 0.0)
    Log::Fatal << "You need to specify the sampling percentage." << endl;

  Log::Info <<
    "Tuning LSH for" << std::endl
    <<"\t numTables = " << numTables << std::endl
    <<"\t numProj = " << numProj << std::endl
    <<"\t numProbes = " << numProbes << std::endl
    <<"\t hashWidth = " << hashWidth << std::endl;

  double recall, selectivity;

  LSHModel<> model(rdata, sampleSize, k);
  model.Predict(N, k, numTables, numProj, numProbes, hashWidth, recall, selectivity);

  cout << "Model predicts " << recall*100 << "\% recall and "
    << selectivity*100 << "\% selectivity." << endl;

  return 0;
}
