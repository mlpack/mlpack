/**
 * @file pca_main.cpp
 * @author Ryan Curtin
 *
 * Main executable to run PCA.
 */
#include <mlpack/core.hpp>

#include "pca.hpp"

using namespace mlpack;
using namespace mlpack::pca;
using namespace std;

// Document program.
PROGRAM_INFO("Principal Components Analysis", "This program performs principal "
    "components analysis on the given dataset.  It will transform the data "
    "onto its principal components, optionally performing dimensionality "
    "reduction by ignoring the principal components with the smallest "
    "eigenvalues.");

// Parameters for program.
PARAM_STRING_REQ("input_file", "Input dataset to perform PCA on.", "i");
PARAM_STRING_REQ("output_file", "File to save modified dataset to.", "o");
PARAM_INT("new_dimensionality", "Desired dimensionality of output dataset.  If "
    "0, no dimensionality reduction is performed.", "d", 0);

PARAM_FLAG("scale", "If set, the data will be scaled before running PCA, such "
    "that the variance of each feature is 1.", "s");
PARAM_FLAG("nocenter", "If set, the data will NOT be centered before performing"
    " PCA.", "N");

int main(int argc, char** argv)
{
  // Parse commandline.
  CLI::ParseCommandLine(argc, argv);

  // Load input dataset.
  string inputFile = CLI::GetParam<string>("input_file");
  arma::mat dataset;
  data::Load(inputFile.c_str(), dataset);

  // Find out what dimension we want.
  size_t newDimension = dataset.n_rows; // No reduction, by default.
  if (CLI::GetParam<int>("new_dimensionality") != 0)
  {
    // Validate the parameter.
    newDimension = (size_t) CLI::GetParam<int>("new_dimensionality");
    if (newDimension > dataset.n_rows)
    {
      Log::Fatal << "New dimensionality (" << newDimension
          << ") cannot be greater than existing dimensionality ("
          << dataset.n_rows << ")!" << endl;
    }
  }

  // Get the options for running PCA.
  const size_t scale = CLI::HasParam("scale");
  const size_t center = !CLI::HasParam("nocenter");

  // Perform PCA.
  PCA p(center, scale);
  Log::Info << "Performing PCA on dataset..." << endl;
  p.Apply(dataset, newDimension);

  // Now save the results.
  string outputFile = CLI::GetParam<string>("output_file");
  data::Save(outputFile, dataset);
}
