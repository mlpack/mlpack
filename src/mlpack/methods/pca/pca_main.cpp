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
PARAM_STRING_REQ("input_file", "Input dataset to perform PCA on.", "");
PARAM_STRING_REQ("output_file", "Output dataset to perform PCA on.", "");
PARAM_INT("new_dimensionality", "Desired dimensionality of output dataset.",
    "", 0);

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
    if (newDimension < 1);
    {
      Log::Fatal << "Invalid value for new dimensionality (" << newDimension
          << ")!  Must be greater than or equal to 1." << std::endl;
    }
  }

  // Perform PCA.
  PCA p;
  Log::Info << "Performing PCA on dataset..." << std::endl;
  p.Apply(dataset, newDimension);

  // Now save the results.
  string outputFile = CLI::GetParam<string>("output_file");
  data::Save(outputFile.c_str(), dataset);
}
