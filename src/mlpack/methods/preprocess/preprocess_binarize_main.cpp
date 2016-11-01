/**
 * @file preprocess_binarize_main.cpp
 * @author Keon Kim
 *
 * binarize CLI executable
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/binarize.hpp>

PROGRAM_INFO("Binarize Data", "This utility takes a dataset and binarizes the "
    "variables into either 0 or 1 given threshold. User can apply binarization "
    "on a dimension or the whole dataset. A dimension can be specified using "
    "--dimension (-d) option. Threshold can also be specified with the "
    "--threshold (-t) option; The default is 0.0."
    "\n\n"
    "The program does not modify the original file, but instead makes a "
    "separate file to save the binarized data; The program requires you to "
    "specify the file name with --output_file (-o)."
    "\n\n"
    "For example, if we want to make all variables greater than 5 in dataset "
    "to 1 and ones that are less than or equal to 5.0 to 0, and save the "
    "result to result.csv, we could run"
    "\n\n"
    "$ mlpack_preprocess_binarize -i dataset.csv -t 5 -o result.csv"
    "\n\n"
    "But if we want to apply this to only the first (0th) dimension of the "
    "dataset, we could run"
    "\n\n"
    "$ mlpack_preprocess_binarize -i dataset.csv -t 5 -d 0 -o result.csv");

// Define parameters for data.
PARAM_MATRIX_IN_REQ("input", "Input data matrix.", "i");
// Define optional parameters.
PARAM_MATRIX_OUT("output", "Matrix in which to save the output.", "o");
PARAM_INT_IN("dimension", "Dimension to apply the binarization. If not set, the"
    " program will binarize every dimension by default.", "d", 0);
PARAM_DOUBLE_IN("threshold", "Threshold to be applied for binarization. If not "
    "set, the threshold defaults to 0.0.", "t", 0.0);

using namespace mlpack;
using namespace arma;
using namespace std;

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);
  const size_t dimension = (size_t) CLI::GetParam<int>("dimension");
  const double threshold = CLI::GetParam<double>("threshold");

  // Check on data parameters.
  if (!CLI::HasParam("dimension"))
    Log::Warn << "You did not specify --dimension, so the program will perform "
        << "binarize on every dimension." << endl;

  if (!CLI::HasParam("threshold"))
    Log::Warn << "You did not specify --threshold, so the threshold will be "
        << "automatically set to '0.0'." << endl;

  if (!CLI::HasParam("output"))
    Log::Warn << "You did not specify --output_file, so no result will be "
        << "saved." << endl;

  // Load the data.
  arma::mat input = std::move(CLI::GetParam<arma::mat>("input"));
  arma::mat output;

  Timer::Start("binarize");
  if (CLI::HasParam("dimension"))
  {
    data::Binarize<double>(input, output, threshold, dimension);
  }
  else
  {
    // binarize the whole data
    data::Binarize<double>(input, output, threshold);
  }
  Timer::Stop("binarize");

  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = std::move(output);
}
