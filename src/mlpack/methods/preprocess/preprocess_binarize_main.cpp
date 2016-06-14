/**
 * @file preprocess_binarize_main.cpp
 * @author Keon Kim
 *
 * split data CLI executable
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/binarize.hpp>

PROGRAM_INFO("Split Data", "This utility takes a dataset and optionally labels "
    "and splits ");

// Define parameters for data.
PARAM_STRING_REQ("input_file", "File containing data,", "i");
// Define optional parameters.
PARAM_STRING("output_file", "File to save the output,", "o", "");
PARAM_INT("feature", "File containing labels", "f", 0);
PARAM_DOUBLE("threshold", "Ratio of test set, if not set,"
    "the threshold defaults to 0.0", "t", 0.0);

using namespace mlpack;
using namespace arma;
using namespace std;

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);
  const string inputFile = CLI::GetParam<string>("input_file");
  const string outputFile = CLI::GetParam<string>("output_file");
  const size_t feature = (size_t) CLI::GetParam<int>("feature");
  const double threshold = CLI::GetParam<double>("threshold");

  // Check on data parameters.
  if (!CLI::HasParam("feature"))
    Log::Warn << "You did not specify --feature, so the program will perform "
              << "binarize on every features." << endl;

  if (!CLI::HasParam("threshold"))
    Log::Warn << "You did not specify --threshold, so the threhold "
              << "will be automatically set to '0.0'." << endl;

  if (!CLI::HasParam("output_file"))
    Log::Warn << "You did not specify --output_file, so no result will be"
              << "saved." << endl;

  // Load the data.
  arma::mat input;
  arma::mat output;
  data::Load(inputFile, input, true);

  Timer::Start("binarize");
  if (CLI::HasParam("feature"))
  {
    data::Binarize<double>(input, output, threshold, feature);
  }
  else
  {
    // binarize the whole data
    data::Binarize<double>(input, output, threshold);
  }
  Timer::Stop("binarize");

  Log::Info << "input" << endl;
  Log::Info << input << endl;
  Log::Info << "output" << endl;
  Log::Info << output << endl;

  if (CLI::HasParam("output_file"))
    data::Save(outputFile, output, false);
}
