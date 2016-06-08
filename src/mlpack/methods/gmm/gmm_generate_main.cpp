/**
 * @file gmm_generate_main.cpp
 * @author Ryan Curtin
 *
 * Load a GMM from file, then generate samples from it.
 */
#include <mlpack/core.hpp>
#include "gmm.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::gmm;

PROGRAM_INFO("GMM Sample Generator",
    "This program is able to generate samples from a pre-trained GMM (use "
    "gmm_train to train a GMM).  It loads a GMM from the file specified with "
    "--input_model_file (-m), and generates a number of samples from that "
    "model; the number of samples is specified by the --samples (-n) parameter."
    "The output samples are saved in the file specified by --output_file "
    "(-o).");

PARAM_STRING_REQ("input_model_file", "File containing input GMM model.", "m");
PARAM_INT_REQ("samples", "Number of samples to generate.", "n");

PARAM_STRING("output_file", "File to save output samples in.", "o", "");
PARAM_INT("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  if (CLI::HasParam("output_file"))
    Log::Warn << "--output_file (-o) is not specified;"
      << "no results will be saved!" << endl;

  if (CLI::GetParam<int>("seed") == 0)
    mlpack::math::RandomSeed(time(NULL));
  else
    mlpack::math::RandomSeed((size_t) CLI::GetParam<int>("seed"));

  if (CLI::GetParam<int>("samples") < 0)
    Log::Fatal << "Parameter to --samples must be greater than 0!" << endl;

  GMM gmm;
  data::Load(CLI::GetParam<string>("input_model_file"), "gmm", gmm, true);

  size_t length = (size_t) CLI::GetParam<int>("samples");
  Log::Info << "Generating " << length << " samples..." << endl;
  arma::mat samples(gmm.Dimensionality(), length);
  for (size_t i = 0; i < length; ++i)
    samples.col(i) = gmm.Random();

  if(CLI::HasParam("output_file"))
    data::Save(CLI::GetParam<string>("output_file"), samples);
}
