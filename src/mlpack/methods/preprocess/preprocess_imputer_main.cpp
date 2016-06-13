/**
 * @file preprocess_imputer_main.cpp
 * @author Keon Kim
 *
 * a utility that provides imputation strategies fore
 * missing values.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/imputer.hpp>
#include <mlpack/core/data/dataset_info.hpp>
#include <mlpack/core/data/map_policies/increment_map_policy.hpp>
#include <mlpack/core/data/impute_strategies/mean_strategy.hpp>

PROGRAM_INFO("Imputer", "This "
    "utility takes an any type of data and provides "
    "imputation strategies for missing data.");

PARAM_STRING_REQ("input_file", "File containing data,", "i");
PARAM_STRING("output_file", "File to save output", "o", "");
PARAM_STRING("missing_value", "User defined missing value", "m", "")
PARAM_STRING("map_policy", "mapping policy to be used while loading", "p", "")
PARAM_STRING("map_to", "custom_strategy option. map to something else", "t", "")
PARAM_STRING("impute_strategy", "imputation strategy to be applied", "s", "")
PARAM_DOUBLE("custom_value", "user_defined custom value", "c", "")
PARAM_INT("feature", "the feature to apply imputation", "f", 0);

using namespace mlpack;
using namespace arma;
using namespace std;

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  const string inputFile = CLI::GetParam<string>("input_file");
  const string missingValue = CLI::GetParam<string>("missing_value");
  const string outputFile = CLI::GetParam<string>("output_file");
  const size_t feature = (size_t) CLI::GetParam<int>("feature");

  arma::mat input;
  data::DatasetInfo info;

  data::Load(inputFile, input, info,  true, true);

  Log::Info << input << endl;

  for (size_t i = 0; i < input.n_rows; ++i)
  {
    Log::Info << info.NumMappings(i) << " mappings in feature "
        << i << "." << endl;
  }

  arma::Mat<double> output(input);

  data::Imputer<
    data::MeanStrategy,
    data::DatasetInfo,
    double> impu;

  impu.Impute(input, output, info, missingValue, feature);

  Log::Info << "input::" << endl;
  Log::Info << input << endl;
  Log::Info << "output::" << endl;
  Log::Info << output << endl;

  if (!outputFile.empty())
  {
    Log::Info << "Saving model to '" << outputFile << "'." << endl;
    data::Save(outputFile, output, false);
  }
}

