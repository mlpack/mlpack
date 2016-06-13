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
PARAM_STRING("impute_strategy", "imputation strategy to be applied", "s", "")
PARAM_DOUBLE("custom_value", "user_defined custom value", "c", 0.0)
PARAM_INT("feature", "the feature to apply imputation", "f", 0);

using namespace mlpack;
using namespace arma;
using namespace std;

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  const string inputFile = CLI::GetParam<string>("input_file");
  const string outputFile = CLI::GetParam<string>("output_file");
  const string missingValue = CLI::GetParam<string>("missing_value");
  const string mapPolicy = CLI::GetParam<string>("map_policy");
  const string imputeStrategy = CLI::GetParam<string>("impute_strategy");
  const double customValue = CLI::GetParam<double>("custom_value");
  const size_t feature = (size_t) CLI::GetParam<int>("feature");

  // warn if user did not specify output_file
  if (!CLI::HasParam("output_file"))
    Log::Warn << "--output_file is not specified, no "
              << "results from this program will be saved!" << endl;

  if (CLI::HasParam("custom_value") && !(imputeStrategy == "custom"))
  {
    Log::Warn << "--custom_value is specified without --impute_strategy, "
              << "--impute_strategy is automatically set to CustomStrategy."
              << endl;
  }

  if ((imputeStrategy == "custom") && !CLI::HasParam("custom_value"))
    Log::Fatal << "--custom_value must be specified when using "
               << "'custom' strategy" << endl;

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


  if (imputeStrategy == "custom")
  {
    data::Imputer<arma::Mat<double>,
                  data::DatasetInfo,
                  data::CustomStrategy> impu;
    impu.template Impute<double>(input,
                                 output,
                                 info,
                                 missingValue,
                                 customValue,
                                 feature);
  }
  else
  {
    data::Imputer<arma::Mat<double>,
                data::DatasetInfo,
                data::MeanStrategy> impu;

    impu.Impute(input, output, info, missingValue, feature);
  }
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

