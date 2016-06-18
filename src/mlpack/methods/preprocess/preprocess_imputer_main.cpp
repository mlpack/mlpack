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
#include <mlpack/core/data/map_policies/increment_policy.hpp>
#include <mlpack/core/data/imputation_methods/mean_imputation.hpp>
#include <mlpack/core/data/imputation_methods/median_imputation.hpp>
#include <mlpack/core/data/imputation_methods/custom_imputation.hpp>
#include <mlpack/core/data/imputation_methods/listwise_deletion.hpp>

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
using namespace data;

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  const string inputFile = CLI::GetParam<string>("input_file");
  const string outputFile = CLI::GetParam<string>("output_file");
  const string missingValue = CLI::GetParam<string>("missing_value");
  const string mapPolicy = CLI::GetParam<string>("map_policy");
  const double customValue = CLI::GetParam<double>("custom_value");
  const size_t feature = (size_t) CLI::GetParam<int>("feature");
  string imputeStrategy = CLI::GetParam<string>("impute_strategy");

  // missing value should be specified
  if (!CLI::HasParam("missing_value"))
    Log::Fatal << "--missing_value must be specified in order to perform "
        << "any imputation strategies." << endl;

  // warn if user did not specify output_file
  if (!CLI::HasParam("output_file"))
    Log::Warn << "--output_file is not specified, no "
        << "results from this program will be saved!" << endl;

  // if custom value is specified, and imputation strategy is not,
  // set imputation strategy to "custom"
  if (CLI::HasParam("custom_value") && !CLI::HasParam("impute_strategy"))
  {
    imputeStrategy = "custom";
    Log::Warn << "--custom_value is specified without --impute_strategy, "
        << "--impute_strategy is automatically set to 'custom'." << endl;
  }

  // custom value and any other impute strategies cannot be specified at
  // the same time.
  if (CLI::HasParam("custom_value") && CLI::HasParam("impute_strategy") &&
      imputeStrategy != "custom")
    Log::Fatal << "--custom_value cannot be specified with "
        << "impute strategies excluding 'custom' strategy" << endl;

  // custom_value must be specified when using "custom" imputation strategy
  if ((imputeStrategy == "custom") && !CLI::HasParam("custom_value"))
    Log::Fatal << "--custom_value must be specified when using "
        << "'custom' strategy" << endl;

  arma::mat input;
  // DatasetInfo holds how the DatasetMapper should map the values.
  // can be specified by passing map_policy classes as template parameters
  // ex) DatasetMapper<IncrementPolicy> info;
  using MapperType = DatasetMapper<IncrementPolicy>;
  MapperType info;

  Load(inputFile, input, info,  true, true);

  // for testing purpose
  Log::Info << input << endl;

  // print how many mapping exist in each features
  for (size_t i = 0; i < input.n_rows; ++i)
  {
    Log::Info << info.NumMappings(i) << " mappings in feature " << i << "."
        << endl;
  }

  arma::Mat<double> output(input);


  Log::Info << "Performing '" << imputeStrategy << "' imputation strategy "
      << "on feature '" << feature << endl;

  // custom strategy only
  if (imputeStrategy == "custom")
  {
    Log::Info << "Replacing all '" << missingValue << "' with '" << customValue
        << "'." << endl;
    Imputer<double, MapperType, CustomImputation<double>> impu(info);
    impu.Impute(input, output, missingValue, customValue, feature);
  }
  else
  {
    Log::Info << "Replacing all '" << missingValue << "' with '"
        << imputeStrategy << "'." << endl;

    Imputer<double, MapperType, MeanImputation<double>> impu(info);
    impu.Impute(input, output, missingValue, feature);
  }

  // for testing purpose
  Log::Info << "input::" << endl;
  Log::Info << input << endl;
  Log::Info << "output::" << endl;
  Log::Info << output << endl;

  if (!outputFile.empty())
  {
    Log::Info << "Saving model to '" << outputFile << "'." << endl;
    Save(outputFile, output, false);
  }
}

