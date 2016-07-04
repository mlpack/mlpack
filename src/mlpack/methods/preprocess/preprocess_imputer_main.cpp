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
#include <mlpack/core/data/map_policies/missing_policy.hpp>
#include <mlpack/core/data/imputation_methods/mean_imputation.hpp>
#include <mlpack/core/data/imputation_methods/median_imputation.hpp>
#include <mlpack/core/data/imputation_methods/custom_imputation.hpp>
#include <mlpack/core/data/imputation_methods/listwise_deletion.hpp>

PROGRAM_INFO("Impute Data", "This utility takes a dataset and converts user "
    "defined missing variable to another to provide more meaningful analysis "
    "\n\n"
    "The program does not modify the original file, but instead makes a "
    "separate file to save the output data; The program requires you to "
    "specify the file name with --output_file (-o)."
    "\n\n"
    "For example, if we consider 'NULL' in dimension 0 to be a missing "
    "variable and want to delete whole row containing the NULL in the "
    "column-wise dataset, and save the result to result.csv, we could run"
    "\n\n"
    "$ mlpack_preprocess_imputer -i dataset.csv -o result.csv -m NULL -d 0 \n"
    "> -s listwise_deletion")

PARAM_STRING_REQ("input_file", "File containing data,", "i");
PARAM_STRING("output_file", "File to save output", "o", "");
PARAM_STRING("missing_value", "User defined missing value", "m", "")
PARAM_STRING("strategy", "imputation strategy to be applied", "s", "")
PARAM_DOUBLE("custom_value", "user_defined custom value", "c", 0.0)
PARAM_INT("dimension", "the dimension to apply imputation", "d", 0);

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
  const double customValue = CLI::GetParam<double>("custom_value");
  const size_t dimension = (size_t) CLI::GetParam<int>("dimension");
  string strategy = CLI::GetParam<string>("strategy");

  // missing value should be specified
  if (!CLI::HasParam("missing_value"))
    Log::Fatal << "--missing_value must be specified in order to perform "
        << "any imputation strategies." << endl;

  // warn if user did not specify output_file
  if (!CLI::HasParam("output_file"))
    Log::Warn << "--output_file is not specified, no "
        << "results from this program will be saved!" << endl;

  // warn if user did not specify dimension
  if (!CLI::HasParam("dimension"))
    Log::Warn << "--dimension is required to be specified!" << endl;

  // if custom value is specified, and imputation strategy is not,
  // set imputation strategy to "custom"
  if (CLI::HasParam("custom_value") && !CLI::HasParam("impute_strategy"))
  {
    strategy = "custom";
    Log::Warn << "--custom_value is specified without --impute_strategy, "
        << "--impute_strategy is automatically set to 'custom'." << endl;
  }

  // custom value and any other impute strategies cannot be specified at
  // the same time.
  if (CLI::HasParam("custom_value") && CLI::HasParam("impute_strategy") &&
      strategy != "custom")
    Log::Fatal << "--custom_value cannot be specified with "
        << "impute strategies excluding 'custom' strategy" << endl;

  // custom_value must be specified when using "custom" imputation strategy
  if ((strategy == "custom") && !CLI::HasParam("custom_value"))
    Log::Fatal << "--custom_value must be specified when using "
        << "'custom' strategy" << endl;

  arma::mat input;
  // Policy tells how the DatasetMapper should map the values.
  std::set<std::string> missingSet;
  missingSet.insert(missingValue);
  MissingPolicy policy(missingSet);
  using MapperType = DatasetMapper<MissingPolicy>;
  DatasetMapper<MissingPolicy> info(policy);

  Load(inputFile, input, info, true, true);

  // for testing purpose
  Log::Info << input << endl;

  // print how many mapping exist in each dimensions
  for (size_t i = 0; i < input.n_rows; ++i)
  {
    Log::Info << info.NumMappings(i) << " mappings in dimension " << i << "."
        << endl;
  }

  arma::Mat<double> output(input);

  Log::Info << "Performing '" << strategy << "' imputation strategy "
      << "on dimension '" << dimension << "'." << endl;

  // custom strategy only
  if (strategy == "custom")
  {
    Log::Info << "Replacing all '" << missingValue << "' with '" << customValue
        << "'." << endl;
    Imputer<double, MapperType, CustomImputation<double>> impu(info);
    impu.Impute(input, output, missingValue, customValue, dimension);
  }
  else
  {
    Log::Info << "Replacing all '" << missingValue << "' with '"
        << strategy << "' strategy." << endl;

    if (strategy == "mean")
    {
      Imputer<double, MapperType, MeanImputation<double>> impu(info);
      impu.Impute(input, output, missingValue, dimension);
    }
    else if (strategy == "median")
    {
      Imputer<double, MapperType, MedianImputation<double>> impu(info);
      impu.Impute(input, output, missingValue, dimension);
    }
    else if (strategy == "listwise_deletion")
    {
      Imputer<double, MapperType, ListwiseDeletion<double>> impu(info);
      impu.Impute(input, output, missingValue, dimension);
    }
    else
    {
      Log::Warn << "You did not choose any imputation strategy" << endl;
    }
  }


  if (!outputFile.empty())
  {
    Log::Info << "Saving model to '" << outputFile << "'." << endl;
    Save(outputFile, output, false);
  }
}

