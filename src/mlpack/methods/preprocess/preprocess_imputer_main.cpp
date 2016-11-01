/**
 * @file preprocess_imputer_main.cpp
 * @author Keon Kim
 *
 * a utility that provides imputation strategies fore
 * missing values.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/imputer.hpp>
#include <mlpack/core/data/dataset_mapper.hpp>
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
    "separate file to save the output data; You can save the output by "
    "specifying the file name with --output_file (-o)."
    "\n\n"
    "For example, if we consider 'NULL' in dimension 0 to be a missing "
    "variable and want to delete whole row containing the NULL in the "
    "column-wise dataset, and save the result to result.csv, we could run"
    "\n\n"
    "$ mlpack_preprocess_imputer -i dataset.csv -o result.csv -m NULL -d 0 \n"
    "> -s listwise_deletion");

PARAM_STRING_IN_REQ("input_file", "File containing data,", "i");
PARAM_STRING_OUT("output_file", "File to save output", "o");
PARAM_STRING_IN("missing_value", "User defined missing value", "m", "");
PARAM_STRING_IN("strategy", "imputation strategy to be applied. Strategies "
    "should be one of 'custom', 'mean', 'median', and 'listwise_deletion'.",
    "s", "");
PARAM_DOUBLE_IN("custom_value", "user_defined custom value", "c", 0.0);
PARAM_INT_IN("dimension", "the dimension to apply imputation", "d", 0);

using namespace mlpack;
using namespace arma;
using namespace std;
using namespace data;

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  const string inputFile = CLI::GetParam<string>("input_file");
  const string outputFile = CLI::GetParam<string>("output_file");
  const string missingValue = CLI::GetParam<string>("missing_value");
  const double customValue = CLI::GetParam<double>("custom_value");
  const size_t dimension = (size_t) CLI::GetParam<int>("dimension");
  string strategy = CLI::GetParam<string>("strategy");

  // The program needs user-defined missing values.
  // Missing values can be any list of strings such as "1", "a", "NULL".
  if (!CLI::HasParam("missing_value"))
    Log::Fatal << "--missing_value must be specified in order to perform "
        << "any imputation strategies." << endl;

  if (!CLI::HasParam("strategy"))
    Log::Fatal << "--strategy must be specified in order to perform "
        << "imputation."<< endl;

  if (!CLI::HasParam("output_file"))
    Log::Warn << "--output_file is not specified, no "
        << "results from this program will be saved!" << endl;

  if (!CLI::HasParam("dimension"))
    Log::Warn << "--dimension is not specified, the imputation will be "
        << "applied to all dimensions."<< endl;

  // If custom value is specified, and imputation strategy is not,
  // set imputation strategy to "custom"
  if (CLI::HasParam("custom_value") && !CLI::HasParam("strategy"))
  {
    strategy = "custom";
    Log::Warn << "--custom_value is specified without --strategy, "
        << "--strategy is automatically set to 'custom'." << endl;
  }

  // Custom value and any other impute strategies cannot be specified at
  // the same time.
  if (CLI::HasParam("custom_value") && CLI::HasParam("strategy") &&
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

  // print how many mapping exist in each dimensions
  std::vector<size_t> dirtyDimensions;
  for (size_t i = 0; i < input.n_rows; ++i)
  {
    size_t numMappings = info.NumMappings(i);
    if (numMappings > 0)
    {
      Log::Info << "Replacing " << numMappings << " values in dimension " << i
          << "." << endl;
      dirtyDimensions.push_back(i);
    }
  }

  if (dirtyDimensions.size() == 0)
  {
    Log::Warn << "The file does not contain any user-defined missing "
        << "variables. The program did not perform any imputation." << endl;
  }
  else if (CLI::HasParam("dimension") &&
      !(std::find(dirtyDimensions.begin(), dirtyDimensions.end(), dimension)
      != dirtyDimensions.end()))
  {
    Log::Warn << "The given dimension of the file does not contain any "
      << "user-defined missing variables. The program did not perform any "
      << "imputation." << endl;
  }
  else
  {
    // Initialize imputer class
    Imputer<double, MapperType, MeanImputation<double>> imputer(info);
    if (strategy == "mean")
    {
      Imputer<double, MapperType, MeanImputation<double>> imputer(info);
    }
    else if (strategy == "median")
    {
      Imputer<double, MapperType, MedianImputation<double>> imputer(info);
    }
    else if (strategy == "listwise_deletion")
    {
      Imputer<double, MapperType, ListwiseDeletion<double>> imputer(info);
    }
    else if (strategy == "custom")
    {
      CustomImputation<double> strat(customValue);
      Imputer<double, MapperType, CustomImputation<double>> imputer(info, strat);
    }
    else
    {
      Log::Fatal << "'" <<  strategy << "' imputation strategy does not exist"
          << endl;
    }

    Timer::Start("imputation");
    if (CLI::HasParam("dimension"))
    {
      // when --dimension is specified,
      // the program will apply the changes to only the given dimension.
      Log::Info << "Performing '" << strategy << "' imputation strategy "
          << "to replace '" << missingValue << "' on dimension " << dimension
          << "." << endl;

      imputer.Impute(input, missingValue, dimension);
    }
    else
    {
      // when --dimension is not specified,
      // the program will apply the changes to all dimensions.
      Log::Info << "Performing '" << strategy << "' imputation strategy "
          << "to replace '" << missingValue << "' on all dimensions." << endl;

      for (size_t i : dirtyDimensions)
      {
        imputer.Impute(input, missingValue, i);
      }
    }
    Timer::Stop("imputation");

    if (!outputFile.empty())
    {
      Log::Info << "Saving results to '" << outputFile << "'." << endl;
      Save(outputFile, input, false);
    }
  }
}

