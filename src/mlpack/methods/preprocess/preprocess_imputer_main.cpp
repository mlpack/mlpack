/**
 * @file preprocess_imputer_main.cpp
 * @author Keon Kim
 *
 * A utility that provides imputation strategies for missing values.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/data/load_impl.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/data/imputer.hpp>
#include <mlpack/core/data/dataset_mapper.hpp>
#include <mlpack/core/data/map_policies/increment_policy.hpp>
#include <mlpack/core/data/map_policies/missing_policy.hpp>
#include <mlpack/core/data/imputation_methods/mean_imputation.hpp>
#include <mlpack/core/data/imputation_methods/median_imputation.hpp>
#include <mlpack/core/data/imputation_methods/custom_imputation.hpp>
#include <mlpack/core/data/imputation_methods/listwise_deletion.hpp>

PROGRAM_INFO("Impute Data",
    // Short description.
    "This utility provides several imputation strategies for missing data. "
    "Given a dataset with missing values, this can impute according to several "
    "strategies, including user-defined values.",
    // Long description.
    "This utility takes a dataset and converts a user-defined missing variable "
    "to another to provide more meaningful analysis."
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
    "> -s listwise_deletion",
    SEE_ALSO("@preprocess_binarize", "#preprocess_binarize"),
    SEE_ALSO("@preprocess_describe", "#preprocess_describe"),
    SEE_ALSO("@preprocess_split", "#preprocess_split"));

PARAM_STRING_IN_REQ("input_file", "File containing data.", "i");
PARAM_STRING_OUT("output_file", "File to save output into.", "o");
PARAM_STRING_IN_REQ("missing_value", "User defined missing value.", "m");
PARAM_STRING_IN_REQ("strategy", "imputation strategy to be applied. Strategies "
    "should be one of 'custom', 'mean', 'median', and 'listwise_deletion'.",
    "s");
PARAM_DOUBLE_IN("custom_value", "User-defined custom imputation value.", "c",
    0.0);
PARAM_INT_IN("dimension", "The dimension to apply imputation to.", "d", 0);

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;
using namespace data;

static void mlpackMain()
{
  const string inputFile = CLI::GetParam<string>("input_file");
  const string outputFile = CLI::GetParam<string>("output_file");
  const string missingValue = CLI::GetParam<string>("missing_value");
  const double customValue = CLI::GetParam<double>("custom_value");
  const size_t dimension = (size_t) CLI::GetParam<int>("dimension");
  string strategy = CLI::GetParam<string>("strategy");

  RequireParamInSet<string>("strategy", { "custom", "mean", "median",
      "listwise_deletion" }, true, "unknown imputation strategy");
  RequireAtLeastOnePassed({ "output_file" }, false, "no output will be saved");

  if (!CLI::HasParam("dimension"))
  {
    Log::Warn << "--dimension is not specified; the imputation will be "
        << "applied to all dimensions."<< endl;
  }

  if (strategy != "custom")
    ReportIgnoredParam("custom_value", "not using custom imputation strategy");
  else
    RequireAtLeastOnePassed({ "custom_value" }, true, "must pass custom "
        "imputation value when using 'custom' imputation strategy");

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
    Timer::Start("imputation");
    if (CLI::HasParam("dimension"))
    {
      // when --dimension is specified,
      // the program will apply the changes to only the given dimension.
      Log::Info << "Performing '" << strategy << "' imputation strategy "
          << "to replace '" << missingValue << "' on dimension " << dimension
          << "." << endl;
      if (strategy == "mean")
      {
        Imputer<double, MapperType, MeanImputation<double>> imputer(info);
        imputer.Impute(input, missingValue, dimension);
      }
      else if (strategy == "median")
      {
        Imputer<double, MapperType, MedianImputation<double>> imputer(info);
        imputer.Impute(input, missingValue, dimension);
      }
      else if (strategy == "listwise_deletion")
      {
        Imputer<double, MapperType, ListwiseDeletion<double>> imputer(info);
        imputer.Impute(input, missingValue, dimension);
      }
      else if (strategy == "custom")
      {
        CustomImputation<double> strat(customValue);
        Imputer<double, MapperType, CustomImputation<double>> imputer(
            info, strat);
        imputer.Impute(input, missingValue, dimension);
      }
      else
      {
        Log::Fatal << "'" <<  strategy << "' imputation strategy does not exist"
            << endl;
      }
    }
    else
    {
      // when --dimension is not specified,
      // the program will apply the changes to all dimensions.
      Log::Info << "Performing '" << strategy << "' imputation strategy "
          << "to replace '" << missingValue << "' on all dimensions." << endl;

      if (strategy == "mean")
      {
        Imputer<double, MapperType, MeanImputation<double>> imputer(info);
        for (size_t i : dirtyDimensions)
          imputer.Impute(input, missingValue, i);
      }
      else if (strategy == "median")
      {
        Imputer<double, MapperType, MedianImputation<double>> imputer(info);
        for (size_t i : dirtyDimensions)
          imputer.Impute(input, missingValue, i);
      }
      else if (strategy == "listwise_deletion")
      {
        Imputer<double, MapperType, ListwiseDeletion<double>> imputer(info);
        for (size_t i : dirtyDimensions)
          imputer.Impute(input, missingValue, i);
      }
      else if (strategy == "custom")
      {
        CustomImputation<double> strat(customValue);
        Imputer<double, MapperType, CustomImputation<double>> imputer(
            info, strat);
        for (size_t i : dirtyDimensions)
          imputer.Impute(input, missingValue, i);
      }
      else
      {
        Log::Fatal << "'" <<  strategy << "' imputation strategy does not "
            << "exist!" << endl;
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
