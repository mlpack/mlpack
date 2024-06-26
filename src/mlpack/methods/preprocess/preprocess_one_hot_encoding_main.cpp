/**
 * @file methods/preprocess/preprocess_one_hot_encoding_main.cpp
 * @author Jeffin Sam
 *
 * A binding to do one-hot encoding on features from a dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME preprocess_one_hot_encoding

#include <mlpack/core/util/mlpack_main.hpp>

// Program Name.
BINDING_USER_NAME("One Hot Encoding");

// Short description.
BINDING_SHORT_DESC("A utility to do one-hot encoding on features of dataset.");

// Long description.
BINDING_LONG_DESC(
    "This utility takes a dataset and a vector of indices and does one-hot "
    "encoding of the respective features at those indices. Indices represent "
    "the IDs of the dimensions to be one-hot encoded."
    "\n\n"
    "If no dimensions are specified with " + PRINT_PARAM_STRING("dimensions") +
    ", then all categorical-type dimensions will be one-hot encoded. "
    "Otherwise, only the dimensions given in " +
    PRINT_PARAM_STRING("dimensions") + " will be one-hot encoded."
    "\n\n"
    "The output matrix with encoded features may be saved with the " +
    PRINT_PARAM_STRING("output") + " parameters.");

// Example.
BINDING_EXAMPLE(
    "So, a simple example where we want to encode 1st and 3rd feature"
    " from dataset " + PRINT_DATASET("X") + " into " +
    PRINT_DATASET("X_output") + " would be"
    "\n\n" +
    PRINT_CALL("preprocess_one_hot_encoding", "input", "X", "output",
        "X_ouput", "dimensions", 1 , "dimensions", 3));

// See also...
BINDING_SEE_ALSO("@preprocess_binarize", "#preprocess_binarize");
BINDING_SEE_ALSO("@preprocess_describe", "#preprocess_describe");
#if BINDING_TYPE == BINDING_TYPE_CLI
BINDING_SEE_ALSO("@preprocess_imputer", "#preprocess_imputer");
#endif
BINDING_SEE_ALSO("One-hot encoding on Wikipedia",
        "https://en.m.wikipedia.org/wiki/One-hot");

// Define parameters for data.
PARAM_MATRIX_AND_INFO_IN_REQ("input", "Matrix containing data.", "i");
PARAM_MATRIX_OUT("output", "Matrix to save one-hot encoded features "
    "data to.", "o");

PARAM_VECTOR_IN(int, "dimensions", "Index of dimensions that need to be one-hot"
    " encoded (if unspecified, all categorical dimensions are one-hot "
    "encoded).", "d");

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
  // Load the data.
  const std::tuple<data::DatasetInfo, arma::mat>& t =
      params.Get<std::tuple<data::DatasetInfo, arma::mat>>("input");

  const data::DatasetInfo& info = std::get<0>(t);
  const arma::mat& data = std::get<1>(t);

  vector<int>& indices = params.Get<vector<int>>("dimensions");
  if (!params.Has("dimensions"))
  {
    // If the user did not specify any dimensions to convert, we pick all the
    // categorical dimensions by default.
    for (size_t d = 0; d < info.Dimensionality(); ++d)
      if (info.Type(d) == data::Datatype::categorical)
        indices.push_back(d);

    // Print which dimensions we selected to one-hot encode.
    if (indices.size() > 0)
    {
      Log::Info << "One-hot encoding categorical dimensions: [";
      for (size_t i = 0; i < indices.size() - 1; ++i)
        Log::Info << indices[i] << ", ";
      Log::Info << indices[indices.size() - 1] << "]." << std::endl;
    }
  }
  else
  {
    // If the user did specify dimensions, let's make sure they are reasonable.
    RequireParamValue<std::vector<int>>(params, "dimensions",
        [data](std::vector<int> x)
        {
          for (int dim : x)
          {
            if (dim < 0 || (size_t) dim > data.n_rows)
            {
              return false;
            }
          }
          return true;
        }, true, "dimensions must be greater than 0 and less than the number of"
        " dimensions");
  }

  // Note that it's possible that zero dimensions are selected for one-hot
  // encoding.
  if (indices.size() > 0)
  {
    vector<size_t> copyIndices(indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
      copyIndices[i] = (size_t)indices[i];
    }

    arma::mat output;
    data::OneHotEncoding(data, (arma::Col<size_t>)(copyIndices), output);
    if (params.Has("output"))
      params.Get<arma::mat>("output") = std::move(output);
  }
  else if (params.Has("output"))
  {
    params.Get<arma::mat>("output") = data; // Copy input to output.
  }
}
