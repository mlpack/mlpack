/**
 * @file methods/preprocess/preprocess_one_hot_encoding_main.cpp
 * @author Jeffin Sam
 *
 * A IO executable to do One-Hot Encoding on features from a dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/data/one_hot_encoding.hpp>

PROGRAM_INFO("One Hot Encoding",
    // Short description.
    "A utility to do one hot encoding on features of dataset.",
    // Long description.
    "This utility takes a dataset and a vector of indices and does one hot "
    "encoding of the respective features at those indices. Indices represent "
    "the IDs of the dimensions to be one-hot encoded."
    "\n\n"
    "The output matrices with encoded features may be saved with the " +
    PRINT_PARAM_STRING("output") + " parameters."
    "\n\n"
    "So, a simple example where we want to encode 1st and 3rd feature"
    " from dataset " + PRINT_DATASET("X") + " into " +
    PRINT_DATASET("X_output") + " would be"
    "\n\n" +
    PRINT_CALL("preprocess_one_hot_encoding", "input", "X", "output",
        "X_ouput", "dimensions", 1 , "dimensions", 3),
    SEE_ALSO("@preprocess_binarize", "#preprocess_binarize"),
    SEE_ALSO("@preprocess_describe", "#preprocess_describe"),
    SEE_ALSO("@preprocess_imputer", "#preprocess_imputer"));

// Define parameters for data.
PARAM_MATRIX_IN_REQ("input", "Matrix containing data.", "i");
PARAM_MATRIX_OUT("output", "Matrix to save one hot encoded features "
    "data to.", "o");

PARAM_VECTOR_IN_REQ(int, "dimensions", "Index of dimensions that"
    "need to be one hot encoded.", "d");

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

static void mlpackMain()
{
  // Load the data.
  const arma::mat& data = IO::GetParam<arma::mat>("input");
  vector<int>& indices =
    IO::GetParam<vector<int> >("dimensions");
  vector<size_t>copyIndices(indices.size());
  for (size_t i = 0; i < indices.size(); i++)
  {
    if (indices[i] < 0 || indices[i] >= data.n_rows)
      throw std::runtime_error("Indices should be having positive value"
      " and less than then the number of dimesnions of the input data");
    copyIndices[i] = (size_t)indices[i];
  }
  arma::mat output;
  data::OneHotEncoding(data, (arma::Col<size_t>)(copyIndices), output);
  if (IO::HasParam("output"))
    IO::GetParam<arma::mat>("output") = std::move(output);
}
