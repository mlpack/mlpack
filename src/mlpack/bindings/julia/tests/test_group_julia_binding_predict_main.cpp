/**
 * @file bindings/julia/tests/test_group_julia_binding_predict_main.cpp
 * @author Ryan Curtin
 *
 * A binding test for Julia.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>

#undef BINDING_NAME
#define BINDING_NAME test_group_julia_binding_predict

#include <mlpack/core/util/mlpack_main.hpp>
#include "test_group_julia_binding.hpp"

using namespace std;
using namespace mlpack;

// Program Name.
BINDING_USER_NAME("Julia grouped binding test prediction");

// Short description.
BINDING_SHORT_DESC(
    "A simple program to test grouped Julia binding functionality.");

// Long description.
BINDING_LONG_DESC(
    "A simple program to test grouped Julia binding functionality.  You can "
    "build mlpack with the BUILD_TESTS option set to off, and this binding will"
    " no longer be built.");

PARAM_MODEL_IN_REQ(TestGroupJuliaBinding, "input_model", "Input model.", "m");
PARAM_MATRIX_IN_REQ("test", "(Fake) test data.", "T");

PARAM_MATRIX_OUT("predictions", "(Fake) test predictions.", "p");

void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
  TestGroupJuliaBinding* t = params.Get<TestGroupJuliaBinding*>("input_model");
  arma::mat test = std::move(params.Get<arma::mat>("test"));

  // Do some kind of computation that makes some kind of predictions.
  arma::mat predictions(1, test.n_cols);
  for (size_t i = 0; i < test.n_cols; ++i)
  {
    predictions[i] = t->Kernel().Evaluate(test.col(i),
        test.col(i) + arma::randu<arma::vec>(test.n_rows));
  }

  params.Get<arma::mat>("predictions") = std::move(predictions);
}
