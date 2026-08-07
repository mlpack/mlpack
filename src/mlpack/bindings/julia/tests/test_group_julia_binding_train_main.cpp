/**
 * @file bindings/julia/tests/test_group_julia_binding_train_main.cpp
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
#define BINDING_NAME test_group_julia_binding_train

#include <mlpack/core/util/mlpack_main.hpp>
#include "test_group_julia_binding.hpp"

using namespace std;
using namespace mlpack;

// Program Name.
BINDING_USER_NAME("Julia grouped binding test");

// Short description.
BINDING_SHORT_DESC(
    "A simple program to test grouped Julia binding functionality.");

// Long description.
BINDING_LONG_DESC(
    "A simple program to test grouped Julia binding functionality.  You can "
    "build mlpack with the BUILD_TESTS option set to off, and this binding will"
    " no longer be built.");

PARAM_MATRIX_IN_REQ("training", "(Fake) training data.", "t");
PARAM_UROW_IN_REQ("training_labels", "(Fake) training labels.", "l");
PARAM_MODEL_OUT(TestGroupJuliaBinding, "output_model", "Output model.", "M");
PARAM_INT_IN("unused_hyperparam", "Unused hyperparameter.", "u", 0);

void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
  arma::mat training = std::move(params.Get<arma::mat>("training"));
  arma::Row<size_t> labels =
      std::move(params.Get<arma::Row<size_t>>("training_labels"));

  // Set the bandwidth to... something.
  TestGroupJuliaBinding* t = new TestGroupJuliaBinding(
      arma::accu(training) + arma::accu(labels));

  params.Get<TestGroupJuliaBinding*>("output_model") = t;
}
