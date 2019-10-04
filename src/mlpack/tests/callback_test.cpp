
#include <ensmallen.hpp>
#include <ensmallen_bits/callbacks/callbacks.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(CallbackTest);

BOOST_AUTO_TEST_CASE(FFNCallbackTest)
{
  arma::mat data;
  arma::mat labels;

  data::Load("lab1.csv", data, true);
  data::Load("lab3.csv", labels, true);

  FFN<MeanSquaredError<>, RandomInitialization> model;

  model.Add<Linear<>>(1, 2);
  model.Add<SigmoidLayer<>>();
  model.Add<Linear<>>(2, 1);
  model.Add<SigmoidLayer<>>();


  std::stringstream stream;
  model.Train(data, labels, ens::PrintLoss(stream));

  BOOST_REQUIRE_GT(stream.str().length(), 0);
}

BOOST_AUTO_TEST_SUITE_END();