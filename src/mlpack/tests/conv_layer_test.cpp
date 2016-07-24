/**
 * @file conv_layer_test.cpp
 * @author Nilay Jain
 *
 * Tests the convolution layer.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/conv_layer.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(ConvLayerTest);
void Test(arma::mat m1, arma::mat m2)
{
  for (size_t i = 0; i < m1.n_cols; ++i)
    BOOST_REQUIRE_CLOSE(m1(i), m2(i), 1e-2);  
}

void Test(arma::cube m1, arma::cube m2)
{
  BOOST_REQUIRE_EQUAL(m1.n_slices, m2.n_slices);
  for (size_t i = 0; i < m1.n_slices; ++i)
    Test(m1.slice(i), m2.slice(i));  
}

void ConvLayerTest()
{
  arma::vec a = arma::linspace<arma::vec>(1, 25, 25);
  arma::cube input(5, 5, 4);
  for (size_t i = 0; i < input.n_slices; ++i)
  {
    int vec_idx = 0, row_idx = 0;
    for (size_t j = 0; j < input.n_rows; ++j)
    {
      input.slice(i).row(j) = a.subvec(vec_idx, vec_idx + 4).t();
      vec_idx += 5;
    }
  }
  ConvLayer<> conv3(4, 1, 3, 3, 1, 1, 1, 1);
  arma::mat id3 = arma::zeros<arma::mat>(3, 3);
  id3(1, 1) = 1;  
  arma::cube conv3_w(3, 3, 4 * 1);
  for (size_t i = 0; i < conv3_w.n_slices; ++i) 
    conv3_w.slice(i) = id3;
  conv3.Weights() = conv3_w;
  
  arma::cube output(5, 5, 1);
  output.slice(0) = input.slice(0) * 4;
  arma::cube convOutput;
  conv3.Forward(input, convOutput);
  Test(convOutput, output);
}

//! tests the forward pass for the conv_layer.
BOOST_AUTO_TEST_CASE(ForwardPassTest)
{
  ConvLayerTest();
}
BOOST_AUTO_TEST_SUITE_END();
