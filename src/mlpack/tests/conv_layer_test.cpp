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

/*
 * Test whether Matrices m1 and m2 have all values within 0.01.
 */
void Test(arma::mat m1, arma::mat m2)
{
  for (size_t i = 0; i < m1.n_elem; ++i)
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

  //! Populate a cube with values 1-25.
  for (size_t i = 0; i < input.n_slices; ++i)
  {
    int vecIdx = 0;
    for (size_t j = 0; j < input.n_rows; ++j)
    {
      input.slice(i).row(j) = a.subvec(vecIdx, vecIdx + 4).t();
      vecIdx += 5;
    }
  }
  // apply convolution layer with padding 1 in all dimension.
  ConvLayer<> conv3(4, 1, 3, 3, 1, 1, 1, 1);
  arma::cube conv3W = arma::zeros<arma::cube>(3, 3, 4 * 1);
  for (size_t i = 0; i < conv3W.n_slices; ++i) 
    conv3W(1, 1, i) = 1;
  // filter applied is the identity kernel.
  conv3.Weights() = conv3W;
  
  // test the forward pass.
  arma::cube output(5, 5, 1);
  output.slice(0) = input.slice(0) * 4;
  arma::cube convOutput;
  conv3.InputParameter() = input;
  conv3.Forward(conv3.InputParameter(), conv3.OutputParameter());
  Test(conv3.OutputParameter(), output);
  
  // for backward pass, let the error is a matrix of ones.
  arma::cube error = arma::ones(5, 5, 1);
  conv3.Backward(conv3.InputParameter(), error, conv3.Delta());
  // test the backward pass.
  Test(conv3.Delta(), arma::ones(5, 5, 4));

  // test the gradient update assuming the delta as identity kernel.
  arma::cube delta = arma::zeros<arma::cube>(5, 5, 1);
  delta(2, 2, 0) = 1;
  arma::mat grad = conv3.InputParameter().slice(0).submat(1, 1, 3, 3);
  arma::cube gradCube(3, 3, 4);
  for (size_t i = 0; i < conv3.InputParameter().n_slices; ++i)
    gradCube.slice(i) = grad;
  conv3.Gradient(conv3.InputParameter(), delta, conv3.Gradient());
  Test(gradCube, conv3.Gradient());
}

//! tests the forward pass for the conv_layer.
BOOST_AUTO_TEST_CASE(ForwardPassTest)
{
  ConvLayerTest();
}
BOOST_AUTO_TEST_SUITE_END();
