/**
 * @file ham_tests.cpp
 * @author Konstantin Sidorov
 *
 * Tests for everything related with
 * Hierarchical Attentve Memory unit implementation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <vector>
#include <algorithm>
#include <utility>

#include <mlpack/core.hpp>

#include <mlpack/core/data/binarize.hpp>

#include <mlpack/methods/ann/augmented/tree_memory.hpp>
#include <mlpack/methods/ann/augmented/ham_unit.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack::ann::augmented;
using namespace mlpack::ann;

FFN<MeanSquaredError<> > ReplaceWriter(size_t dim)
{
  FFN<MeanSquaredError<> > writeModel;
  writeModel.Add<Linear<> >(2 * dim, dim);
  writeModel.ResetParameters();
  writeModel.Parameters().rows(0, dim * dim - 1) = arma::zeros(dim * dim);
  writeModel.Parameters().rows(dim * dim, 2 * dim * dim - 1) =
      arma::vectorise(arma::eye(dim, dim));
  writeModel.Parameters().rows(2 * dim * dim, 2 * dim * dim + dim - 1) = arma::zeros(dim);
  return writeModel;
}

FFN<MeanSquaredError<> > AddJoiner(size_t dim) {
  FFN<MeanSquaredError<> > joinModel;
  joinModel.Add<Linear<> >(2 * dim, dim);
  joinModel.ResetParameters();
  joinModel.Parameters().rows(0, dim * dim - 1) = arma::vectorise(arma::eye(dim, dim));
  joinModel.Parameters().rows(dim * dim, 2 * dim * dim - 1) = arma::vectorise(arma::eye(dim, dim));
  joinModel.Parameters().rows(2 * dim * dim, 2 * dim * dim + dim - 1) = arma::zeros(dim);
  return joinModel;
}

template<typename T>
arma::Mat<T> convertToArma(std::vector<std::vector<T>> stlArray)
{
  size_t rows = stlArray[0].size(), cols = stlArray.size();
  arma::Mat<T> target(rows, cols);
  size_t ptr = 0;
  for (std::vector<T> el : stlArray)
  {
    target.col(ptr++) = arma::Col<T>(el);
  }
  return target;
}

BOOST_AUTO_TEST_SUITE(HAMTest);

// Test for TreeMemory operations with 1 memory cell for 1-dim vectors.
BOOST_AUTO_TEST_CASE(TreeMemoryTestMinimum)
{
  FFN<MeanSquaredError<> > J = AddJoiner(1), W = ReplaceWriter(1);
  // With these definitions, mem is exactly one of the well-known data structres
  // in competitive programming - segment tree.
  TreeMemory<double> mem(1, 1, J, W);
  // Initialize memory with zeros.
  std::vector<std::vector<double>> initMemSTL = {{0}};
  arma::mat initMem = convertToArma(initMemSTL);
  mem.Initialize(initMem);
  // Check the consistency of memory and its contents.
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, 1);
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).at(0, 0), 0.);
  // Now update the only cell with [12] vector
  // and check that memory was correctly updated.
  mem.Update(0, arma::mat("12."));
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, 1);
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).at(0, 0), 12.);
}

// Test for TreeMemory operations with 1 memory cell for 5-dim vectors.
BOOST_AUTO_TEST_CASE(TreeMemoryTestMinimumNDim)
{
  size_t memSize = 5;
  // Initialize memory with zeros.
  FFN<MeanSquaredError<> > J = AddJoiner(memSize), W = ReplaceWriter(memSize);
  TreeMemory<double> mem(
      1, memSize, J, W);
  std::vector<std::vector<double>> initMemSTL = {{0, 0, 0, 0, 0}};
  arma::mat initMem = convertToArma(initMemSTL);
  mem.Initialize(initMem);
  // Check the consistency of memory and its contents.
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, memSize);
  for (size_t i = 0; i < memSize; ++i) {
    BOOST_REQUIRE_EQUAL(mem.Leaf(0).at(i, 0), 0.);
  }
  // Now update the only cell with [12, ..., 12] vector
  // and check that memory was correctly updated.
  mem.Update(0, 12 * arma::ones(memSize, 1));
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, memSize);
  for (size_t i = 0; i < memSize; ++i)
    BOOST_REQUIRE_EQUAL(mem.Leaf(0).at(i, 0), 12.);
}

// Test for TreeMemory operations with 8 memory cells for 1-dim vectors.
BOOST_AUTO_TEST_CASE(TreeMemoryTestPowerOfTwo)
{
  FFN<MeanSquaredError<> > J = AddJoiner(1), W = ReplaceWriter(1);
  TreeMemory<double> mem(8, 1, J, W);
  // Initialize memory with zeros.
  std::vector<std::vector<double>> initMemSTL =
      {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};
  arma::mat initMem = convertToArma(initMemSTL);
  mem.Initialize(initMem);
  // Check the memory contents.
  for (size_t idx = 0; idx < 15; ++idx)
    BOOST_REQUIRE_EQUAL(mem.Cell(idx).at(0, 0), 0);
  // Now update the first cell with [1] vector, the second - with [2] vector
  // and check that memory was correctly updated.
  mem.Update(0, arma::mat("1."));
  mem.Update(1, arma::mat("2."));
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, 1);
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).at(0, 0), 1);
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, 1);
  BOOST_REQUIRE_EQUAL(mem.Leaf(1).at(0, 0), 2);
  // Also, check that the parent-child relationships between inner nodes
  // are correctly enforced by the TreeMemory.
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.LeafIndex(0))).at(0, 0), 3);
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.LeafIndex(0)))).at(0, 0), 3);  
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.Parent(mem.LeafIndex(0))))).at(0, 0), 3); 
  // Finally, check the memory consistency.
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.LeafIndex(0))).n_elem, 1);
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.LeafIndex(0)))).n_elem, 1);  
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.Parent(mem.LeafIndex(0))))).n_elem, 1); 
}

// Test for TreeMemory operations with 8 memory cells for 5-dim vectors.
BOOST_AUTO_TEST_CASE(TreeMemoryTestPowerOfTwoNDim)
{
  size_t memSize = 4;
  FFN<MeanSquaredError<> > J = AddJoiner(memSize), W = ReplaceWriter(memSize);
  TreeMemory<double> mem(
      8, memSize, J, W);
  // Initialize memory with zeros.
  std::vector<std::vector<double>> initMemSTL =
      {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
       {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  arma::mat initMem = convertToArma(initMemSTL);
  mem.Initialize(initMem);
  // Check the memory contents.
  for (size_t idx = 0; idx < 15; ++idx)
  {
    for (size_t i = 0; i < memSize; ++i)
    {
      BOOST_REQUIRE_EQUAL(mem.Cell(idx).at(i, 0), 0);
    }
  }
  // Now update the first cell with [1, ..., 1] vector,
  // the second - with [2, ..., 2] vector
  // and check that memory was correctly updated.
  mem.Update(0, arma::ones(memSize, 1));
  mem.Update(1, 2 * arma::ones(memSize, 1));
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, memSize);
  for (size_t i = 0; i < memSize; ++i)
    BOOST_REQUIRE_EQUAL(mem.Leaf(0).at(i, 0), 1);
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, memSize);
  for (size_t i = 0; i < memSize; ++i)
    BOOST_REQUIRE_EQUAL(mem.Leaf(1).at(i, 0), 2);
  // Also, check that the parent-child relationships between inner nodes
  // are correctly enforced by the TreeMemory.
  for (size_t i = 0; i < memSize; ++i)
  {
    BOOST_REQUIRE_EQUAL(mem.Cell(
      mem.Parent(mem.LeafIndex(0))).at(i, 0), 3);
    BOOST_REQUIRE_EQUAL(mem.Cell(
      mem.Parent(mem.Parent(mem.LeafIndex(0)))).at(i, 0), 3);  
    BOOST_REQUIRE_EQUAL(mem.Cell(
      mem.Parent(mem.Parent(mem.Parent(mem.LeafIndex(0))))).at(i, 0), 3); 
  }
  // Finally, check the memory consistency.
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.LeafIndex(0))).n_elem, memSize);
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.LeafIndex(0)))).n_elem, memSize);  
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.Parent(mem.LeafIndex(0))))).n_elem, memSize); 
}

// Test for TreeMemory operations with 9 memory cells for 1-dim vectors.
BOOST_AUTO_TEST_CASE(TreeMemoryTestArbitrary) {
  FFN<MeanSquaredError<> > J = AddJoiner(1), W = ReplaceWriter(1);
  TreeMemory<double> mem(9, 1, J, W);
  // Initialize memory with zeros.
  std::vector<std::vector<double>> initMemSTL =
      {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};
  arma::mat initMem = convertToArma(initMemSTL);
  mem.Initialize(initMem);
  // Check the memory contents.
  for (size_t idx = 0; idx < 31; ++idx)
    BOOST_REQUIRE_EQUAL(mem.Cell(idx).at(0, 0), 0);
  // Now update the first cell with [1] vector,
  // the second - with [2] vector,
  // the ninth and the last - with [-3] vector
  // and check that memory was correctly updated.
  mem.Update(0, arma::mat("1."));
  mem.Update(1, arma::mat("2."));
  mem.Update(8, arma::mat("-3."));
  // Also, check that the parent-child relationships between inner nodes
  // are correctly enforced by the TreeMemory.
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).at(0, 0), 1);
  BOOST_REQUIRE_EQUAL(mem.Leaf(1).at(0, 0), 2);
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.LeafIndex(0))).at(0, 0), 3);
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.LeafIndex(0)))).at(0, 0), 3);
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.Parent(mem.LeafIndex(0))))).at(0, 0), 3); 
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.Parent(mem.Parent(mem.LeafIndex(0)))))).at(0, 0), 
    0);
  BOOST_REQUIRE_EQUAL(mem.Leaf(8).at(0, 0), -3);
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.LeafIndex(8))).at(0, 0), -3);
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.LeafIndex(8)))).at(0, 0), -3);
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.Parent(mem.LeafIndex(8))))).at(0, 0), -3); 
  BOOST_REQUIRE_EQUAL(mem.Parent(mem.LeafIndex(0)),
                      mem.Parent(mem.LeafIndex(1)));
  for (size_t i = 0; i < 9; ++i) {
    // Not quite compliant with the style guide,
    // but at least it fits in 80 charactes.
    BOOST_REQUIRE_EQUAL(
      mem.Parent(mem.Parent(mem.Parent(mem.Parent(mem.LeafIndex(i))))),
      mem.Root()
    );
  }
}

// Testing the forward pass of HAM on predefined EMBED, JOIN, WRITE, SEARCH
// functions and a predefined FFN controller.
BOOST_AUTO_TEST_CASE(BlindHAMUnitTest) {
  size_t nDim = 4, seqLen = 4;
  // Embed model is just an identity function.
  FFN<MeanSquaredError<> > embedModel;
  embedModel.Add<Linear<> >(nDim, nDim);
  // Identity = apply identity linear transformation + add zero bias.
  arma::mat embedParams = arma::zeros(nDim * nDim + nDim, 1);
  embedParams.rows(0, nDim * nDim - 1) =
      arma::vectorise(arma::eye(nDim, nDim));
  // Join function is sum of its two vector inputs.
  FFN<MeanSquaredError<> > joinModel;
  joinModel.Add<Linear<> >(2 * nDim, nDim);
  arma::mat joinParams = arma::zeros(2 * nDim * nDim + nDim, 1);
  joinParams.rows(0, nDim * nDim - 1) = arma::vectorise(arma::eye(nDim, nDim));
  joinParams.rows(nDim * nDim, 2 * nDim * nDim - 1) = arma::vectorise(arma::eye(nDim, nDim));
  // Write function is replacing its old input with its new input.
  FFN<MeanSquaredError<> > writeModel;
  writeModel.Add<Linear<> >(2 * nDim, nDim);
  arma::mat writeParams = arma::zeros(2 * nDim * nDim + nDim, 1);
  writeParams.rows(nDim * nDim, 2 * nDim * nDim - 1) =
      arma::vectorise(arma::eye(nDim, nDim));
  // Search model is a constant model that ignores its input and returns 1 / 3.
  FFN<MeanSquaredError<> > searchModel;
  searchModel.Add<Linear<> >(2 * nDim, 1);
  searchModel.Add<SigmoidLayer<> >();
  arma::mat searchParams = arma::zeros(2 * nDim + 1, 1);
  searchParams.at(2 * nDim) = -log(2);
  // Controller is a feedforward model: sigmoid(5x1 + x2 - x3 - 2x4).
  FFN<CrossEntropyError<> > controller;
  controller.Add<Linear<> >(nDim, 1);
  controller.Add<SigmoidLayer<> >();
  arma::mat controllerParams = arma::zeros(nDim + 1, 1);
  controllerParams.rows(0, nDim - 1) = arma::vec("5 1 -1 -2");

  // Pack all the parameters into a single vector.
  arma::mat allParams(
    embedParams.n_elem + searchParams.n_elem + controllerParams.n_elem +
    joinParams.n_elem + writeParams.n_elem, 1);
  size_t ptr = 0;
  std::vector<arma::mat*> ordering{
      &embedParams,
      &searchParams,
      &controllerParams,
      &joinParams,
      &writeParams
  };
  for (arma::mat* el : ordering)
  {
    allParams.rows(ptr, ptr + el->n_elem - 1) = *el;
    ptr += el->n_elem;
  }

  // Now run the HAM unit (the initial sequence is:
  // [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])
  HAMUnit<> hamUnit(seqLen, nDim, embedModel, joinModel, searchModel, writeModel, controller);
  hamUnit.ResetParameters();
  hamUnit.Parameters() = allParams;
  hamUnit.OutputParameters();

  arma::mat input("1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1;");
  input = input.t();
  arma::mat output;
  hamUnit.Forward(std::move(input), std::move(output));
  std::cerr << output;
  arma::mat targetOutput("0.4174; 0.4743; 0.5167; 0.5485;");
  BOOST_REQUIRE_SMALL(arma::abs(output - targetOutput).max(), 1e-4);

  // Finally, test that the parameters are stored correctly.
  arma::mat params = hamUnit.Parameters();
  std::vector<double> target;
  for (double el : embedModel.Parameters()) target.push_back(el);
  for (double el : searchModel.Parameters()) target.push_back(el);
  for (double el : controller.Parameters()) target.push_back(el);
  for (double el : joinModel.Parameters()) target.push_back(el);
  for (double el : writeModel.Parameters()) target.push_back(el);
  BOOST_REQUIRE_EQUAL(params.n_elem, target.size());
  for (size_t i = 0; i < params.n_elem; ++i) {
    BOOST_REQUIRE_SMALL(params.at(i, 0) - target[i], 1e-4);
  }
}

BOOST_AUTO_TEST_SUITE_END();
