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

BOOST_AUTO_TEST_CASE(TreeMemoryTestMinimum)
{
  /*AddJoiner<double> J;
  ReplaceWriter<double> W;*/
  FFN<MeanSquaredError<> > J = AddJoiner(1), W = ReplaceWriter(1);
  // With these definitions, mem is exactly one of the well-known data structres
  // in competitive programming - segment tree.
  TreeMemory<double> mem(1, 1, J, W);
  std::vector<std::vector<double>> initMemSTL = {{0}};
  arma::mat initMem = convertToArma(initMemSTL);
  mem.Initialize(initMem);
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, 1);
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).at(0, 0), 0.);
  mem.Update(0, arma::mat("12."));
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, 1);
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).at(0, 0), 12.);
}

BOOST_AUTO_TEST_CASE(TreeMemoryTestMinimumNDim)
{
  /*AddJoiner<double> J;
  ReplaceWriter<double> W;*/
  size_t memSize = 5;
  FFN<MeanSquaredError<> > J = AddJoiner(memSize), W = ReplaceWriter(memSize);
  TreeMemory<double> mem(
      1, memSize, J, W);
  std::vector<std::vector<double>> initMemSTL = {{0, 0, 0, 0, 0}};
  arma::mat initMem = convertToArma(initMemSTL);
  mem.Initialize(initMem);
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, memSize);
  for (size_t i = 0; i < memSize; ++i) {
    BOOST_REQUIRE_EQUAL(mem.Leaf(0).at(i, 0), 0.);
  }
  mem.Update(0, 12 * arma::ones(memSize, 1));
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, memSize);
  for (size_t i = 0; i < memSize; ++i)
    BOOST_REQUIRE_EQUAL(mem.Leaf(0).at(i, 0), 12.);
}

BOOST_AUTO_TEST_CASE(TreeMemoryTestPowerOfTwo)
{
  /*AddJoiner<double> J;
  ReplaceWriter<double> W;*/
  FFN<MeanSquaredError<> > J = AddJoiner(1), W = ReplaceWriter(1);
  TreeMemory<double> mem(8, 1, J, W);
  std::vector<std::vector<double>> initMemSTL =
      {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};
  arma::mat initMem = convertToArma(initMemSTL);
  mem.Initialize(initMem);
  for (size_t idx = 0; idx < 15; ++idx)
    BOOST_REQUIRE_EQUAL(mem.Cell(idx).at(0, 0), 0);
  mem.Update(0, arma::mat("1."));
  mem.Update(1, arma::mat("2."));
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, 1);
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).at(0, 0), 1);
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, 1);
  BOOST_REQUIRE_EQUAL(mem.Leaf(1).at(0, 0), 2);
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.LeafIndex(0))).at(0, 0), 3);
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.LeafIndex(0)))).at(0, 0), 3);  
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.Parent(mem.LeafIndex(0))))).at(0, 0), 3); 
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.LeafIndex(0))).n_elem, 1);
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.LeafIndex(0)))).n_elem, 1);  
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.Parent(mem.LeafIndex(0))))).n_elem, 1); 
}

BOOST_AUTO_TEST_CASE(TreeMemoryTestPowerOfTwoNDim)
{
  /*AddJoiner<double> J;
  ReplaceWriter<double> W;*/
  size_t memSize = 4;
  FFN<MeanSquaredError<> > J = AddJoiner(memSize), W = ReplaceWriter(memSize);
  TreeMemory<double> mem(
      8, memSize, J, W);
  std::vector<std::vector<double>> initMemSTL =
      {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
       {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  arma::mat initMem = convertToArma(initMemSTL);
  mem.Initialize(initMem);
  for (size_t idx = 0; idx < 15; ++idx)
  {
    for (size_t i = 0; i < memSize; ++i)
    {
      BOOST_REQUIRE_EQUAL(mem.Cell(idx).at(i, 0), 0);
    }
  }
  mem.Update(0, arma::ones(memSize, 1));
  mem.Update(1, 2 * arma::ones(memSize, 1));
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, memSize);
  for (size_t i = 0; i < memSize; ++i)
    BOOST_REQUIRE_EQUAL(mem.Leaf(0).at(i, 0), 1);
  BOOST_REQUIRE_EQUAL(mem.Leaf(0).n_elem, memSize);
  for (size_t i = 0; i < memSize; ++i)
    BOOST_REQUIRE_EQUAL(mem.Leaf(1).at(i, 0), 2);
  for (size_t i = 0; i < memSize; ++i)
  {
    BOOST_REQUIRE_EQUAL(mem.Cell(
      mem.Parent(mem.LeafIndex(0))).at(i, 0), 3);
    BOOST_REQUIRE_EQUAL(mem.Cell(
      mem.Parent(mem.Parent(mem.LeafIndex(0)))).at(i, 0), 3);  
    BOOST_REQUIRE_EQUAL(mem.Cell(
      mem.Parent(mem.Parent(mem.Parent(mem.LeafIndex(0))))).at(i, 0), 3); 
  }
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.LeafIndex(0))).n_elem, memSize);
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.LeafIndex(0)))).n_elem, memSize);  
  BOOST_REQUIRE_EQUAL(mem.Cell(
    mem.Parent(mem.Parent(mem.Parent(mem.LeafIndex(0))))).n_elem, memSize); 
}

BOOST_AUTO_TEST_CASE(TreeMemoryTestArbitrary) {
  /*AddJoiner<double> J;
  ReplaceWriter<double> W;*/
  FFN<MeanSquaredError<> > J = AddJoiner(1), W = ReplaceWriter(1);
  TreeMemory<double> mem(9, 1, J, W);
  std::vector<std::vector<double>> initMemSTL =
      {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};
  arma::mat initMem = convertToArma(initMemSTL);
  mem.Initialize(initMem);
  for (size_t idx = 0; idx < 31; ++idx)
    BOOST_REQUIRE_EQUAL(mem.Cell(idx).at(0, 0), 0);
  mem.Update(0, arma::mat("1."));
  mem.Update(1, arma::mat("2."));
  mem.Update(8, arma::mat("-3."));
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

BOOST_AUTO_TEST_CASE(BlindHAMUnitTest) {
  size_t nDim = 4, seqLen = 4;
  // Embed model is just an identity function.
  FFN<MeanSquaredError<> > embedModel;
  embedModel.Add<Linear<> >(nDim, nDim);
  embedModel.ResetParameters();
  // Identity = apply identity linear transformation + add zero bias.
  embedModel.Parameters().rows(0, nDim * nDim - 1) =
      arma::vectorise(arma::eye(nDim, nDim));
  embedModel.Parameters().rows(nDim * nDim, nDim * nDim + nDim - 1) =
      arma::zeros(nDim);
  // Join function is sum of its two vector inputs.
  FFN<MeanSquaredError<> > joinModel;
  joinModel.Add<Linear<> >(2 * nDim, nDim);
  joinModel.ResetParameters();
  joinModel.Parameters().rows(0, nDim * nDim - 1) = arma::vectorise(arma::eye(nDim, nDim));
  joinModel.Parameters().rows(nDim * nDim, 2 * nDim * nDim - 1) = arma::vectorise(arma::eye(nDim, nDim));
  joinModel.Parameters().rows(2 * nDim * nDim, 2 * nDim * nDim + nDim - 1) = arma::zeros(nDim);
  // Write function is replacing its old input with its new input.
  FFN<MeanSquaredError<> > writeModel;
  writeModel.Add<Linear<> >(2 * nDim, nDim);
  writeModel.ResetParameters();
  writeModel.Parameters().rows(0, nDim * nDim - 1) = arma::zeros(nDim * nDim);
  writeModel.Parameters().rows(nDim * nDim, 2 * nDim * nDim - 1) =
      arma::vectorise(arma::eye(nDim, nDim));
  writeModel.Parameters().rows(2 * nDim * nDim, 2 * nDim * nDim + nDim - 1) = arma::zeros(nDim);
  // Search model is a constant model that ignores its input and returns 1 / 3.
  FFN<MeanSquaredError<> > searchModel;
  searchModel.Add<Linear<> >(2 * nDim, 1);
  searchModel.ResetParameters();
  searchModel.Parameters().rows(0, 2 * nDim - 1) = arma::zeros(2 * nDim);
  searchModel.Parameters().at(2 * nDim) = -log(2);
  searchModel.Add<SigmoidLayer<> >();

  HAMUnit<> hamUnit(seqLen, nDim, embedModel, joinModel, searchModel, writeModel);

  arma::mat input("1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1;");
  input = input.t();
  arma::mat output;
  hamUnit.Predict(std::move(input), std::move(output));
  std::cerr << "Output (hopefully 0):\n" << output << "\n";
}

BOOST_AUTO_TEST_SUITE_END();
