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

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack::ann::augmented;

template<typename T>
struct ReplaceWriter
{
  arma::Col<T> operator() (arma::Col<T> a, arma::Col<T> b) { return b; }
};

template<typename T>
struct AddJoiner
{
  arma::Col<T> operator() (arma::Col<T> a, arma::Col<T> b) { return a + b; }
};

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
  AddJoiner<double> J;
  ReplaceWriter<double> W;
  // With these definitions, mem is exactly one of the well-known data structres
  // in competitive programming - segment tree.
  TreeMemory<double, AddJoiner<double>, ReplaceWriter<double>> mem(1, 1, J, W);
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
  AddJoiner<double> J;
  ReplaceWriter<double> W;
  size_t memSize = 5;
  TreeMemory<double, AddJoiner<double>, ReplaceWriter<double>> mem(
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
  AddJoiner<double> J;
  ReplaceWriter<double> W;
  TreeMemory<double, AddJoiner<double>, ReplaceWriter<double>> mem(8, 1, J, W);
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
  AddJoiner<double> J;
  ReplaceWriter<double> W;
  size_t memSize = 4;
  TreeMemory<double, AddJoiner<double>, ReplaceWriter<double>> mem(
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
  AddJoiner<double> J;
  ReplaceWriter<double> W;
  TreeMemory<double, AddJoiner<double>, ReplaceWriter<double>> mem(9, 1, J, W);
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

BOOST_AUTO_TEST_SUITE_END();
