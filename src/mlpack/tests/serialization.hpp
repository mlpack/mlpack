/**
 * @file tests/serialization.hpp
 * @author Ryan Curtin
 *
 * Miscellaneous utility functions for serialization tests.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_TESTS_SERIALIZATION_CATCH_HPP
#define MLPACK_TESTS_SERIALIZATION_CATCH_HPP

#include <mlpack/core.hpp>

#include "test_catch_tools.hpp"
#include "catch.hpp"

namespace mlpack {

// Test function for loading and saving Armadillo objects.
template<typename CubeType,
         typename IArchiveType,
         typename OArchiveType>
void TestArmadilloSerialization(arma::Cube<CubeType>& x)
{
  // First save it.
  // Use type_info name to get unique file name for serialization test files.
  std::string fileName = FilterFileName(typeid(IArchiveType).name());
  std::ofstream ofs(fileName, std::ios::binary);

  {
    OArchiveType o(ofs);
    o(CEREAL_NVP(x));
  }

  ofs.close();

  // Now load it.
  arma::Cube<CubeType> orig(x);
  std::ifstream ifs(fileName, std::ios::binary);

  {
    IArchiveType i(ifs);
    i(CEREAL_NVP(x));
  }
  ifs.close();

  remove(fileName.c_str());

  REQUIRE(x.n_rows == orig.n_rows);
  REQUIRE(x.n_cols == orig.n_cols);
  REQUIRE(x.n_elem_slice == orig.n_elem_slice);
  REQUIRE(x.n_slices == orig.n_slices);
  REQUIRE(x.n_elem == orig.n_elem);

  for (size_t slice = 0; slice != x.n_slices; ++slice)
  {
    const auto& origSlice = orig.slice(slice);
    const auto& xSlice = x.slice(slice);
    for (size_t i = 0; i < x.n_cols; ++i)
    {
      for (size_t j = 0; j < x.n_rows; ++j)
      {
        if (double(origSlice(j, i)) == 0.0)
          REQUIRE(double(xSlice(j, i)) == Approx(0.0).margin(1e-8 / 100));
        else
          REQUIRE(double(origSlice(j, i)) ==
              Approx(double(xSlice(j, i))).epsilon(1e-8 / 100));
      }
    }
  }
}

// Test all serialization strategies.
template<typename CubeType>
void TestAllArmadilloSerialization(arma::Cube<CubeType>& x)
{
  TestArmadilloSerialization<CubeType, cereal::XMLInputArchive,
      cereal::XMLOutputArchive>(x);
  TestArmadilloSerialization<CubeType, cereal::JSONInputArchive,
      cereal::JSONOutputArchive>(x);
  TestArmadilloSerialization<CubeType, cereal::BinaryInputArchive,
      cereal::BinaryOutputArchive>(x);
}

// Test function for loading and saving Armadillo objects.
template<typename MatType,
         typename IArchiveType,
         typename OArchiveType>
void TestArmadilloSerialization(MatType& x)
{
  // First save it.
  std::string fileName = FilterFileName(typeid(IArchiveType).name());
  std::ofstream ofs(fileName, std::ios::binary);

  {
    OArchiveType o(ofs);
    o(CEREAL_NVP(x));
  }

  ofs.close();

  // Now load it.
  MatType orig(x);
  std::ifstream ifs(fileName, std::ios::binary);

  {
    IArchiveType i(ifs);
    i(CEREAL_NVP(x));
  }
  ifs.close();

  remove(fileName.c_str());

  REQUIRE(x.n_rows == orig.n_rows);
  REQUIRE(x.n_cols == orig.n_cols);
  REQUIRE(x.n_elem == orig.n_elem);

  for (size_t i = 0; i < x.n_cols; ++i)
    for (size_t j = 0; j < x.n_rows; ++j)
      if (double(orig(j, i)) == 0.0)
        REQUIRE(double(x(j, i)) == Approx(0.0).margin(1e-8 / 100));
      else
          REQUIRE(double(orig(j, i)) ==
              Approx(double(x(j, i))).epsilon(1e-8 / 100));
}

// Test all serialization strategies.
template<typename MatType>
void TestAllArmadilloSerialization(MatType& x)
{
  TestArmadilloSerialization<MatType, cereal::XMLInputArchive,
      cereal::XMLOutputArchive>(x);
  TestArmadilloSerialization<MatType, cereal::JSONInputArchive,
      cereal::JSONOutputArchive>(x);
  TestArmadilloSerialization<MatType, cereal::BinaryInputArchive,
      cereal::BinaryOutputArchive>(x);
}

// Save and load an mlpack object.
// The re-loaded copy is placed in 'newT'.
template<typename T, typename IArchiveType, typename OArchiveType>
void SerializeObject(T& t, T& newT)
{
  std::string fileName = FilterFileName(typeid(T).name());
  std::ofstream ofs(fileName, std::ios::binary);

  {
    OArchiveType o(ofs);

    T& x(t);
    o(CEREAL_NVP(x));
  }
  ofs.close();

  std::ifstream ifs(fileName, std::ios::binary);

  {
    IArchiveType i(ifs);
    T& x(newT);
    i(CEREAL_NVP(x));
  }
  ifs.close();

  remove(fileName.c_str());
}

// Test mlpack serialization with all three archive types.
template<typename T>
void SerializeObjectAll(T& t, T& xmlT, T& jsonT, T& binaryT)
{
  SerializeObject<T, cereal::XMLInputArchive,
      cereal::XMLOutputArchive>(t, xmlT);
  SerializeObject<T, cereal::JSONInputArchive,
      cereal::JSONOutputArchive>(t, jsonT);
  SerializeObject<T, cereal::BinaryInputArchive,
      cereal::BinaryOutputArchive>(t, binaryT);
}

// Save and load a non-default-constructible mlpack object.
template<typename T, typename IArchiveType, typename OArchiveType>
void SerializePointerObject(T* t, T*& newT)
{
  std::string fileName = FilterFileName(typeid(T).name());
  std::ofstream ofs(fileName, std::ios::binary);

  {
    OArchiveType o(ofs);
    o(CEREAL_POINTER(t));
  }
  ofs.close();

  std::ifstream ifs(fileName, std::ios::binary);

  {
    IArchiveType i(ifs);
    i(CEREAL_POINTER(newT));
  }
  ifs.close();
  remove(fileName.c_str());
}

template<typename T>
void SerializePointerObjectAll(T* t, T*& xmlT, T*& jsonT, T*& binaryT)
{
  SerializePointerObject<T, cereal::JSONInputArchive,
      cereal::JSONOutputArchive>(t, jsonT);
  SerializePointerObject<T, cereal::BinaryInputArchive,
      cereal::BinaryOutputArchive>(t, binaryT);
  SerializePointerObject<T, cereal::XMLInputArchive,
      cereal::XMLOutputArchive>(t, xmlT);
}

// Utility function to check the equality of two Armadillo matrices.
void CheckMatrices(const arma::mat& x,
                   const arma::mat& xmlX,
                   const arma::mat& jsonX,
                   const arma::mat& binaryX);

void CheckMatrices(const arma::Mat<size_t>& x,
                   const arma::Mat<size_t>& xmlX,
                   const arma::Mat<size_t>& jsonX,
                   const arma::Mat<size_t>& binaryX);

void CheckMatrices(const arma::cube& x,
                   const arma::cube& xmlX,
                   const arma::cube& jsonX,
                   const arma::cube& binaryX);

} // namespace mlpack

#endif
