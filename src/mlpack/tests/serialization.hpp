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
#ifndef MLPACK_TESTS_SERIALIZATION_HPP
#define MLPACK_TESTS_SERIALIZATION_HPP

#include <boost/serialization/serialization.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

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
  bool success = true;

  {
    OArchiveType o(ofs);

    try
    {
      o << BOOST_SERIALIZATION_NVP(x);
    }
    catch (boost::archive::archive_exception& e)
    {
      success = false;
    }
  }

  BOOST_REQUIRE_EQUAL(success, true);
  ofs.close();

  // Now load it.
  arma::Cube<CubeType> orig(x);
  success = true;
  std::ifstream ifs(fileName, std::ios::binary);

  {
    IArchiveType i(ifs);

    try
    {
      i >> BOOST_SERIALIZATION_NVP(x);
    }
    catch (boost::archive::archive_exception& e)
    {
      success = false;
    }
  }
  ifs.close();

  remove(fileName.c_str());

  BOOST_REQUIRE_EQUAL(success, true);

  BOOST_REQUIRE_EQUAL(x.n_rows, orig.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_cols, orig.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_elem_slice, orig.n_elem_slice);
  BOOST_REQUIRE_EQUAL(x.n_slices, orig.n_slices);
  BOOST_REQUIRE_EQUAL(x.n_elem, orig.n_elem);

  for (size_t slice = 0; slice != x.n_slices; ++slice)
  {
    const auto& origSlice = orig.slice(slice);
    const auto& xSlice = x.slice(slice);
    for (size_t i = 0; i < x.n_cols; ++i)
    {
      for (size_t j = 0; j < x.n_rows; ++j)
      {
        if (double(origSlice(j, i)) == 0.0)
          BOOST_REQUIRE_SMALL(double(xSlice(j, i)), 1e-8);
        else
          BOOST_REQUIRE_CLOSE(double(origSlice(j, i)), double(xSlice(j, i)),
              1e-8);
      }
    }
  }
}

// Test all serialization strategies.
template<typename CubeType>
void TestAllArmadilloSerialization(arma::Cube<CubeType>& x)
{
  TestArmadilloSerialization<CubeType, boost::archive::xml_iarchive,
      boost::archive::xml_oarchive>(x);
  TestArmadilloSerialization<CubeType, boost::archive::text_iarchive,
      boost::archive::text_oarchive>(x);
  TestArmadilloSerialization<CubeType, boost::archive::binary_iarchive,
      boost::archive::binary_oarchive>(x);
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
  bool success = true;

  {
    OArchiveType o(ofs);

    try
    {
      o << BOOST_SERIALIZATION_NVP(x);
    }
    catch (boost::archive::archive_exception& e)
    {
      success = false;
    }
  }

  BOOST_REQUIRE_EQUAL(success, true);
  ofs.close();

  // Now load it.
  MatType orig(x);
  success = true;
  std::ifstream ifs(fileName, std::ios::binary);

  {
    IArchiveType i(ifs);

    try
    {
      i >> BOOST_SERIALIZATION_NVP(x);
    }
    catch (boost::archive::archive_exception& e)
    {
      success = false;
    }
  }
  ifs.close();

  remove(fileName.c_str());

  BOOST_REQUIRE_EQUAL(success, true);

  BOOST_REQUIRE_EQUAL(x.n_rows, orig.n_rows);
  BOOST_REQUIRE_EQUAL(x.n_cols, orig.n_cols);
  BOOST_REQUIRE_EQUAL(x.n_elem, orig.n_elem);

  for (size_t i = 0; i < x.n_cols; ++i)
    for (size_t j = 0; j < x.n_rows; ++j)
      if (double(orig(j, i)) == 0.0)
        BOOST_REQUIRE_SMALL(double(x(j, i)), 1e-8);
      else
        BOOST_REQUIRE_CLOSE(double(orig(j, i)), double(x(j, i)), 1e-8);
}

// Test all serialization strategies.
template<typename MatType>
void TestAllArmadilloSerialization(MatType& x)
{
  TestArmadilloSerialization<MatType, boost::archive::xml_iarchive,
      boost::archive::xml_oarchive>(x);
  TestArmadilloSerialization<MatType, boost::archive::text_iarchive,
      boost::archive::text_oarchive>(x);
  TestArmadilloSerialization<MatType, boost::archive::binary_iarchive,
      boost::archive::binary_oarchive>(x);
}

// Save and load an mlpack object.
// The re-loaded copy is placed in 'newT'.
template<typename T, typename IArchiveType, typename OArchiveType>
void SerializeObject(T& t, T& newT)
{
  std::string fileName = FilterFileName(typeid(T).name());
  std::ofstream ofs(fileName, std::ios::binary);
  bool success = true;

  {
    OArchiveType o(ofs);

    try
    {
      o << BOOST_SERIALIZATION_NVP(t);
    }
    catch (boost::archive::archive_exception& e)
    {
      MLPACK_CERR_STREAM << e.what() << std::endl;
      success = false;
    }
  }
  ofs.close();

  BOOST_REQUIRE_EQUAL(success, true);

  std::ifstream ifs(fileName, std::ios::binary);

  {
    IArchiveType i(ifs);

    try
    {
      i >> BOOST_SERIALIZATION_NVP(newT);
    }
    catch (boost::archive::archive_exception& e)
    {
      MLPACK_COUT_STREAM << e.what() << "\n";
      success = false;
    }
  }
  ifs.close();

  remove(fileName.c_str());

  BOOST_REQUIRE_EQUAL(success, true);
}

// Test mlpack serialization with all three archive types.
template<typename T>
void SerializeObjectAll(T& t, T& xmlT, T& textT, T& binaryT)
{
  SerializeObject<T, boost::archive::xml_iarchive,
      boost::archive::xml_oarchive>(t, xmlT);
  SerializeObject<T, boost::archive::text_iarchive,
      boost::archive::text_oarchive>(t, textT);
  SerializeObject<T, boost::archive::binary_iarchive,
      boost::archive::binary_oarchive>(t, binaryT);
}

// Save and load a non-default-constructible mlpack object.
template<typename T, typename IArchiveType, typename OArchiveType>
void SerializePointerObject(T* t, T*& newT)
{
  std::string fileName = FilterFileName(typeid(T).name());
  std::ofstream ofs(fileName, std::ios::binary);
  bool success = true;

  {
    OArchiveType o(ofs);
    try
    {
      o << BOOST_SERIALIZATION_NVP(t);
    }
    catch (boost::archive::archive_exception& e)
    {
      MLPACK_COUT_STREAM << e.what() << "\n";
      success = false;
    }
  }
  ofs.close();

  BOOST_REQUIRE_EQUAL(success, true);

  std::ifstream ifs(fileName, std::ios::binary);

  {
    IArchiveType i(ifs);

    try
    {
      i >> BOOST_SERIALIZATION_NVP(newT);
    }
    catch (std::exception& e)
    {
      MLPACK_COUT_STREAM << e.what() << "\n";
      success = false;
    }
  }
  ifs.close();

  remove(fileName.c_str());

  BOOST_REQUIRE_EQUAL(success, true);
}

template<typename T>
void SerializePointerObjectAll(T* t, T*& xmlT, T*& textT, T*& binaryT)
{
  SerializePointerObject<T, boost::archive::text_iarchive,
      boost::archive::text_oarchive>(t, textT);
  SerializePointerObject<T, boost::archive::binary_iarchive,
      boost::archive::binary_oarchive>(t, binaryT);
  SerializePointerObject<T, boost::archive::xml_iarchive,
      boost::archive::xml_oarchive>(t, xmlT);
}

// Utility function to check the equality of two Armadillo matrices.
void CheckMatrices(const arma::mat& x,
                   const arma::mat& xmlX,
                   const arma::mat& textX,
                   const arma::mat& binaryX);

void CheckMatrices(const arma::Mat<size_t>& x,
                   const arma::Mat<size_t>& xmlX,
                   const arma::Mat<size_t>& textX,
                   const arma::Mat<size_t>& binaryX);

void CheckMatrices(const arma::cube& x,
                   const arma::cube& xmlX,
                   const arma::cube& textX,
                   const arma::cube& binaryX);

} // namespace mlpack

#endif
