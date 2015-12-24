/**
 * @file serialization.hpp
 * @author Ryan Curtin
 *
 * Miscellaneous utility functions for serialization tests.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_TESTS_SERIALIZATION_HPP
#define __MLPACK_TESTS_SERIALIZATION_HPP

#include <boost/serialization/serialization.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

namespace mlpack {

// Test function for loading and saving Armadillo objects.
template<typename MatType,
         typename IArchiveType,
         typename OArchiveType>
void TestArmadilloSerialization(MatType& x)
{
  // First save it.
  std::ofstream ofs("test", std::ios::binary);
  OArchiveType o(ofs);

  bool success = true;
  try
  {
    o << BOOST_SERIALIZATION_NVP(x);
  }
  catch (boost::archive::archive_exception& e)
  {
    success = false;
  }

  BOOST_REQUIRE_EQUAL(success, true);
  ofs.close();

  // Now load it.
  MatType orig(x);
  success = true;
  std::ifstream ifs("test", std::ios::binary);
  IArchiveType i(ifs);

  try
  {
    i >> BOOST_SERIALIZATION_NVP(x);
  }
  catch (boost::archive::archive_exception& e)
  {
    success = false;
  }

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

  remove("test");
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
  std::ofstream ofs("test", std::ios::binary);
  OArchiveType o(ofs);

  bool success = true;
  try
  {
    o << data::CreateNVP(t, "t");
  }
  catch (boost::archive::archive_exception& e)
  {
    success = false;
  }
  ofs.close();

  BOOST_REQUIRE_EQUAL(success, true);

  std::ifstream ifs("test", std::ios::binary);
  IArchiveType i(ifs);

  try
  {
    i >> data::CreateNVP(newT, "t");
  }
  catch (boost::archive::archive_exception& e)
  {
    success = false;
  }
  ifs.close();

  BOOST_REQUIRE_EQUAL(success, true);
}

// Test mlpack serialization with all three archive types.
template<typename T>
void SerializeObjectAll(T& t, T& xmlT, T& textT, T& binaryT)
{
  SerializeObject<T, boost::archive::text_iarchive,
      boost::archive::text_oarchive>(t, textT);
  SerializeObject<T, boost::archive::binary_iarchive,
      boost::archive::binary_oarchive>(t, binaryT);
  SerializeObject<T, boost::archive::xml_iarchive,
      boost::archive::xml_oarchive>(t, xmlT);
}

// Save and load a non-default-constructible mlpack object.
template<typename T, typename IArchiveType, typename OArchiveType>
void SerializePointerObject(T* t, T*& newT)
{
  std::ofstream ofs("test", std::ios::binary);
  OArchiveType o(ofs);

  bool success = true;
  try
  {
    o << data::CreateNVP(*t, "t");
  }
  catch (boost::archive::archive_exception& e)
  {
    success = false;
  }
  ofs.close();

  BOOST_REQUIRE_EQUAL(success, true);

  std::ifstream ifs("test", std::ios::binary);
  IArchiveType i(ifs);

  try
  {
    newT = new T(i);
  }
  catch (std::exception& e)
  {
    success = false;
  }
  ifs.close();

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

} // namespace mlpack

#endif
