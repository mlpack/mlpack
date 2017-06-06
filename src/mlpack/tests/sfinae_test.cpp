/**
 * @file sfinae_test.cpp
 * @author Kirill Mishchenko
 *
 * Test file for SFINAE utilities.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/util/sfinae_utility.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(SFINAETest);

class A
{
 public:
  void M(const arma::mat&, const arma::Row<size_t>&, double);
  void M(const arma::mat&, const arma::Row<size_t>&, double, double);
  void M(const arma::vec&, size_t, double);
};

class B
{
 public:
  void M(const arma::mat&, const arma::rowvec&);
};

template<typename Class, typename...T>
using MForm1 =
    void(Class::*)(const arma::mat&, const arma::Row<size_t>&, T...);

template<typename Class, typename...T>
using MForm2 = void(Class::*)(const arma::mat&, const arma::rowvec&, T...);

HAS_METHOD_FORM(M, HasM);

BOOST_AUTO_TEST_CASE(HasMethodFormTest)
{
  static_assert(!HasM<A, MForm1, 0>::value, "value should be false");
  static_assert(HasM<A, MForm1, 1>::value, "value should be true");
  static_assert(HasM<A, MForm1, 2>::value, "value should be true");

  static_assert(!HasM<B, MForm1, 0>::value, "value should be false");
  static_assert(!HasM<B, MForm1, 1>::value, "value should be false");
  static_assert(!HasM<B, MForm1, 2>::value, "value should be false");

  static_assert(!HasM<A, MForm2, 0>::value, "value should be false");
  static_assert(!HasM<A, MForm2, 1>::value, "value should be false");
  static_assert(!HasM<A, MForm2, 2>::value, "value should be false");

  static_assert(HasM<B, MForm2, 0>::value, "value should be true");
  static_assert(!HasM<B, MForm2, 1>::value, "value should be false");
  static_assert(!HasM<B, MForm2, 2>::value, "value should be false");
}

BOOST_AUTO_TEST_SUITE_END();
