/**
 * @file test_tools.hpp
 * @author Ryan Curtin
 *
 * This file includes some useful macros for tests.
 */
#ifndef MLPACK_TESTS_TEST_TOOLS_HPP
#define MLPACK_TESTS_TEST_TOOLS_HPP

#include <boost/version.hpp>

// This is only necessary for pre-1.36 Boost.Test.
#if BOOST_VERSION < 103600

#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/auto_unit_test.hpp>

// This depends on other macros.  Probably not a great idea... but it works, and
// we only need it for ancient Boost versions.
#define BOOST_REQUIRE_GE( L, R ) \
    BOOST_REQUIRE_EQUAL( (L >= R), true )

#define BOOST_REQUIRE_NE( L, R ) \
    BOOST_REQUIRE_EQUAL( (L != R), true )

#define BOOST_REQUIRE_LE( L, R ) \
    BOOST_REQUIRE_EQUAL( (L <= R), true )

#define BOOST_REQUIRE_LT( L, R ) \
    BOOST_REQUIRE_EQUAL( (L < R), true )

#define BOOST_REQUIRE_GT( L, R ) \
    BOOST_REQUIRE_EQUAL( (L > R), true )

#endif

// Require the approximation L to be within a relative error of E respect to the
// actual value R.
#define REQUIRE_RELATIVE_ERR( L, R, E ) \
    BOOST_REQUIRE_LE( std::abs((R) - (L)), (E) * std::abs(R))

#endif
