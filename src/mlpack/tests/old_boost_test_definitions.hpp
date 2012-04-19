/**
 * @file old_boost_test_definitions.hpp
 * @author Ryan Curtin
 *
 * Ancient Boost.Test versions don't act how we expect.  This file includes the
 * things we need to fix that.
 */
#ifndef __MLPACK_TESTS_OLD_BOOST_TEST_DEFINITIONS_HPP
#define __MLPACK_TESTS_OLD_BOOST_TEST_DEFINITIONS_HPP

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

#endif
