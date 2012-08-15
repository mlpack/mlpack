/**
 * @file mlpack_test.cpp
 *
 * Simple file defining the name of the overall test for MLPACK.  Each
 * individual test is contained in its own file.
 * This file is part of MLPACK 1.0.2.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#define BOOST_TEST_MODULE MLPACKTest

#include <boost/version.hpp>

// We only need to do this for old Boost versions.
#if BOOST_VERSION < 103600
  #define BOOST_AUTO_TEST_MAIN
#endif

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"
