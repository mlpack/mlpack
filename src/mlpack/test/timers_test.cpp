/**
 * @file timers_test.cpp
 * @author Grzegorz Krajewski
 *
 * Test for the Timers. Duplicate start and stop.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <boost/test/unit_test.hpp>
#include <mlpack/core.hpp>

using namespace mlpack;

BOOST_AUTO_TEST_SUITE(TimersTest);

BOOST_AUTO_TEST_CASE(DuplicateStartTest1)
{
  Timer::Start("test_timer");
  
  BOOST_REQUIRE_THROW(Timer::Start("test_timer"), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(DuplicateStopTest1)
{
  Timer::Stop("test_timer");
  
  BOOST_REQUIRE_THROW(Timer::Stop("test_timer"), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(DuplicateStartTest2)
{
  Timer::Start("test_timer1");
  Timer::Start("test_timer2");
  
  BOOST_REQUIRE_THROW(Timer::Start("test_timer1"), std::runtime_error);
  BOOST_REQUIRE_THROW(Timer::Start("test_timer2"), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(DuplicateStopTest2)
{
  Timer::Stop("test_timer1");
  Timer::Stop("test_timer2");
  
  BOOST_REQUIRE_THROW(Timer::Stop("test_timer1"), std::runtime_error);
  BOOST_REQUIRE_THROW(Timer::Stop("test_timer2"), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END();
