/**
 * @file timer_test.cpp
 * @author Matthew Amidon, Ryan Curtin
 *
 * Test for the timer class
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef _WIN32
  #include <sys/time.h>
#endif

// For Sleep().
#ifdef _WIN32
  #include <windows.h>
#endif

#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;

BOOST_AUTO_TEST_SUITE(TimerTest);

/**
 * We should be able to start and then stop a timer multiple times and it should
 * save the value.
 */
BOOST_AUTO_TEST_CASE(MultiRunTimerTest)
{
  Timer::Start("test_timer");

  // On Windows (or, at least, in Windows not using VS2010) we cannot use
  // usleep() because it is not provided.  Instead we will use Sleep() for a
  // number of milliseconds.
  #ifdef _WIN32
  Sleep(10);
  #else
  usleep(10000);
  #endif

  Timer::Stop("test_timer");

  BOOST_REQUIRE_GE(Timer::Get("test_timer").count(), 10000);

  // Restart it.
  Timer::Start("test_timer");

  #ifdef _WIN32
  Sleep(10);
  #else
  usleep(10000);
  #endif

  Timer::Stop("test_timer");

  BOOST_REQUIRE_GE(Timer::Get("test_timer").count(), 20000);

  // Just one more time, for good measure...
  Timer::Start("test_timer");

  #ifdef _WIN32
  Sleep(20);
  #else
  usleep(20000);
  #endif

  Timer::Stop("test_timer");

  BOOST_REQUIRE_GE(Timer::Get("test_timer").count(), 40000);
}

BOOST_AUTO_TEST_CASE(TwiceStopTimerTest)
{
  Timer::Start("test_timer");
  Timer::Stop("test_timer");

  BOOST_REQUIRE_THROW(Timer::Stop("test_timer"), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(TwiceStartTimerTest)
{
  Timer::Start("test_timer");

  BOOST_REQUIRE_THROW(Timer::Start("test_timer"), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END();
