/**
 * @file tests/timer_test.cpp
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

#include "catch.hpp"

using namespace mlpack;

/**
 * We should be able to start and then stop a timer multiple times and it should
 * save the value.
 */
TEST_CASE("MultiRunTimerTest", "[TimerTest]")
{
  Timer::EnableTiming();
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

  REQUIRE(Timer::Get("test_timer").count() >= 10000);

  // Restart it.
  Timer::Start("test_timer");

  #ifdef _WIN32
  Sleep(10);
  #else
  usleep(10000);
  #endif

  Timer::Stop("test_timer");

  REQUIRE(Timer::Get("test_timer").count() >= 20000);

  // Just one more time, for good measure...
  Timer::Start("test_timer");

  #ifdef _WIN32
  Sleep(20);
  #else
  usleep(20000);
  #endif

  Timer::Stop("test_timer");

  REQUIRE(Timer::Get("test_timer").count() >= 40000);
  Timer::DisableTiming();
}

TEST_CASE("TwiceStopTimerTest", "[TimerTest]")
{
  Timer::EnableTiming();
  Timer::Start("test_timer");
  Timer::Stop("test_timer");

  REQUIRE_THROWS_AS(Timer::Stop("test_timer"), std::runtime_error);

  Timer::DisableTiming();
}

TEST_CASE("TwiceStartTimerTest", "[TimerTest]")
{
  Timer::EnableTiming();
  Timer::Start("test_timer");

  REQUIRE_THROWS_AS(Timer::Start("test_timer"), std::runtime_error);
  Timer::Stop("test_timer");
  Timer::DisableTiming();
}

TEST_CASE("MultithreadTimerTest", "[TimerTest]")
{
  Timer::EnableTiming();
  // Make three different threads all start a timer then stop a timer.
  std::thread threads[3];
  for (size_t i = 0; i < 3; ++i)
  {
    threads[i] = std::thread([]()
        {
          Timer::Start("thread_timer");

          #ifdef _WIN32
          Sleep(20);
          #else
          int restarts = 0;
          // Catch occasional EINTR failures.
          while (usleep(20000) != 0 && restarts < 3)
            ++restarts;
          #endif

          Timer::Stop("thread_timer");
        });
  }

  for (size_t i = 0; i < 3; ++i)
    threads[i].join();

  // If we made it this far without a problem, then the multithreaded part has
  // worked.  Next we ensure that the total timer time is counting multiple
  // threads.
  REQUIRE(Timer::Get("thread_timer") > std::chrono::microseconds(50000));
}

TEST_CASE("DisabledTimingTest", "[TimerTest]")
{
  // It should be disabled by default but let's be paranoid.
  Timer::DisableTiming();

  Timer::Start("test_timer");
  #ifdef _WIN32
  Sleep(20);
  #else
  usleep(20000);
  #endif
  Timer::Stop("test_timer");

  REQUIRE(Timer::Get("test_timer") == std::chrono::microseconds(0));
}
