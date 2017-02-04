/**
 * @file timers.hpp
 * @author Matthew Amidon
 * @author Marcus Edel
 *
 * Timers for MLPACK.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTILITIES_TIMERS_HPP
#define MLPACK_CORE_UTILITIES_TIMERS_HPP

#include <map>
#include <string>
#include <chrono> // chrono library for cross platform timer calculation

#if defined(_WIN32)
 // uint64_t isn't defined on every windows.
  #if !defined(HAVE_UINT64_T)
    #if SIZEOF_UNSIGNED_LONG == 8
      typedef unsigned long uint64_t;
    #else
      typedef unsigned long long  uint64_t;
    #endif  // SIZEOF_UNSIGNED_LONG
  #endif  // HAVE_UINT64_T
#endif

namespace mlpack {

/**
 * The timer class provides a way for mlpack methods to be timed.  The three
 * methods contained in this class allow a named timer to be started and
 * stopped, and its value to be obtained.
 */
class Timer
{
 public:
  /**
   * Start the given timer.  If a timer is started, then stopped, then
   * re-started, then re-stopped, the final value of the timer is the length of
   * both runs -- that is, mlpack timers are additive for each time they are
   * run, and do not reset.
   *
   * @note A std::runtime_error exception will be thrown if a timer is started
   * twice.
   *
   * @param name Name of timer to be started.
   */
  static void Start(const std::string& name);

  /**
   * Stop the given timer.
   *
   * @note A std::runtime_error exception will be thrown if a timer is started
   * twice.
   *
   * @param name Name of timer to be stopped.
   */
  static void Stop(const std::string& name);

  /**
   * Get the value of the given timer.
   *
   * @param name Name of timer to return value of.
   */
  static std::chrono::microseconds Get(const std::string& name);
};

class Timers
{
 public:
  //! Nothing to do for the constructor.
  Timers() { }

  /**
   * Returns a copy of all the timers used via this interface.
   */
  std::map<std::string, std::chrono::microseconds>& GetAllTimers();

  /**
   * Returns a copy of the timer specified.
   *
   * @param timerName The name of the timer in question.
   */
  std::chrono::microseconds GetTimer(const std::string& timerName);

  /**
   * Prints the specified timer.  If it took longer than a minute to complete
   * the timer will be displayed in days, hours, and minutes as well.
   *
   * @param timerName The name of the timer in question.
   */
  void PrintTimer(const std::string& timerName);

  /**
   * Initializes a timer, available like a normal value specified on
   * the command line.  Timers are of type timeval.  If a timer is started, then
   * stopped, then re-started, then stopped, the final timer value will be the
   * length of both runs of the timer.
   *
   * @param timerName The name of the timer in question.
   */
  void StartTimer(const std::string& timerName);

  /**
   * Halts the timer, and replaces it's value with
   * the delta time from it's start
   *
   * @param timerName The name of the timer in question.
   */
  void StopTimer(const std::string& timerName);

  /**
   * Returns state of the given timer.
   *
   * @param timerName The name of the timer in question.
   */
  bool GetState(std::string timerName);

 private:
  //! A map of all the timers that are being tracked.
  std::map<std::string, std::chrono::microseconds> timers;
  //! A map that contains whether or not each timer is currently running.
  std::map<std::string, bool> timerState;
  //! A map for the starting values of the timers.
  std::map<std::string, std::chrono::high_resolution_clock::time_point>
      timerStartTime;

  std::chrono::high_resolution_clock::time_point GetTime();
};

} // namespace mlpack

#endif // MLPACK_CORE_UTILITIES_TIMERS_HPP
