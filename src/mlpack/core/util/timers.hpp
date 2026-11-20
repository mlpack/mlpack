/**
 * @file core/util/timers.hpp
 * @author Matthew Amidon
 * @author Marcus Edel
 * @author Ryan Curtin
 *
 * Timers for mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTILITIES_TIMERS_HPP
#define MLPACK_CORE_UTILITIES_TIMERS_HPP

#include <atomic>
#include <chrono> // chrono library for cross platform timer calculation.
#include <iomanip>
#include <list>
#include <map>
#ifndef MLPACK_NO_STD_MUTEX
#include <mutex>
#endif
#include <string>
#include <thread> // std::thread is used for thread safety.

#if defined(_WIN32)
  // uint64_t isn't defined on every windows.
  #if !defined(HAVE_UINT64_T)
    #if SIZEOF_UNSIGNED_LONG == 8
      using uint64_t = unsigned long;
    #else
      using uint64_t = unsigned long long;
    #endif // SIZEOF_UNSIGNED_LONG
  #endif // HAVE_UINT64_T
#endif

namespace mlpack {

/**
 * The timer class provides a way for mlpack methods to be timed.  The three
 * methods contained in this class allow a named timer to be started and
 * stopped, and its value to be obtained.  A named timer is specific to the
 * thread it is running on, so if you start a timer in one thread, it cannot be
 * stopped from a different thread.
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

  /**
   * Enable timing of mlpack programs.  Do not run this while timers are
   * running!
   */
  static void EnableTiming();

  /**
   * Disable timing of mlpack programs.  Do not run this while timers are
   * running!
   */
  static void DisableTiming();

  /**
   * Stop and reset all running timers.  This removes all knowledge of any
   * existing timers.
   */
  static void ResetAll();

  /**
   * Returns a copy of all the timers used via this interface.
   */
  static std::map<std::string, std::chrono::microseconds> GetAllTimers();
};

namespace util {

class Timers
{
 public:
  //! Default to disabled.
  Timers() : enabled(false) { }

  /**
   * Returns a copy of all the timers used via this interface.
   */
  std::map<std::string, std::chrono::microseconds> GetAllTimers();

  /**
   * Reset the timers.  This stops all running timers and removes them.  Whether
   * or not timing is enabled will not be changed.
   */
  void Reset();

  /**
   * Returns a copy of the timer specified.  This contains the sum of the timing
   * results for timers that have been stopped with this name.
   *
   * @param timerName The name of the timer in question.
   */
  std::chrono::microseconds Get(const std::string& timerName);

  /**
   * Prints the specified timer.  If it took longer than a minute to complete
   * the timer will be displayed in days, hours, and minutes as well.
   *
   * @param timerName The number of microseconds to print.
   */
  static std::string Print(const std::chrono::microseconds& totalDuration);

  /**
   * Initializes a timer, available like a normal value specified on
   * the command line.  Timers are of type timeval.  If a timer is started, then
   * stopped, then re-started, then stopped, the final timer value will be the
   * length of both runs of the timer.
   *
   * @param timerName The name of the timer in question.
   * @param threadId Id of the thread accessing the timer.
   */
  void Start(const std::string& timerName,
             const std::thread::id& threadId = std::thread::id());

  /**
   * Halts the timer, and replaces its value with the delta time from its start.
   *
   * @param timerName The name of the timer in question.
   * @param threadId Id of the thread accessing the timer.
   */
  void Stop(const std::string& timerName,
            const std::thread::id& threadId = std::thread::id());

  /**
   * Stop all timers.
   */
  void StopAllTimers();

  //! Modify whether or not timing is enabled.
  std::atomic<bool>& Enabled() { return enabled; }
  //! Get whether or not timing is enabled.
  bool Enabled() const { return enabled; }

 private:
  //! A map of all the timers that are being tracked.
  std::map<std::string, std::chrono::microseconds> timers;
#ifndef MLPACK_NO_STD_MUTEX
  //! A mutex for modifying the timers.
  std::mutex timersMutex;
#endif
  //! A map for the starting values of the timers.
  std::map<std::thread::id, std::map<std::string,
      std::chrono::high_resolution_clock::time_point>> timerStartTime;

  //! Whether or not timing is enabled.
  std::atomic<bool> enabled;
};

} // namespace util
} // namespace mlpack

// Note that the implementation is not included, to avoid include ordering
// issues!

#endif // MLPACK_CORE_UTILITIES_TIMERS_HPP
