/**
 * @file timers.hpp
 * @author Matthew Amidon
 * @author Marcus Edel
 *
 * Timers for MLPACK.
 *
 * This file is part of MLPACK 1.0.10.
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
#ifndef __MLPACK_CORE_UTILITIES_TIMERS_HPP
#define __MLPACK_CORE_UTILITIES_TIMERS_HPP

#include <map>
#include <string>

#if defined(__unix__) || defined(__unix)
  #include <time.h>       // clock_gettime()
  #include <sys/time.h>   // timeval, gettimeofday()
  #include <unistd.h>     // flags like  _POSIX_VERSION
#elif defined(__MACH__) && defined(__APPLE__)
  #include <mach/mach_time.h>   // mach_timebase_info,
                                // mach_absolute_time()

  // TEMPORARY
  #include <time.h>       // clock_gettime()
  #include <sys/time.h>   // timeval, gettimeofday()
  #include <unistd.h>     // flags like  _POSIX_VERSION
#elif defined(_WIN32)
  #include <windows.h>  //GetSystemTimeAsFileTime(),
                        // QueryPerformanceFrequency(),
                        // QueryPerformanceCounter()
  #include <winsock.h>  //timeval on windows

  // uint64_t isn't defined on every windows.
  #if !defined(HAVE_UINT64_T)
    #if SIZEOF_UNSIGNED_LONG == 8
      typedef unsigned long uint64_t;
    #else
      typedef unsigned long long  uint64_t;
    #endif  // SIZEOF_UNSIGNED_LONG
  #endif  // HAVE_UINT64_T

  //gettimeofday has no equivalent will need to write extra code for that.
  #if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
    #define DELTA_EPOCH_IN_MICROSECS 11644473600000000Ui64
  #else
    #define DELTA_EPOCH_IN_MICROSECS 11644473600000000ULL
  #endif // _MSC_VER, _MSC_EXTENSIONS
#else
  #error "unknown OS"
#endif

namespace mlpack {

/**
 * The timer class provides a way for MLPACK methods to be timed.  The three
 * methods contained in this class allow a named timer to be started and
 * stopped, and its value to be obtained.
 */
class Timer
{
 public:
  /**
   * Start the given timer.  If a timer is started, then stopped, then
   * re-started, then re-stopped, the final value of the timer is the length of
   * both runs -- that is, MLPACK timers are additive for each time they are
   * run, and do not reset.
   *
   * @note Undefined behavior will occur if a timer is started twice.
   *
   * @param name Name of timer to be started.
   */
  static void Start(const std::string& name);

  /**
   * Stop the given timer.
   *
   * @note Undefined behavior will occur if a timer is started twice.
   *
   * @param name Name of timer to be stopped.
   */
  static void Stop(const std::string& name);

  /**
   * Get the value of the given timer.
   *
   * @param name Name of timer to return value of.
   */
  static timeval Get(const std::string& name);
};

class Timers
{
 public:
  //! Nothing to do for the constructor.
  Timers() { }

  /**
   * Returns a copy of all the timers used via this interface.
   */
  std::map<std::string, timeval>& GetAllTimers();

  /**
   * Returns a copy of the timer specified.
   *
   * @param timerName The name of the timer in question.
   */
  timeval GetTimer(const std::string& timerName);

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

 private:
  std::map<std::string, timeval> timers;

  void FileTimeToTimeVal(timeval* tv);
  void GetTime(timeval* tv);
};

}; // namespace mlpack

#endif // __MLPACK_CORE_UTILITIES_TIMERS_HPP
