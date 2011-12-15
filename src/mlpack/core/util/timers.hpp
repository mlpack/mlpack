/**
 * @file timers.hpp
 * @author Matthew Amidon
 *
 * Timers for MLPACK.
 */
#ifndef __MLPACK_CORE_UTILITIES_TIMERS_HPP
#define __MLPACK_CORE_UTILITIES_TIMERS_HPP

#include <map>
#include <string>

#ifndef _WIN32
  #include <sys/time.h> //linux
#else
  #include <winsock.h> //timeval on windows
  #include <windows.h> //GetSystemTimeAsFileTime on windows
//gettimeofday has no equivalent will need to write extra code for that.
  #if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
    #define DELTA_EPOCH_IN_MICROSECS 11644473600000000Ui64
  #else
    #define DELTA_EPOCH_IN_MICROSECS 11644473600000000ULL
  #endif
#endif //_WIN32

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
   * Start the given timer.
   *
   * @param name Name of timer to be started.
   */
  static void Start(const std::string name);

  /**
   * Stop the given timer.
   *
   * @param name Name of timer to be stopped.
   */
  static void Stop(const std::string name);

  /**
   * Get the value of the given timer.
   *
   * @param name Name of timer to return value of.
   */
  static timeval Get(const std::string name);
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
  timeval GetTimer(const std::string timerName);

  /**
   * Prints the specified timer.  If it took longer than a minute to complete
   * the timer will be displayed in days, hours, and minutes as well.
   *
   * @param timerName The name of the timer in question.
   */
  void PrintTimer(const std::string timerName);

  /**
   * Initializes a timer, available like a normal value specified on
   * the command line.  Timers are of type timval
   *
   * @param timerName The name of the timer in question.
   */
  void StartTimer(const std::string timerName);

  /**
   * Halts the timer, and replaces it's value with
   * the delta time from it's start
   *
   * @param timerName The name of the timer in question.
   */
  void StopTimer(const std::string timerName);

 private:
  std::map<std::string, timeval> timers;

  void FileTimeToTimeVal(timeval* tv);
};

}; // namespace mlpack

#endif // __MLPACK_CORE_UTILITIES_TIMERS_HPP
