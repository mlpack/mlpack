/**
 * @file core/util/timers_impl.hpp
 * @author Matthew Amidon
 * @author Marcus Edel
 * @author Ryan Curtin
 *
 * Implementation of timers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "timers.hpp"

#include "forward.hpp"
#include "io.hpp"
#include "log.hpp"

#include <map>
#include <string>

namespace mlpack {

/**
 * Start the given timer.
 */
inline void Timer::Start(const std::string& name)
{
  IO::GetSingleton().timer.Start(name, std::this_thread::get_id());
}

/**
 * Stop the given timer.
 */
inline void Timer::Stop(const std::string& name)
{
  IO::GetSingleton().timer.Stop(name, std::this_thread::get_id());
}

/**
 * Get the given timer, summing over all threads.
 */
inline std::chrono::microseconds Timer::Get(const std::string& name)
{
  return IO::GetSingleton().timer.Get(name);
}

// Enable timing.
inline void Timer::EnableTiming()
{
  IO::GetSingleton().timer.Enabled() = true;
}

// Disable timing.
inline void Timer::DisableTiming()
{
  IO::GetSingleton().timer.Enabled() = false;
}

// Reset all timers.  Save state of enabled.
inline void Timer::ResetAll()
{
  IO::GetSingleton().timer.Reset();
}

inline std::map<std::string, std::chrono::microseconds> Timer::GetAllTimers()
{
  return IO::GetSingleton().timer.GetAllTimers();
}

namespace util {

// Reset a Timers object.
inline void Timers::Reset()
{
#ifndef MLPACK_NO_STD_MUTEX
  std::lock_guard<std::mutex> lock(timersMutex);
#endif
  timers.clear();
  timerStartTime.clear();
}

inline std::map<std::string, std::chrono::microseconds> Timers::GetAllTimers()
{
#ifndef MLPACK_NO_STD_MUTEX
  // Make a copy of the timer.
  std::lock_guard<std::mutex> lock(timersMutex);
#endif
  return timers;
}

inline std::chrono::microseconds Timers::Get(const std::string& timerName)
{
  if (!enabled)
    return std::chrono::microseconds(0);

#ifndef MLPACK_NO_STD_MUTEX
  std::lock_guard<std::mutex> lock(timersMutex);
#endif
  return timers[timerName];
}

inline std::string Timers::Print(const std::chrono::microseconds& totalDuration)
{
  // Convert microseconds to seconds.
  std::chrono::seconds totalDurationSec =
      std::chrono::duration_cast<std::chrono::seconds>(totalDuration);
  std::chrono::microseconds totalDurationMicroSec =
      std::chrono::duration_cast<std::chrono::microseconds>(
      totalDuration % std::chrono::seconds(1));

  std::ostringstream oss;
  oss << totalDurationSec.count() << "." << std::setw(6)
      << std::setfill('0') << totalDurationMicroSec.count() << "s";

  // Also output convenient day/hr/min/sec.
  // The following line is a custom duration for a day.
  using days = std::chrono::duration<int, std::ratio<60 * 60 * 24, 1>>;
  days d = std::chrono::duration_cast<days>(totalDuration);
  std::chrono::hours h = std::chrono::duration_cast<std::chrono::hours>(
      totalDuration % days(1));
  std::chrono::minutes m = std::chrono::duration_cast<std::chrono::minutes>(
      totalDuration % std::chrono::hours(1));
  std::chrono::seconds s = std::chrono::duration_cast<std::chrono::seconds>(
      totalDuration % std::chrono::minutes(1));
  // No output if it didn't even take a minute.
  if (!(d.count() == 0 && h.count() == 0 && m.count() == 0))
  {
    bool output = false; // Denotes if we have output anything yet.
    oss << " (";

    // Only output units if they have nonzero values (yes, a bit tedious).
    if (d.count() > 0)
    {
      oss << d.count() << " days";
      output = true;
    }

    if (h.count() > 0)
    {
      if (output)
        oss << ", ";
      oss << h.count() << " hrs";
      output = true;
    }

    if (m.count() > 0)
    {
      if (output)
        oss << ", ";
      oss << m.count() << " mins";
      output = true;
    }

    if (s.count() > 0)
    {
      if (output)
        oss << ", ";
      oss << s.count() << "." << std::setw(1)
          << (totalDurationMicroSec.count() / 100000) << " secs";
    }

    oss << ")";
  }

  oss << std::endl;
  return oss.str();
}

inline void Timers::StopAllTimers()
{
  // Terminate the program timers.  Don't use StopTimer() since that modifies
  // the map and would invalidate our iterators.
#ifndef MLPACK_NO_STD_MUTEX
  std::lock_guard<std::mutex> lock(timersMutex);
#endif

  std::chrono::high_resolution_clock::time_point currTime =
      std::chrono::high_resolution_clock::now();
  for (auto it : timerStartTime)
  {
    for (auto it2 : it.second)
    {
      timers[it2.first] +=
          std::chrono::duration_cast<std::chrono::microseconds>(
          currTime - it2.second);
    }
  }

  // If all timers are stopped, we can clear the maps.
  timerStartTime.clear();
}

inline void Timers::Start(const std::string& timerName,
                          const std::thread::id& threadId)
{
  // Don't do anything if we aren't timing.
  if (!enabled)
    return;

#ifndef MLPACK_NO_STD_MUTEX
  std::lock_guard<std::mutex> lock(timersMutex);
#endif

  if ((timerStartTime.count(threadId) > 0) &&
      (timerStartTime[threadId].count(timerName)))
  {
    std::ostringstream error;
    error << "Timer::Start(): timer '" << timerName
        << "' has already been started";
    throw std::runtime_error(error.str());
  }

  std::chrono::high_resolution_clock::time_point currTime =
      std::chrono::high_resolution_clock::now();

  // If the timer is added for the first time.
  if (timers.count(timerName) == 0)
  {
    timers[timerName] = (std::chrono::microseconds) 0;
  }

  timerStartTime[threadId][timerName] = currTime;
}

inline void Timers::Stop(const std::string& timerName,
                         const std::thread::id& threadId)
{
  // Don't do anything if we aren't timing.
  if (!enabled)
    return;

#ifndef MLPACK_NO_STD_MUTEX
  std::lock_guard<std::mutex> lock(timersMutex);
#endif

  if ((timerStartTime.count(threadId) == 0) ||
      (timerStartTime[threadId].count(timerName) == 0))
  {
    std::ostringstream error;
    error << "Timer::Stop(): no timer with name '" << timerName
        << "' currently running";
    throw std::runtime_error(error.str());
  }

  std::chrono::high_resolution_clock::time_point currTime =
      std::chrono::high_resolution_clock::now();

  // Calculate the delta time.
  timers[timerName] += std::chrono::duration_cast<std::chrono::microseconds>(
      currTime - timerStartTime[threadId][timerName]);

  // Remove the entries.
  timerStartTime[threadId].erase(timerName);
  if (timerStartTime[threadId].empty())
    timerStartTime.erase(threadId);
}

} // namespace util
} // namespace mlpack
