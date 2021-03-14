/**
 * @file core/util/timers.cpp
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
#include "io.hpp"
#include "log.hpp"

#include <map>
#include <string>

using namespace mlpack;
using namespace std;
using namespace chrono;

/**
 * Start the given timer.
 */
void Timer::Start(const string& name)
{
  IO::GetSingleton().timer.StartTimer(name, this_thread::get_id());
}

/**
 * Stop the given timer.
 */
void Timer::Stop(const string& name)
{
  IO::GetSingleton().timer.StopTimer(name, this_thread::get_id());
}

/**
 * Get the given timer, summing over all threads.
 */
microseconds Timer::Get(const string& name)
{
  return IO::GetSingleton().timer.GetTimer(name);
}

// Enable timing.
void Timer::EnableTiming()
{
  IO::GetSingleton().timer.Enabled() = true;
}

// Disable timing.
void Timer::DisableTiming()
{
  IO::GetSingleton().timer.Enabled() = false;
}

// Reset all timers.  Save state of enabled.
void Timer::ResetAll()
{
  IO::GetSingleton().timer.Reset();
}

// Reset a Timers object.
void Timers::Reset()
{
  lock_guard<mutex> lock(timersMutex);
  timers.clear();
  timerStartTime.clear();
}

map<string, microseconds> Timers::GetAllTimers()
{
  // Make a copy of the timer.
  lock_guard<mutex> lock(timersMutex);
  return timers;
}

microseconds Timers::GetTimer(const string& timerName)
{
  if (!enabled)
    return microseconds(0);

  lock_guard<mutex> lock(timersMutex);
  return timers[timerName];
}

bool Timers::GetState(const string& timerName,
                      const thread::id& threadId)
{
  lock_guard<mutex> lock(timersMutex);
  if (timerStartTime.count(threadId) == 0)
    return 0;
  return (timerStartTime[threadId].count(timerName) > 0);
}

void Timers::PrintTimer(const string& timerName)
{
  // Convert microseconds to seconds.
  microseconds totalDuration = GetTimer(timerName);
  seconds totalDurationSec = duration_cast<seconds>(totalDuration);
  microseconds totalDurationMicroSec =
      duration_cast<microseconds>(totalDuration % seconds(1));
  Log::Info << totalDurationSec.count() << "." << setw(6)
      << setfill('0') << totalDurationMicroSec.count() << "s";

  // Also output convenient day/hr/min/sec.
  // The following line is a custom duration for a day.
  typedef duration<int, ratio<60 * 60 * 24, 1>> days;
  days d = duration_cast<days>(totalDuration);
  hours h = duration_cast<hours>(totalDuration % days(1));
  minutes m = duration_cast<minutes>(totalDuration % hours(1));
  seconds s = duration_cast<seconds>(totalDuration % minutes(1));
  // No output if it didn't even take a minute.
  if (!(d.count() == 0 && h.count() == 0 && m.count() == 0))
  {
    bool output = false; // Denotes if we have output anything yet.
    Log::Info << " (";

    // Only output units if they have nonzero values (yes, a bit tedious).
    if (d.count() > 0)
    {
      Log::Info << d.count() << " days";
      output = true;
    }

    if (h.count() > 0)
    {
      if (output)
        Log::Info << ", ";
      Log::Info << h.count() << " hrs";
      output = true;
    }

    if (m.count() > 0)
    {
      if (output)
        Log::Info << ", ";
      Log::Info << m.count() << " mins";
      output = true;
    }

    if (s.count() > 0)
    {
      if (output)
        Log::Info << ", ";
      Log::Info << s.count() << "." << setw(1)
          << (totalDurationMicroSec.count() / 100000) << " secs";
    }

    Log::Info << ")";
  }

  Log::Info << endl;
}

void Timers::StopAllTimers()
{
  // Terminate the program timers.  Don't use StopTimer() since that modifies
  // the map and would invalidate our iterators.
  lock_guard<mutex> lock(timersMutex);

  high_resolution_clock::time_point currTime = high_resolution_clock::now();
  for (auto it : timerStartTime)
    for (auto it2 : it.second)
      timers[it2.first] += duration_cast<microseconds>(currTime - it2.second);

  // If all timers are stopped, we can clear the maps.
  timerStartTime.clear();
}

void Timers::StartTimer(const string& timerName,
                        const thread::id& threadId)
{
  // Don't do anything if we aren't timing.
  if (!enabled)
    return;

  lock_guard<mutex> lock(timersMutex);

  if ((timerStartTime.count(threadId) > 0) &&
      (timerStartTime[threadId].count(timerName)))
  {
    ostringstream error;
    error << "Timer::Start(): timer '" << timerName
        << "' has already been started";
    throw runtime_error(error.str());
  }

  high_resolution_clock::time_point currTime = high_resolution_clock::now();

  // If the timer is added for the first time.
  if (timers.count(timerName) == 0)
  {
    timers[timerName] = (microseconds) 0;
  }

  timerStartTime[threadId][timerName] = currTime;
}

void Timers::StopTimer(const string& timerName,
                       const thread::id& threadId)
{
  // Don't do anything if we aren't timing.
  if (!enabled)
    return;

  lock_guard<mutex> lock(timersMutex);

  if ((timerStartTime.count(threadId) == 0) ||
      (timerStartTime[threadId].count(timerName) == 0))
  {
    ostringstream error;
    error << "Timer::Stop(): no timer with name '" << timerName
        << "' currently running";
    throw runtime_error(error.str());
  }

  high_resolution_clock::time_point currTime = high_resolution_clock::now();

  // Calculate the delta time.
  timers[timerName] += duration_cast<microseconds>(currTime -
      timerStartTime[threadId][timerName]);

  // Remove the entries.
  timerStartTime[threadId].erase(timerName);
  if (timerStartTime[threadId].empty())
    timerStartTime.erase(threadId);
}
