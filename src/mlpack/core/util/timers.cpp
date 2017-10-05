/**
 * @file timers.cpp
 * @author Matthew Amidon
 * @author Marcus Edel
 *
 * Implementation of timers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "timers.hpp"
#include "cli.hpp"
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
  CLI::GetSingleton().timer.StartTimer(name, this_thread::get_id());
}

/**
 * Stop the given timer.
 */
void Timer::Stop(const string& name)
{
  CLI::GetSingleton().timer.StopTimer(name, this_thread::get_id());
}

/**
 * Get the given timer, summing over all threads.
 */
microseconds Timer::Get(const string& name)
{
  microseconds result(0);
  for (auto it : CLI::GetSingleton().timer.GetAllTimers())
    if (it.second.count(name) > 0)
      result += it.second[name];

  return result;
}

map<thread::id, map<string, microseconds>>&
Timers::GetAllTimers()
{
  return timers;
}

list<string> Timers::GetAllTimerNames()
{
  list<string> l;
  for (auto it : CLI::GetSingleton().timer.GetAllTimers())
    for (auto it2 : it.second)
      l.push_back(it2.first);

  // Filter duplicates.
  l.unique();

  return l;
}

microseconds Timers::GetTimer(const string& timerName,
                              const thread::id& threadId)
{
  return timers[threadId][timerName];
}

bool Timers::GetState(const string& timerName,
                      const thread::id& threadId)
{
  return timerState[threadId][timerName];
}

void Timers::PrintTimer(const string& timerName)
{
  microseconds totalDuration(0);
  for (auto it : timers)
    if (it.second.count(timerName) > 0)
      totalDuration += it.second[timerName];

  // Convert microseconds to seconds.
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
      output = true;
    }

    Log::Info << ")";
  }

  Log::Info << endl;
}

high_resolution_clock::time_point Timers::GetTime()
{
  return high_resolution_clock::now();
}

void Timers::StartTimer(const string& timerName,
                        const thread::id& threadId)
{
  if ((timerState[threadId][timerName] == 1) && (timerName != "total_time"))
  {
    ostringstream error;
    error << "Timer::Start(): timer '" << timerName
        << "' has already been started";
    throw runtime_error(error.str());
  }

  timerState[threadId][timerName] = true;

  high_resolution_clock::time_point currTime = GetTime();

  // If the timer is added first time.
  if (timers[threadId].count(timerName) == 0)
  {
    timers[threadId][timerName] = (microseconds) 0;
  }

  timerStartTime[threadId][timerName] = currTime;
}

void Timers::StopTimer(const string& timerName,
                       const thread::id& threadId)
{
  if ((timerState[threadId][timerName] == 0) && (timerName != "total_time"))
  {
    ostringstream error;
    error << "Timer::Stop(): timer '" << timerName
        << "' has already been stopped";
    throw runtime_error(error.str());
  }

  timerState[threadId][timerName] = false;

  high_resolution_clock::time_point currTime = GetTime();

  // Calculate the delta time.
  timers[threadId][timerName] += duration_cast<microseconds>(currTime -
      timerStartTime[threadId][timerName]);
}
