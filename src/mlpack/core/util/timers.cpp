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
using namespace std::chrono;

/**
 * Start the given timer.
 */
void Timer::Start(const std::string& name)
{
  CLI::GetSingleton().timer.StartTimer(name);
}

/**
 * Stop the given timer.
 */
void Timer::Stop(const std::string& name)
{
  CLI::GetSingleton().timer.StopTimer(name);
}

/**
 * Get the given timer.
 */
microseconds Timer::Get(const std::string& name)
{
  return CLI::GetSingleton().timer.GetTimer(name);
}

std::map<std::string, microseconds>& Timers::GetAllTimers()
{
  return timers;
}

microseconds Timers::GetTimer(const std::string& timerName)
{
  return timers[timerName];
}

bool Timers::GetState(std::string timerName)
{
  return timerState[timerName];
}

void Timers::PrintTimer(const std::string& timerName)
{
  microseconds totalDuration = timers[timerName];
  // Convert microseconds to seconds.
  seconds totalDurationSec = duration_cast<seconds>(totalDuration);
  microseconds totalDurationMicroSec =
      duration_cast<microseconds>(totalDuration % seconds(1));
  Log::Info << totalDurationSec.count() << "." << std::setw(6)
      << std::setfill('0') << totalDurationMicroSec.count() << "s";

  // Also output convenient day/hr/min/sec.
  // The following line is a custom duration for a day.
  typedef duration<int, std::ratio<60 * 60 * 24, 1>> days;
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
      Log::Info << s.count() << "." << std::setw(1)
          << (totalDurationMicroSec.count() / 100000) << " secs";
      output = true;
    }

    Log::Info << ")";
  }

  Log::Info << std::endl;
}

high_resolution_clock::time_point Timers::GetTime()
{
  return high_resolution_clock::now();
}

void Timers::StartTimer(const std::string& timerName)
{
  if ((timerState[timerName] == 1) && (timerName != "total_time"))
  {
    std::ostringstream error;
    error << "Timer::Start(): timer '" << timerName
        << "' has already been started";
    throw std::runtime_error(error.str());
  }

  timerState[timerName] = true;

  high_resolution_clock::time_point currTime = GetTime();

  // If the timer is added first time
  if (timers.count(timerName) == 0)
  {
    timers[timerName] = (microseconds) 0;
  }

  timerStartTime[timerName] = currTime;
}

void Timers::StopTimer(const std::string& timerName)
{
  if ((timerState[timerName] == 0) && (timerName != "total_time"))
  {
    std::ostringstream error;
    error << "Timer::Stop(): timer '" << timerName
        << "' has already been stopped";
    throw std::runtime_error(error.str());
  }

  timerState[timerName] = false;

  high_resolution_clock::time_point currTime = GetTime();

  // Calculate the delta time.
  timers[timerName] += duration_cast<microseconds>(currTime -
      timerStartTime[timerName]);
}
