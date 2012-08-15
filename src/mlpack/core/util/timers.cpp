/**
 * @file timers.cpp
 * @author Matthew Amidon
 *
 * Implementation of timers.
 *
 * This file is part of MLPACK 1.0.2.
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
#include "timers.hpp"
#include "cli.hpp"
#include "log.hpp"

#include <map>
#include <string>

using namespace mlpack;

// On Windows machines, we need to define timersub.
#ifdef _WIN32
inline void timersub(const timeval* tvp, const timeval* uvp, timeval* vvp)
{
  vvp->tv_sec = tvp->tv_sec - uvp->tv_sec;
  vvp->tv_usec = tvp->tv_usec - uvp->tv_usec;
  if (vvp->tv_usec < 0)
  {
     --vvp->tv_sec;
     vvp->tv_usec += 1000000;
  }
}
#endif

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
timeval Timer::Get(const std::string& name)
{
  return CLI::GetSingleton().timer.GetTimer(name);
}

std::map<std::string, timeval>& Timers::GetAllTimers()
{
  return timers;
}

timeval Timers::GetTimer(const std::string& timerName)
{
  std::string name(timerName);
  return timers[name];
}

void Timers::PrintTimer(const std::string& timerName)
{
  timeval& t = timers[timerName];
  Log::Info << t.tv_sec << "." << std::setw(6) << std::setfill('0')
      << t.tv_usec << "s";

  // Also output convenient day/hr/min/sec.
  int days = t.tv_sec / 86400; // Integer division rounds down.
  int hours = (t.tv_sec % 86400) / 3600;
  int minutes = (t.tv_sec % 3600) / 60;
  int seconds = (t.tv_sec % 60);
  // No output if it didn't even take a minute.
  if (!(days == 0 && hours == 0 && minutes == 0))
  {
    bool output = false; // Denotes if we have output anything yet.
    Log::Info << " (";

    // Only output units if they have nonzero values (yes, a bit tedious).
    if (days > 0)
    {
      Log::Info << days << " days";
      output = true;
    }

    if (hours > 0)
    {
      if (output)
        Log::Info << ", ";
      Log::Info << hours << " hrs";
      output = true;
    }

    if (minutes > 0)
    {
      if (output)
        Log::Info << ", ";
      Log::Info << minutes << " mins";
      output = true;
    }

    if (seconds > 0)
    {
      if (output)
        Log::Info << ", ";
      Log::Info << seconds << "." << std::setw(1) << (t.tv_usec / 100000) <<
          "secs";
      output = true;
    }

    Log::Info << ")";
  }

  Log::Info << std::endl;
}

void Timers::StartTimer(const std::string& timerName)
{
  timeval tmp;

  tmp.tv_sec = 0;
  tmp.tv_usec = 0;

#ifndef _WIN32
  gettimeofday(&tmp, NULL);
#else
  FileTimeToTimeVal(&tmp);
#endif

  // Check to see if the timer already exists.  If it does, we'll subtract the
  // old value.
  if (timers.count(timerName) == 1)
  {
    timeval tmpDelta;

    timersub(&tmp, &timers[timerName], &tmpDelta);

    tmp = tmpDelta;
  }

  timers[timerName] = tmp;
}

#ifdef _WIN32
void Timers::FileTimeToTimeVal(timeval* tv)
{
  FILETIME ftime;
  uint64_t ptime = 0;
  // Acquire the file time.
  GetSystemTimeAsFileTime(&ftime);
  // Now convert FILETIME to timeval.
  ptime |= ftime.dwHighDateTime;
  ptime = ptime << 32;
  ptime |= ftime.dwLowDateTime;
  ptime /= 10;
  ptime -= DELTA_EPOCH_IN_MICROSECS;

  tv->tv_sec = (long) (ptime / 1000000UL);
  tv->tv_usec = (long) (ptime % 1000000UL);
}
#endif // _WIN32

void Timers::StopTimer(const std::string& timerName)
{
  timeval delta, b, a = timers[timerName];

#ifndef _WIN32
  gettimeofday(&b, NULL);
#else
  FileTimeToTimeVal(&b);
#endif
  // Calculate the delta time.
  timersub(&b, &a, &delta);
  timers[timerName] = delta;
}
