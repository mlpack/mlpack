/**
 * @file timers.cpp
 * @author Matthew Amidon
 *
 * Implementation of timers.
 */
#include "timers.hpp"
#include "cli.hpp"
#include "log.hpp"

#include <map>
#include <string>

using namespace mlpack;

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
        Log::Info << ",";
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
  timers[timerName] = tmp;
}

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
  ptime -= DELTA_EPOC_IN_MICROSECONDS;

  tv.tv_sec = (long) (ptime / 1000000UL);
  tv.tv_usec = (long) (ptime % 1000000UL);
}

#endif // _WIN32
