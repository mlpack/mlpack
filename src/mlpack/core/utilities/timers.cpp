/**
 * @file timers.cpp
 * @author Matthew Amidon
 *
 * Implementation of timers.
 */
#include "timers.hpp"
#include "../io/cli.hpp"
#include "../io/log.hpp"

#include <map>
#include <string>

using namespace mlpack;

std::map<std::string, timeval> Timers::timers;

std::map<std::string, timeval> Timers::GetAllTimers()
{
  return timers;
}

timeval Timers::GetTimer(const char* timerName)
{
  std::string name(timerName);
  return timers[name];
}

void Timers::PrintTimer(const char* timerName)
{
  std::string name=timerName;
  timeval& t = timers[name];
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

void Timers::StartTimer(const char* timerName)
{
  // Convert the name to std::string.
  std::string name(timerName);
  timeval tmp;

  tmp.tv_sec = 0;
  tmp.tv_usec = 0;

#ifndef _WIN32
  gettimeofday(&tmp, NULL);
#else
  FileTimeToTimeVal(&tmp);
#endif
  timers[name] = tmp;
}

void Timers::StopTimer(const char* timerName)
{
  std::string name(timerName);
  timeval delta, b, a = timers[name];

#ifndef _WIN32
  gettimeofday(&b, NULL);
#else
  FileTimeToTimeVal(&b);
#endif
  // Calculate the delta time.
  timersub(&b, &a, &delta);
  timers[name] = delta;
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
