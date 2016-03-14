/**
 * @file timers.cpp
 * @author Matthew Amidon
 * @author Marcus Edel
 *
 * Implementation of timers.
 */
#include "timers.hpp"
#include "cli.hpp"
#include "log.hpp"

#include <map>
#include <string>

using namespace mlpack;

inline std::chrono::microseconds getTimeDuration(const std::chrono::high_resolution_clock::time_point start,
                                                  const std::chrono::high_resolution_clock::time_point end)
{
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

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
std::chrono::microseconds Timer::Get(const std::string& name)
{
  return CLI::GetSingleton().timer.GetTimer(name);
}

std::map<std::string, std::chrono::microseconds>& Timers::GetAllTimers()
{
  return timers;
}

std::chrono::microseconds Timers::GetTimer(const std::string& timerName)
{
  return timers[timerName];
}

bool Timers::GetState(std::string timerName)
{
  return timerState[timerName];
}

void Timers::PrintTimer(const std::string& timerName)
{
  std::chrono::microseconds totalDuration = timers[timerName];
  // Converting microseconds to seconds
  std::chrono::seconds totalDurationSec = std::chrono::duration_cast<std::chrono::seconds>(totalDuration);
  std::chrono::microseconds totalDurationMicroSec = std::chrono::duration_cast<std::chrono::microseconds>(totalDuration % std::chrono::seconds(1));
  Log::Info << totalDurationSec.count() << "." << std::setw(6) << std::setfill('0')
      << totalDurationMicroSec.count() << "s";

  // Also output convenient day/hr/min/sec.
  // The following line is a custom duration for a day
  std::chrono::duration<int, std::ratio<60*60*24,1> > days = std::chrono::duration_cast<std::chrono::duration<int, std::ratio<60*60*24,1> > >(totalDuration); // Integer division rounds down.
  std::chrono::hours hours = std::chrono::duration_cast<std::chrono::hours>(totalDuration % std::chrono::duration<int, std::ratio<60*60*24,1> >(1));
  std::chrono::minutes minutes = std::chrono::duration_cast<std::chrono::minutes>(totalDuration % std::chrono::hours(1));
  std::chrono::seconds seconds = std::chrono::duration_cast<std::chrono::seconds>(totalDuration % std::chrono::minutes(1));
  // No output if it didn't even take a minute.
  if (!(days.count() == 0 && hours.count() == 0 && minutes.count() == 0))
  {
    bool output = false; // Denotes if we have output anything yet.
    Log::Info << " (";

    // Only output units if they have nonzero values (yes, a bit tedious).
    if (days.count() > 0)
    {
      Log::Info << days.count() << " days";
      output = true;
    }

    if (hours.count() > 0)
    {
      if (output)
        Log::Info << ", ";
      Log::Info << hours.count() << " hrs";
      output = true;
    }

    if (minutes.count() > 0)
    {
      if (output)
        Log::Info << ", ";
      Log::Info << minutes.count() << " mins";
      output = true;
    }

    if (seconds.count() > 0)
    {
      if (output)
        Log::Info << ", ";
      Log::Info << seconds.count() << "." << std::setw(1) << (totalDurationMicroSec.count() / 100000) <<
          "secs";
      output = true;
    }

    Log::Info << ")";
  }

  Log::Info << std::endl;
}

std::chrono::high_resolution_clock::time_point Timers::GetTime()
{
  return std::chrono::high_resolution_clock::now();
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

  std::chrono::high_resolution_clock::time_point currTime = GetTime();

  // If the timer is added first time
  if (timers.count(timerName) == 0)
  {
    timers[timerName] = (std::chrono::microseconds)0;
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

  std::chrono::high_resolution_clock::time_point currTime = GetTime();

  // Calculate the delta time.
  timers[timerName] += getTimeDuration(timerStartTime[timerName], currTime);
}
