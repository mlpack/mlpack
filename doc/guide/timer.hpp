/*! @page timer MLPACK Timers

@section timerintro Introduction

MLPACK provides a simple timer interface for the timing of machine learning
methods.  The results of any timers used during the program are displayed at
output by the mlpack::CLI object, when --verbose is given:

@code
$ allknn -i=data.csv -k 5
<...>
[INFO ] Program timers:
[INFO ]   computing_neighbors: 0.044764s
[INFO ]   total_time: 0.061249s
[INFO ]   tree_building: 0.003075s
@endcode

@section usingtimer Timer API

The mlpack::Timer class provides three simple methods:

@code
void Timer::Start(const char* name);
void Timer::Stop(const char* name);
timeval Timer::Get(const char* name);
@endcode

Each timer is given a name, and is referenced by that name.

A "total_time" timer is run by default for each MLPACK program.

@section example Timer Example

Below is a very simple example of timer usage in code.

@code
#include <mlpack/core.hpp>

using namespace mlpack;

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  // Start a timer.
  Timer::Start("some_timer");

  // Do some things.
  DoSomeStuff();

  // Stop the timer.
  Timer::Stop("some_timer");
}
@endcode

If the --verbose flag was given to this executable, the resultant time that
"some_timer" ran for would be shown.

*/
