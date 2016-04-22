/*! @page timer mlpack Timers

@section timerintro Introduction

mlpack provides a simple timer interface for the timing of machine learning
methods.  The results of any timers used during the program are displayed at
output by the mlpack::CLI object, when --verbose is given:

@code
$ mlpack_knn -r dataset.csv -n neighbors_out.csv -d distances_out.csv -k 5 -v
<...>
[INFO ] Program timers:
[INFO ]   computing_neighbors: 0.010650s
[INFO ]   loading_data: 0.002567s
[INFO ]   saving_data: 0.001115s
[INFO ]   total_time: 0.149816s
[INFO ]   tree_building: 0.000534s
@endcode

@section usingtimer Timer API

The mlpack::Timer class provides three simple methods:

@code
void Timer::Start(const char* name);
void Timer::Stop(const char* name);
timeval Timer::Get(const char* name);
@endcode

Each timer is given a name, and is referenced by that name.  You can call \c
Timer::Start() and \c Timer::Stop() multiple times for a particular timer name,
and the result will be the sum of the runs of the timer.  Note that \c
Timer::Stop() must be called before \c Timer::Start() is called again,
otherwise a std::runtime_error exception will be thrown.

A "total_time" timer is run by default for each mlpack program.

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
