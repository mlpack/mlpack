# mlpack Timers

mlpack provides a simple timer interface for the timing of machine learning
methods.  The results of any timers used during the program are displayed at
output by any command-line binding, when `--verbose` is given:

```sh
$ mlpack_knn -r dataset.csv -n neighbors_out.csv -d distances_out.csv -k 5 -v
<...>
[INFO ] Program timers:
[INFO ]   computing_neighbors: 0.010650s
[INFO ]   loading_data: 0.002567s
[INFO ]   saving_data: 0.001115s
[INFO ]   total_time: 0.149816s
[INFO ]   tree_building: 0.000534s
```

## Timer API

In C++, the `mlpack::Timers` class can be used to add timers to a program.  The
`mlpack::Timers` class provides three simple methods:

```c++
void Timer::Start(const char* name);
void Timer::Stop(const char* name);
timeval Timer::Get(const char* name);
```

Every binding is called with an `mlpack::Timers&`, which can be used in the body
of that binding.  For the sake of this discussion, let us call that object
`timers`.

Each timer is given a name, and is referenced by that name.  You can call
`timers.Start()` and `timers.Stop()` multiple times for a particular timer name,
and the result will be the sum of the runs of the timer.  Note that
`timers.Stop()` must be called before `timers.Start()` is called again,
otherwise a `std::runtime_error` exception will be thrown.

A `"total_time"` timer is run automatically for each mlpack binding.

## Timer Example

Below is a very simple example of timer usage in code.

```c++
#include <mlpack/core.hpp>
#include <mlpack/core/util/io.hpp>
#define BINDING_TYPE BINDING_TYPE_CLI
#include <mlpack/core/util/mlpack_main.hpp>

using namespace mlpack;

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Start a timer.
  timers.Start("some_timer");

  // Do some things.
  DoSomeStuff();

  // Stop the timer.
  timers.Stop("some_timer");
}
@endcode

If the `verbose` flag was given to this binding, then a command-line binding
would print the time that `"some_timer"` ran for at the end of the program's
output.
