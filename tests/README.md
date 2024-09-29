# mlpack Tests

This directory contains code and data used to test all the algorithms and functions implemented in mlpack.

## Test Directories Structure

- *_test.cpp - tests for non-neural-network methods
- ann/*_test.cpp - tests for code relating to neural networks (including
  reinforcement learning)
- main_tests/*_test.cpp - binding tests
- data - data needed to run the tests

## Add tests

We have a rich test suite, consisting of almost 2000 tests (and still counting). It is suggested to add tests when:

- Adding new functionality.
- Fixing regressions and bugs.

## Building Tests

To build the test suite you can simply run `make mlpack_test` from a build
directory that has been properly configured with CMake..

## To run Tests

We use `Catch2` to write our tests. To run all tests, you can simply run:

`./bin/mlpack_test`

To run all tests in a particular file you can run:

`./bin/mlpack_test "[testname]"`

where `testname` is the name of the test suite. For example to run all
collaborative filtering tests implemented in `cv_test.cpp` you can run:

`./bin/mlpack_test "[CVTest]"`

Now similarly you can run all the binding related tests using:

`./bin/mlpack_test "[BindingTests]"`

To run a single test, you can explicitly provide the name of the test, for example, to run
`BinaryClassificationMetricsTest` implemented in `cv_test.cpp` you can run the following:

`./bin/mlpack_test BinaryClassificationMetricsTest`

Catch2 provides many other features like filtering; check out the
[Catch2 reference section](https://github.com/catchorg/Catch2/blob/devel/docs/Readme.md#top)
for more details.
