# Mlpack Tests

This directory contains code and data used to test all the algorthims and function implemented in mlpack.

## Test Directories Structure

- *_test.cpp - All the test realted to algorthims
- main_tests/*_test.cpp - All binding realted test
- data - Data needed to run the test

## Add tests 

We have a rich test suite, consisting of almost 2000 test and still counting. So it is suggested to add tests when:

- Adding new functionality.
- Fixing regressions and bugs.

## Building Test

To build test you can simply run `make mlpack_test`.

## To run Test 

We use `Catch2` to write our test and to run the test, you can simply run

` ./bin/mlpack_test`

Also to run all the test in a particular file you can issue a command as

`./bin/mlpack_test "[testname]"`

for example to run all the test which is in `cv_test.cpp` file you can issue a command as

`./bin/mlpack_test "[CVTest]"`

Now similary you can run all the binding related test as 

`./bin/mlpack_test "[BindingTests]"`

To run a single test you can explicitly give the name of the test, for example to run
`BinaryClassificationMetricsTest` present in `cv_test.cpp` file you can run the following
command:

`./bin/mlpack_test BinaryClassificationMetricsTest`

Catch2 also allows very other command which you can view using:

`./bin/mlpack_test -h`

Such as to view all the test and realted tags you can issue the following command:

`./bin/mlpack_test -t`
