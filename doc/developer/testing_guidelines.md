# Introduction

Software testing is an integral part of development workflow of every major piece of technology that you see around. Naturally, it is one of the core guidelines to write unit tests and `mlpack` follows the same ideology in our `tests` submodule. We all make mistakes, and introducing tests not only helps to realize those mistakes, it also makes the review process much faster. A proper test framework also ensures that one does not simply disturb the higher-level API correctness, while meddling with the lower level stuff, saving a huge amount of time.

It is also worth noting, that while it is great to follow all of these mentioned guidelines, a number of them are open to interpretation depending upon unique situations. So we highly encourage people to start a conversation whenever in doubt. Shall we begin?

## What to Test?

While we appreciate your dedication to test each individual line of your code separately in a unit test, it is simply too redundant. However, this doesn't mean that you should leave areas of your code unchecked at all. Ultimately, we wish to get a working and correct implementation of a functionality out of your code. Ideally we strive to test out individual functions and the use of those functions in conjunction with previously available functionality.

## Objective and Limits

The end goal of software testing is to provide as much code coverage as possible for every line of code written. `mlpack` wouldn't be useful at all if simply the code you wrote is wrong; wrong code is subjectively much worse than no code at all. A good test suite would aim to test the logical (the implementation arrives at the correct solution) and semantic (the implementation arrives at the solution using the correct methodology) aspects of the code. Syntactical checking is provided by the compiler, so we do not worry about that. However, there are still hurdles to realizing this objective. With `mlpack`, we expect contributors to provide valid test-cases, which itself is an error prone task, especially in situations where manual calculation is hard. Furthermore, valid test-cases only test the logical correctness of the code, semantical issues can arise as well. Take the following piece of code from a previous case,

### Logically Correct but Semantically Incorrect
```
Incorrect
g.each_col() += arma::sum(norm.each_col() % -stdInv, 1)
                + var % arma::mean(-2 * inputMean, 1) / input.n_cols;

Correct
g.each_col() += (arma::sum(norm.each_col() % -stdInv, 1)
                 + (var % arma::mean(-2 * inputMean, 1))) / input.n_cols;
```

The above piece of code is used while computing the backward function of `Batch Normalization` layer. On a quick look, it doesn't seem to be radically different in the two cases, however, there is a subtle difference,

```
Assuming,
a = arma::sum(norm.each_col() % -stdInv, 1); 
b = var % arma::mean(-2 * inputMean, 1);
c = input.n_cols;

Incorrect
g.each_col() += a + b / c; or more simply, g.each_col() += a + (b / c);

Correct
g.each_col() += (a + b) / c;
```
What's even scarier is that for the existing test case, the above two virtually provided the same output (logical correctness). Such implementations are quite hard to analyze and debug, and often go unnoticed by the developers who implement them. Hence, we also request our contributors to be patient with the review process, we just don't want to rush and end up shipping faulty code.

## Good Test Sources

Ideally, the best source of logical tests are values computed manually by defining the methodology implemented in code, and plugging in some random cases. However, these are error prone too, since the number of test cases one may implement are limited and might not cover all the corner cases. We may suggest you to implement some of the original test routines and architectures used by the authors of the said technique, since these tend to be the most scrutinized aspects of their work. Finally, in cases where none of the above are viable options, one can look up for the test cases implemented in other major libraries (`PyTorch`, `Tensorflow` and the rest), and replicate their results. This is the least preferred approach to writing test cases, but comes in handy in certain scenarios. For example, take the following cases:

### Manual Computation
```
You implemented a recently published activation function module, however the
publication doesn't mention any practical value based results, except for
the formula. Here, manual computation is the most plausible approach for
creating tests for your code.
```

### Authored Test Cases
```
You implemented the module for a new type of Generative Adversarial Network (GAN)
from a peer reviewed publication and wish to test the same. However, testing is an
abstract concept here, since GANs are used to primarily generate data. In such a
scenario, it is advised to replicate the test models depicted in the paper, and
comparing the output data (or imagery) directly with the standard results
mentioned in the paper.
```

### External Libraries
```
You implemented the module for computing 2D convolutions on the given matrix.
However, it is cumbersome to manually compute the different test cases with
different parameters (stride, filter size, padding and so on). Also, since it
is a general functionality, you couldn't find any authored publications with
test cases for the same. As a last resort, you can head to
tensorflow.nn.conv_2d() (https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)
and pass the input matrix and other parameters and record the result, later
using it as a test case for the results generated from your implementation.
```

## Creating Tests

We use `Catch2` as the unit test framework. For a detailed explanation of the usage, 
please refer to the [documentation](https://github.com/catchorg/Catch2/blob/devel/docs/Readme.md).
However, for a quick view, you can refer to the previously implemented test cases in 
`src/mlpack/tests/` under the appropriate sub-directory as well as [README.md](https://github.com/mlpack/mlpack/blob/master/src/mlpack/tests/README.md) in `src/mlpack/tests/`.

### Tests With Random Input
For time and resource-based constraints, `Travis` only runs the test cases for a single time. This is however not the correct way to test out the code implementations which require a random input. It is possible that the code may pass on a single run, but while on multiple runs, it might lead to a failure (like the way it is done in `Jenkins`). For such a case, the right way to test the random input based tests is to set the random seed and run many times locally on your machine to be sure. Add `mlpack::math::RandomSeed(std::time(NULL))` to the top of your test case, compile and then run the following script in the terminal:
```
$ i=0; while(true); do echo $i; bin/mlpack_test --rng-seed $i "[testname]" 2>&1 | grep 'fatal error'; i=$(($i + 1)); sleep 1; done
```

It's useful to run that script for ~1000 iterations and make sure the test does not fail in that period.  If you don't do this, then you should not be surprised if a bug gets opened and you get tagged in it because the test you wrote that got merged was not stable. :)

Note that if you are writing a test for a binding (i.e. something in `src/mlpack/tests/main_tests/`), then `BINDING_TYPE_TEST` is defined such that `math::RandomSeed()` doesn't do anything.  Therefore, in place of a call to `RandomSeed()`, use this code (which is just the implementation of `RandomSeed()`):

```
const size_t seed = std::time(NULL);
mlpack::math::randGen.seed((uint32_t) seed);
srand((unsigned int) seed);
arma::arma_rng::set_seed(seed);
```

Then you can build and use the same loop as above.

## Reporting and Analyzing Failures

Okay, so we're through with the conception and the implementation of the test cases for your code. However, it may still happen that your code may fail to compile, raise a test-case exception, or a test failure. Here's what you can do to correct the implementation:

### Code Compilation Failure
Nothing much to say here, compilation failure means that your usage of language syntax is simply wrong. Fortunately, C++ enjoys the support of powerful compilers, which would point out the line of error, as well as the candidate fixtures, if applicable. Being able to make sense out of compilation errors is an art in itself, one that comes with practice and patience. So don't rush yourself through your errors.

### Test Case Exception
If for some reason your code compiles well, but raises an exception in one of the test cases whenever you run the test suite, it can broadly point to two issues. Either your implementation of the test case is semantically incorrect, or your implementation of your code is leading to a runtime error. Finding out the case out of these two should help you in clearing out your problem.

### Test Failure
Much like the previous case, test failures can emerge from two cases as well. Either your code is logically incorrect and leads to the wrong answer, or simply the test value your are trying to compare your output against is plain wrong. Re-checking your code and test value is the only way to go here. If that doesn't help, we're there to alleviate the pain.