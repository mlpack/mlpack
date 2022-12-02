# mlpack in C++ quickstart

This page describes how you can quickly get started using mlpack in C++ and
gives a few examples of usage, and pointers to deeper documentation.

Keep in mind that mlpack also has interfaces to other languages, and quickstart
guides for those other languages are available too.  If that is what you are
looking for, see the quickstarts for [Python](python.md),
[the command line](cli.md), [Julia](julia.md), [R](R.md), or [Go](go.md).

## Installing mlpack

To use mlpack in C++, you only need the header files associated with the
libraries, and the dependencies Armadillo and ensmallen (detailed in the
[main README](../../README.md)).  The headers may already be pre-packaged for
your distribution; for instance, for Ubuntu and Debian you can simply run the
command

```sh
sudo apt-get install mlpack-dev
```

and on Fedora or Red Hat:

```sh
sudo dnf install mlpack
```

If you run a different distribution, mlpack may be packaged under a different
name.  And if it is not packaged, you can use a Docker image from Dockerhub:

```sh
docker run -it mlpack/mlpack /bin/bash
```

This Docker image has mlpack headers already installed.

If you prefer to build mlpack from scratch, see the
[main README](../../README.md).

## Installing mlpack from vcpkg

The mlpack port in vcpkg is kept up to date by Microsoft team members and community contributors. The url of vcpkg is: https://github.com/Microsoft/vcpkg . You can download and install mlpack using the vcpkg dependency manager:

```shell
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh  # ./bootstrap-vcpkg.bat for Windows
./vcpkg integrate install
./vcpkg install mlpack
```

If the version is out of date, please [create an issue or pull request](https://github.com/Microsoft/vcpkg) on the vcpkg repository.

## Simple quickstart example

As a really simple example of how to use mlpack in C++, let's do some simple
classification on a subset of the standard machine learning `covertype` dataset.
We'll first split the dataset into a training set and a test set, then we'll
train an mlpack random forest on the training data, and finally we'll print the
accuracy of the random forest on the test dataset.

The first step is to download the covertype dataset onto your system so that it
is available for the program.  A shell command below is given to do this:

```sh
# Get the dataset and unpack it.
wget https://www.mlpack.org/datasets/covertype-small.data.csv.gz
wget https://www.mlpack.org/datasets/covertype-small.labels.csv.gz
gunzip covertype-small.data.csv.gz covertype-small.labels.csv.gz
```

With that in place, let's write a C++ program to split the data and perform the
classification:

```c++
// Define these to print extra informational output and warnings.
#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN
#include <mlpack.hpp>

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace std;

int main()
{
  // Load the datasets.
  mat dataset;
  Row<size_t> labels;
  if (!data::Load("covertype-small.data.csv", dataset))
    throw std::runtime_error("Could not read covertype-small.data.csv!");
  if (!data::Load("covertype-small.labels.csv", labels))
    throw std::runtime_error("Could not read covertype-small.labels.csv!");

  // Labels are 1-7, but we want 0-6 (we are 0-indexed in C++).
  labels -= 1;

  // Now split the dataset into a training set and test set, using 30% of the
  // dataset for the test set.
  mat trainDataset, testDataset;
  Row<size_t> trainLabels, testLabels;
  data::Split(dataset, labels, trainDataset, testDataset, trainLabels,
      testLabels, 0.3);

  // Create the RandomForest object and train it on the training data.
  RandomForest r(trainDataset,
                 trainLabels,
                 7 /* number of classes */,
                 10 /* number of trees */,
                 3 /* minimum leaf size */);

  // Compute and print the training error.
  Row<size_t> trainPredictions;
  r.Classify(trainDataset, trainPredictions);
  const double trainError =
      arma::accu(trainPredictions != trainLabels) * 100.0 / trainLabels.n_elem;
  cout << "Training error: " << trainError << "%." << endl;

  // Now compute predictions on the test points.
  Row<size_t> testPredictions;
  r.Classify(testDataset, testPredictions);
  const double testError =
      arma::accu(testPredictions != testLabels) * 100.0 / testLabels.n_elem;
  cout << "Test error: " << testError << "%." << endl;
}
```

Now, you can compile the program with your favorite C++ compiler; here's an
example command that uses `g++`, and assumes the file above is saved as
`cpp_quickstart_1.cpp`.

```sh
g++ -O3 -std=c++14 -o cpp_quickstart_1 cpp_quickstart_1.cpp -larmadillo -fopenmp
```

Then, you can run the program easily:

```sh
./cpp_quickstart_1
```

We can see by looking at the output that we achieve reasonably good accuracy on
the test dataset (80%+):

```
Training error: 19.4329%.
Test error: 24.17%.
```

It's easy to modify the code above to do more complex things, or to use
different mlpack learners, or to interface with other machine learning toolkits.

## Using mlpack for movie recommendations

In this example, we'll train a collaborative filtering model using mlpack's `CF`
class.  We'll train this on this
[MovieLens dataset](https://grouplens.org/datasets/movielens/), and then we'll
use the model that we train to give recommendations.

First, download the MovieLens dataset:

```sh
wget https://www.mlpack.org/datasets/ml-20m/ratings-only.csv.gz
wget https://www.mlpack.org/datasets/ml-20m/movies.csv.gz
gunzip ratings-only.csv.gz movies.csv.gz
```

Next, we can use the following C++ code:

```cpp
// Define these to print extra informational output and warnings.
#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN
#include <mlpack.hpp>

using namespace arma;
using namespace mlpack;
using namespace mlpack::cf;
using namespace std;

int main()
{
  // Load the ratings.
  mat ratings;
  if (!data::Load("ratings-only.csv", ratings))
    throw std::runtime_error("Could not load ratings-only.csv!");
  // Now, load the names of the movies as a single-feature categorical dataset.
  // We can use `moviesInfo.UnmapString(i, 0)` to get the i'th string.
  data::DatasetInfo moviesInfo;
  mat movies; // This will be unneeded.
  if (!data::Load("movies.csv", movies, moviesInfo))
    throw std::runtime_error("Could not load movies.csv!");

  // Split the ratings into a training set and a test set, using 10% of the
  // dataset for the test set.
  mat trainRatings, testRatings;
  data::Split(ratings, trainRatings, testRatings, 0.1);

  // Train the CF model using RegularizedSVD as the decomposition algorithm.
  // Here we use a rank of 10 for the decomposition.
  CFType<RegSVDPolicy> cf(
      trainRatings,
      RegSVDPolicy(),
      5, /* number of users to use for similarity computations */
      10 /* rank of decomposition */);

  // Now compute the RMSE for the test set user and item combinations.  To do
  // this we must assemble the list of users and items.
  Mat<size_t> combinations(2, testRatings.n_cols);
  for (size_t i = 0; i < testRatings.n_cols; ++i)
  {
    combinations(0, i) = size_t(testRatings(0, i)); // (user)
    combinations(1, i) = size_t(testRatings(1, i)); // (item)
  }
  vec predictions;
  cf.Predict(combinations, predictions);
  const double rmse = norm(predictions - testRatings.row(2).t(), 2) /
      sqrt((double) testRatings.n_cols);
  std::cout << "RMSE of trained model is " << rmse << "." << endl;

  // Compute the top 10 movies for user 1.
  Col<size_t> users = { 1 };
  Mat<size_t> recommendations;
  cf.GetRecommendations(10, recommendations, users);

  // Now print each movie.
  cout << "Recommendations for user 1:" << endl;
  for (size_t i = 0; i < recommendations.n_elem; ++i)
  {
    cout << "  " << (i + 1) << ". "
        << moviesInfo.UnmapString(recommendations[i], 2) << "." << endl;
  }
}
```

This can be compiled the same way as before, assuming the code is saved as
`cpp_quickstart_2.cpp`:

```sh
g++ -O3 -std=c++14 -o cpp_quickstart_2 cpp_quickstart_2.cpp -fopenmp -larmadillo
```

And then it can be easily run:

```
./cpp_quickstart_2
```

Here is some example output, showing that user 1 seems to have good taste in
movies:

```
RMSE of trained model is 0.795323.
Recommendations for user 1:
  1: Casablanca (1942)
  2: Pan's Labyrinth (Laberinto del fauno, El) (2006)
  3: Godfather, The (1972)
  4: Answer This! (2010)
  5: Life Is Beautiful (La Vita è bella) (1997)
  6: Adventures of Tintin, The (2011)
  7: Dark Knight, The (2008)
  8: Out for Justice (1991)
  9: Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)
  10: Schindler's List (1993)
```

## Next steps with mlpack

Now that you have done some simple work with mlpack, you have seen how it can
easily plug into a data science production workflow in C++.  But these two
examples have only shown a little bit of the functionality of mlpack.  Lots of
other functionality is available.

Some of this functionality is demonstrated in the
[examples repository](https://github.com/mlpack/examples).

A full list of all classes and functions that mlpack implements can be found by
browsing the well-commented source code.
