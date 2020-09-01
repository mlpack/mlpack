/**
 * @file go_quickstart.hpp
 * @author Yashwant Singh Parihar

@page go_quickstart mlpack in Go quickstart guide

@section go_quickstart_intro Introduction

This page describes how you can quickly get started using mlpack from Go and
gives a few examples of usage, and pointers to deeper documentation.

This quickstart guide is also available for @ref python_quickstart "Python",
@ref cli_quickstart "the command-line", @ref julia_quickstart "Julia" and
@ref r_quickstart "R".

@section go_quickstart_install Installing mlpack

Installing the mlpack bindings for Go is somewhat time-consuming as the library
must be built; you can run the following code:

@code{.sh}
go get -u -d mlpack.org/v1/mlpack
cd ${GOPATH}/src/mlpack.org/v1/mlpack
make install
@endcode

Building the Go bindings from scratch is a little more in-depth, though.  For
information on that, follow the instructions on the @ref build page, and be sure
to specify @c -DBUILD_GO_BINDINGS=ON to CMake;

@section go_quickstart_example Simple mlpack quickstart example

As a really simple example of how to use mlpack from Go, let's do some
simple classification on a subset of the standard machine learning @c covertype
dataset.  We'll first split the dataset into a training set and a testing set,
then we'll train an mlpack random forest on the training data, and finally we'll
print the accuracy of the random forest on the test dataset.

You can copy-paste this code directly into main.go to run it.
@code{.go}
package main

import (
  "mlpack.org/v1/mlpack"
  "fmt"
)
func main() {

  // Download dataset.
  mlpack.DownloadFile("https://www.mlpack.org/datasets/covertype-small.data.csv.gz",
                      "data.csv.gz")
  mlpack.DownloadFile("https://www.mlpack.org/datasets/covertype-small.labels.csv.gz",
                      "labels.csv.gz")

  // Extract/Unzip the dataset.
  mlpack.UnZip("data.csv.gz", "data.csv")
  dataset, _ := mlpack.Load("data.csv")

  mlpack.UnZip("labels.csv.gz", "labels.csv")
  labels, _ := mlpack.Load("labels.csv")

  // Split the dataset using mlpack.
  params := mlpack.PreprocessSplitOptions()
  params.InputLabels = labels
  params.TestRatio = 0.3
  params.Verbose = true
  test, test_labels, train, train_labels :=
      mlpack.PreprocessSplit(dataset, params)

  // Train a random forest.
  rf_params := mlpack.RandomForestOptions()
  rf_params.NumTrees = 10
  rf_params.MinimumLeafSize = 3
  rf_params.PrintTrainingAccuracy = true
  rf_params.Training = train
  rf_params.Labels = train_labels
  rf_params.Verbose = true
  rf_model, _, _ := mlpack.RandomForest(rf_params)

  // Predict the labels of the test points.
  rf_params_2 := mlpack.RandomForestOptions()
  rf_params_2.Test = test
  rf_params_2.InputModel = &rf_model
  rf_params_2.Verbose = true
  _, predictions, _ := mlpack.RandomForest(rf_params_2)

  // Now print the accuracy.
  rows, _ := predictions.Dims()
  var sum int = 0
  for i := 0; i < rows; i++ {
    if (predictions.At(i, 0) == test_labels.At(i, 0)) {
      sum = sum + 1
    }
  }
  fmt.Print(sum, " correct out of ", rows, " (",
      (float64(sum) / float64(rows)) * 100, "%).\n")
}
@endcode

We can see that we achieve reasonably good accuracy on the test dataset (80%+);
if we use the full @c covertype.csv.gz, the accuracy should increase
significantly (but training will take longer).

It's easy to modify the code above to do more complex things, or to use
different mlpack learners, or to interface with other machine learning toolkits.

@section go_quickstart_whatelse What else does mlpack implement?

The example above has only shown a little bit of the functionality of mlpack.
Lots of other commands are available with different functionality.  A full list
of each of these commands and full documentation can be found on the following
page:

 - <a href="https://www.mlpack.org/doc/mlpack-git/go_documentation.html">Go documentation</a>

You can also use the GoDoc to explore the @c mlpack module and its
functions; every function comes with comprehensive documentation.

For more information on what mlpack does, see https://www.mlpack.org/.
Next, let's go through another example for providing movie recommendations with
mlpack.

@section go_quickstart_movierecs Using mlpack for movie recommendations

In this example, we'll train a collaborative filtering model using mlpack's
<tt><a href="https://www.mlpack.org/doc/mlpack-git/go_documentation.html#cf">Cf()</a></tt> method.  We'll train this on the MovieLens dataset from
https://grouplens.org/datasets/movielens/, and then we'll use the model that we
train to give recommendations.

You can copy-paste this code directly into main.go to run it.

@code{.go}
package main

import (
  "github.com/frictionlessdata/tableschema-go/csv"
  "mlpack.org/v1/mlpack"
  "gonum.org/v1/gonum/mat"
  "fmt"
)
func main() {

  // Download dataset.
  mlpack.DownloadFile("https://www.mlpack.org/datasets/ml-20m/ratings-only.csv.gz",
                      "ratings-only.csv.gz")
  mlpack.DownloadFile("https://www.mlpack.org/datasets/ml-20m/movies.csv.gz",
                      "movies.csv.gz")

  // Extract dataset.
  mlpack.UnZip("ratings-only.csv.gz", "ratings-only.csv")
  ratings, _ := mlpack.Load("ratings-only.csv")

  mlpack.UnZip("movies.csv.gz", "movies.csv")
  table, _ := csv.NewTable(csv.FromFile("movies.csv"), csv.LoadHeaders())
  movies, _ := table.ReadColumn("title")

  // Split the dataset using mlpack.
  params := mlpack.PreprocessSplitOptions()
  params.TestRatio = 0.1
  params.Verbose = true
  ratings_test, _, ratings_train, _ := mlpack.PreprocessSplit(ratings, params)

  // Train the model.  Change the rank to increase/decrease the complexity of the
  // model.
  cf_params := mlpack.CfOptions()
  cf_params.Training = ratings_train
  cf_params.Test = ratings_test
  cf_params.Rank = 10
  cf_params.Verbose = true
  cf_params.Algorithm = "RegSVD"
  _, cf_model := mlpack.Cf(cf_params)

  // Now query the 5 top movies for user 1.
  cf_params_2 := mlpack.CfOptions()
  cf_params_2.InputModel = &cf_model
  cf_params_2.Recommendations = 10
  cf_params_2.Query = mat.NewDense(1, 1, []float64{1})
  cf_params_2.Verbose = true
  cf_params_2.MaxIterations = 10
  output, _ := mlpack.Cf(cf_params_2)

  // Get the names of the movies for user 1.
  fmt.Println("Recommendations for user 1")
  for i := 0; i < 10; i++ {
    fmt.Println(i, ":", movies[int(output.At(0 , i))])
  }
}
@endcode

Here is some example output, showing that user 1 seems to have good taste in
movies:

@code{.unparsed}
Recommendations for user 1:
  0: Casablanca (1942)
  1: Pan's Labyrinth (Laberinto del fauno, El) (2006)
  2: Godfather, The (1972)
  3: Answer This! (2010)
  4: Life Is Beautiful (La Vita è bella) (1997)
  5: Adventures of Tintin, The (2011)
  6: Dark Knight, The (2008)
  7: Out for Justice (1991)
  8: Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)
  9: Schindler's List (1993)
@endcode

@section go_quickstart_nextsteps Next steps with mlpack

Now that you have done some simple work with mlpack, you have seen how it can
easily plug into a data science workflow in Go.  A great thing to do next
would be to look at more documentation for the Go mlpack bindings:

 - <a href="https://www.mlpack.org/doc/mlpack-git/go_documentation.html">Go mlpack
   binding documentation</a>

Also, mlpack is much more flexible from C++ and allows much greater
functionality.  So, more complicated tasks are possible if you are willing to
write C++.  To get started learning about mlpack in C++, the following resources
might be helpful:

 - <a href="https://www.mlpack.org/doc/mlpack-git/doxygen/tutorials.html">mlpack
   C++ tutorials</a>
 - <a href="https://www.mlpack.org/doc/mlpack-git/doxygen/build.html">mlpack
   build and installation guide</a>
 - <a href="https://www.mlpack.org/doc/mlpack-git/doxygen/sample.html">Simple
   sample C++ mlpack programs</a>
 - <a href="https://www.mlpack.org/doc/mlpack-git/doxygen/index.html">mlpack
   Doxygen documentation homepage</a>

 */
