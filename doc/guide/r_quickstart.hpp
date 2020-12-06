/**
 * @file r_quickstart.hpp
 * @author Yashwant Singh Parihar

@page r_quickstart mlpack in R quickstart guide

@section r_quickstart_intro Introduction

This page describes how you can quickly get started using mlpack from R and
gives a few examples of usage, and pointers to deeper documentation.

This quickstart guide is also available for @ref python_quickstart "Python",
@ref cli_quickstart "the command-line", @ref julia_quickstart "Julia" and
@ref go_quickstart "Go".

@section r_quickstart_install Installing mlpack binary package

Installing the mlpack bindings for R is straightforward; you can just use
CRAN:

@code{.R}
install.packages('mlpack')
@endcode

@section r_quickstart_source_install Installing mlpack package from source

Building the R bindings from scratch is a little more in-depth, though.  For
information on that, follow the instructions on the @ref build page, and be sure
to specify @c -DBUILD_R_BINDINGS=ON to CMake; you may need to also set the
location of the R program with @c -DR_EXECUTABLE=/path/to/R.

@section r_quickstart_example Simple mlpack quickstart example

As a really simple example of how to use mlpack from R, let's do some
simple classification on a subset of the standard machine learning @c covertype
dataset.  We'll first split the dataset into a training set and a testing set,
then we'll train an mlpack random forest on the training data, and finally we'll
print the accuracy of the random forest on the test dataset.

You can copy-paste this code directly into R to run it.

@code{.R}
if(!requireNamespace("data.table", quietly = TRUE)) { install.packages("data.table") }
suppressMessages({
    library("mlpack")
    library("data.table")
})

# Load the dataset from an online URL.  Replace with 'covertype.csv.gz' if you
# want to use on the full dataset.
df <- fread("https://www.mlpack.org/datasets/covertype-small.csv.gz")

# Split the labels.
labels <- df[, .(label)]
dataset <- df[, label:=NULL]

# Split the dataset using mlpack.
prepdata <- preprocess_split(input = dataset,
                             input_labels = labels,
                             test_ratio = 0.3,
                             verbose = TRUE)

# Train a random forest.
output <- random_forest(training = prepdata$training,
                        labels = prepdata$training_labels,
                        print_training_accuracy = TRUE,
                        num_trees = 10,
                        minimum_leaf_size = 3,
                        verbose = TRUE)
rf_model <- output$output_model

# Predict the labels of the test points.
output <- random_forest(input_model = rf_model,
                        test = prepdata$test,
                        verbose = TRUE)

# Now print the accuracy.  The third return value ('probabilities'), which we
# ignored here, could also be used to generate an ROC curve.
correct <- sum(output$predictions == prepdata$test_labels)
cat(correct, "out of", length(prepdata$test_labels), "test points correct",
    correct / length(prepdata$test_labels) * 100.0, "%\n")
@endcode

We can see that we achieve reasonably good accuracy on the test dataset (80%+);
if we use the full @c covertype.csv.gz, the accuracy should increase
significantly (but training will take longer).

It's easy to modify the code above to do more complex things, or to use
different mlpack learners, or to interface with other machine learning toolkits.

@section r_quickstart_whatelse What else does mlpack implement?

The example above has only shown a little bit of the functionality of mlpack.
Lots of other commands are available with different functionality.  A full list
of each of these commands and full documentation can be found on the following
page:

 - <a href="https://www.mlpack.org/doc/mlpack-git/r_documentation.html">r documentation</a>

For more information on what mlpack does, see https://www.mlpack.org/.
Next, let's go through another example for providing movie recommendations with
mlpack.

@section r_quickstart_movierecs Using mlpack for movie recommendations

In this example, we'll train a collaborative filtering model using mlpack's
<tt><a href="https://www.mlpack.org/doc/mlpack-git/r_documentation.html#cf">cf()</a></tt> method.  We'll train this on the MovieLens dataset from
https://grouplens.org/datasets/movielens/, and then we'll use the model that we
train to give recommendations.

You can copy-paste this code directly into R to run it.

@code{.R}
if(!requireNamespace("data.table", quietly = TRUE)) { install.packages("data.table") }
suppressMessages({
    library("mlpack")
    library("data.table")
})

# First, load the MovieLens dataset.  This is taken from files.grouplens.org/
# but reposted on mlpack.org as unpacked and slightly preprocessed data.
ratings <- fread("http://www.mlpack.org/datasets/ml-20m/ratings-only.csv.gz")
movies <- fread("http://www.mlpack.org/datasets/ml-20m/movies.csv.gz")

# Hold out 10% of the dataset into a test set so we can evaluate performance.
predata <- preprocess_split(input = ratings,
                            test_ratio = 0.1,
                            verbose = TRUE)

# Train the model.  Change the rank to increase/decrease the complexity of the
# model.
output <- cf(training = predata$training,
             test = predata$test,
             rank = 10,
             verbose = TRUE,
             max_iteration=2,
             algorithm = "RegSVD")
cf_model <- output$output_model

# Now query the 5 top movies for user 1.
output <- cf(input_model = cf_model,
             query = matrix(1),
             recommendations = 10,
             verbose = TRUE)

# Get the names of the movies for user 1.
cat("Recommendations for user 1:\n")
for (i in 1:10) {
  cat("  ", i, ":", as.character(movies[output$output[i], 3]), "\n")
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

@section r_quickstart_nextsteps Next steps with mlpack

After working through this overview to `mlpack`'s R package, we hope you are
inspired to use `mlpack`' in your data science workflow.  We recommend as part
of your next steps to look at more documentation for the R mlpack bindings:

 - <a href="https://www.mlpack.org/doc/mlpack-git/r_documentation.html">R mlpack
   binding documentation</a>

Also, mlpack is much more flexible from C++ and allows much greater
functionality.  So, more complicated tasks are possible if you are willing to
write C++ (or perhaps Rcpp).  To get started learning about mlpack in C++, the
following resources might be helpful:

 - <a href="https://www.mlpack.org/doc/mlpack-git/doxygen/tutorials.html">mlpack
   C++ tutorials</a>
 - <a href="https://www.mlpack.org/doc/mlpack-git/doxygen/build.html">mlpack
   build and installation guide</a>
 - <a href="https://www.mlpack.org/doc/mlpack-git/doxygen/sample.html">Simple
   sample C++ mlpack programs</a>
 - <a href="https://www.mlpack.org/doc/mlpack-git/doxygen/index.html">mlpack
   Doxygen documentation homepage</a>

 */
