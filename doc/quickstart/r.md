# mlpack in R quickstart guide

This page describes how you can get started using mlpack from R, and
gives a few examples of usage as well as pointers to additional documentation.

This quickstart guide is also available for [C++](cpp.md), [Python](python.md),
[Julia](julia.md), [the command line](cli.md), and [Go](go.md).

## Installing mlpack

Installing the mlpack bindings for R is straightforward as you can use
CRAN:

```r
install.packages('mlpack')
```

Building the R bindings from scratch is a little more in-depth, though.  For
information on that, follow the instructions in the
[installation guide](../user/install.md#compile-bindings-manually). Note that
the mlpack R binding turn off support for the STB, DR_LIBS, and HTTPLIB
libraries which may be of lesser interst to R users. If desired, support at
the C++ level can be enabled by setting per-library `#define` statements.

## Simple mlpack quickstart example

As a really simple example of how to use mlpack from R, let us do some
simple classification on a subset of the standard machine learning `covertype`
dataset.  We will first split the dataset into a training set and a testing set,
then we will train an mlpack random forest on the training data, and finally
we will print the accuracy of the random forest on the test dataset.

You can copy-paste this code directly into R to run it.

```r
if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
suppressMessages({
    library("mlpack")
    library("data.table")
})

# Load the dataset from an online URL.  Replace with 'covertype.csv.gz' if you
# want to use on the full dataset.
url <- "https://www.mlpack.org/datasets/covertype-small.csv.gz"
dataset <- fread(url, showProgress=FALSE)

# Split the labels.
labels <- dataset[, label]  # extract column 'label' as a vector
dataset[, label := NULL]    # remove column 'label'

# Split the dataset using mlpack.
prepdata <- preprocess_split(input = dataset,
                             input_labels = labels,
                             test_ratio = 0.3,
                             verbose = TRUE)

# Train a random forest.
rf_model <- random_forest_train(training = prepdata$training,
                                labels = prepdata$training_labels,
                                print_training_accuracy = TRUE,
                                num_trees = 10,
                                minimum_leaf_size = 3,
                                verbose = TRUE)

# Predict the labels of the test points.
output <- predict(rf_model, newdata = prepdata$test)

# Now print the accuracy.  The third return value ('probabilities'), which we
# ignored here, could also be used to generate an ROC curve.
correct <- sum(output == prepdata$test_labels)
cat(correct, "out of", length(prepdata$test_labels), "test points correct",
    correct / length(prepdata$test_labels) * 100.0, "%\n")
```

We can see that we achieve reasonably good accuracy on the test dataset (80%+);
if we use the full `covertype.csv.gz`, the accuracy should increase
significantly (but training will take longer).

It is easy to modify the code above to do more complex things, or to use
different mlpack learners, or to interface with other machine learning toolkits.

## Using mlpack for movie recommendations

In this example, we will train a collaborative filtering model using mlpack's
[`cf()`](../user/bindings/r.md#cf) method.
We will train this on the
[MovieLens dataset](https://grouplens.org/datasets/movielens/), and then we will
use the model that we train to give recommendations.

You can copy-paste this code directly into R to run it.

```r
if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
suppressMessages({
    library("mlpack")
    library("data.table")
})

# First, load the MovieLens dataset.  This is taken from files.grouplens.org/
# but reposted on mlpack.org as unpacked and slightly preprocessed data.
ratings <- fread("http://www.mlpack.org/datasets/ml-20m/ratings-only.csv.gz",
                 showProgress=FALSE)
movies <- fread("http://www.mlpack.org/datasets/ml-20m/movies.csv.gz",
                showProgress=FALSE)

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
  cat("  ", i, ":", movies[output$output[i], title], "\n")
}
```

Here is some example output, showing that user 1 seems to have good taste in
movies:

```
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
```

## Next steps with mlpack

After working through this overview to `mlpack`'s R package, we hope you are
inspired to use `mlpack`' in your data science workflow.  However, the two
examples above have only shown a little bit of the functionality of mlpack.
Lots of other functions are available with different functionality.  A full list
of each of these functions and full documentation can be found on the following
page:

 - [R documentation](../user/bindings/r.md)

Also, mlpack is much more flexible from C++ and allows much greater
functionality.  So, more complicated tasks are possible if you are willing to
write C++ (perhaps relying on Rcpp).  To get started learning about mlpack in
C++, a good starting point is the [C++ quickstart guide](cpp.md).
