# Dataset splitting

mlpack provides a simple functions for splitting a dataset into a training set
and a test set.

 * [`data::Split()`](#datasplitdata): split a dataset into a training set
   and test set, optionally with labels.

 * [`data::StratifiedSplit()`](#datastratifiedsplit): perform a stratified
   split, ensuring that the training and test set have the same ratios of each
   label.

---

## `data::SplitData()`

 * `data::Split(input, inputLabels, trainData, testData, trainLabels,
   testLabels, testRatio, shuffleData=true)`
   - Perform a standard train/test split with labels, with a factor of
     `testRatio` of the dataset stored in the test set.
   - If `shuffleData` is `false`, the first points in the dataset will be used
     for the training set, and the last points will be used for the test set.

 * `data::Split(input, trainData, testData, testRatio, shuffleData=true)`
   - Perform a standard train/test split without labels, with a factor of
     `testRatio` of the dataset stored in the test set.
   - If `shuffleData` is `false`, the first points in the dataset will be used
     for the training set, and the last points for the test set.

---

## `data::StratifiedSplit()`

 * `data::StratifiedSplit(input, inputLabels, trainData, testData, trainLabels,
   testLabels, testRatio, shuffleData=true)`
   - Perform a stratified train/test split, with a factor of `testRatio` of the
     dataset stored in the test set.
   - A stratified split ensures that the ratio of classes in the training and
     test sets matches the ratio of classes in the original dataset.  This can
     be useful for highly imbalanced datasets.
   - If `shuffleData` is `false`, the first points in the dataset for each class
     will be used for the training set, and the last points for the test set.

---

## Parameters

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `input` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) data matrix. | _(N/A)_ |
| `inputLabels` | [`arma::Row<size_t>`](../matrices.md) | Labels for data matrix.  Should have length `data.n_cols`. | _(N/A)_ |
| `trainData` | [`arma::mat&`](../matrices.md) | Matrix to store training points in.  Will be set to size `data.n_rows` x `((1.0 - testRatio) * data.n_cols)`. | _(N/A)_ |
| `testData` | [`arma::mat&`](../matrices.md) | Matrix to store test points in. Will be set to size `data.n_rows` x `testRatio * data.n_cols`. | _(N/A)_ |
| `trainLabels` | [`arma::Row<size_t>&`](../matrices.md) | Vector to store training labels in.  Will be set to length `trainData.n_cols`. | _(N/A)_ |
| `testLabels` | [`arma::Row<size_t>&`](../matrices.md) | Vector to store test labels in.  Will be set to length `testData.n_cols`. | _(N/A)_ |
| `testRatio` | `double` | Fraction of columns in `input` to use for test set. Typically between 0.1 and 0.25. | _(N/A)_ |
| `shuffleData` | `bool` | If `true`, then training and test sets are sampled randomly from `input`. | `true` |

***Notes:***

 - Any matrix type matching the Armadillo API can be used for `input`,
   `trainData`, and `testData` (e.g. `arma::fmat`, `arma::sp_mat`, etc.).
    * All three matrices must have the same type.
    * [`arma::field<>`](https://arma.sourceforge.net/docs.html#field) types can
      also be used.

 - Any dense vector or matrix type matching the Armadillo API can be used for
   `labels`, `trainLabels`, and `testLabels` (e.g. `arma::uvec`,
   `arma::Col<unsigned short>`, etc.).
    * All three label parameters must have the same type.
    * [`arma::field<>`](https://arma.sourceforge.net/docs.html#field) types may
      also be used, so long as the object type of the `field` is a vector type.

## Example usage

Split the unlabeled `cloud` dataset, using 20% of the dataset for the test set.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset, true);

arma::mat trainData, testData;

// Split the data, using 20% of the data for the test set.
mlpack::data::Split(dataset, trainData, testData, 0.2);

// Print the size of each matrix.
std::cout << "Full data size:     " << dataset.n_rows << " x " << dataset.n_cols
    << "." << std::endl;

std::cout << "Training data size: " << trainData.n_rows << " x "
    << trainData.n_cols << "." << std::endl;
std::cout << "Test data size:     " << testData.n_rows << " x "
    << testData.n_cols << "." << std::endl;
```

---

Split the mixed categorical `telecom_churn` dataset and associated responses for
regression, using 25% of the dataset for the test set.  Use 32-bit floating
point elements to represent both the data and responses.

```c++
// See https://datasets.mlpack.org/telecom_churn.arff.
arma::fmat dataset;
mlpack::data::DatasetInfo info; // Holds which dimensions are categorical.
mlpack::data::Load("telecom_churn.arff", dataset, info, true);

// See https://datasets.mlpack.org/telecom_churn.responses.csv.
arma::frowvec labels;
mlpack::data::Load("telecom_churn.responses.csv", labels, true);

arma::fmat trainData, testData;
arma::frowvec trainLabels, testLabels;

// Split the data, using 25% of the data for the test set.
// Note that Split() can accept many different types for the data and the
// labels---here we pass arma::frowvec instead of arma::Row<size_t>.
mlpack::data::Split(dataset, labels, trainData, testData, trainLabels,
    testLabels, 0.25);

// Print the size of each matrix.
std::cout << "Full data size:       " << dataset.n_rows << " x "
    << dataset.n_cols << "." << std::endl;
std::cout << "Full labels size:     " << labels.n_elem << "." << std::endl;

std::cout << std::endl;

std::cout << "Training data size:   " << trainData.n_rows << " x "
    << trainData.n_cols << "." << std::endl;
std::cout << "Training labels size: " << trainLabels.n_elem << "." << std::endl;
std::cout << "Test data size:     " << testData.n_rows << " x "
    << testData.n_cols << "." << std::endl;
std::cout << "Test labels size:   " << testLabels.n_elem << "." << std::endl;
```

---

Split the `movielens` dataset, which is a sparse matrix.  Don't shuffle when
splitting.

```c++
// See https://datasets.mlpack.org/movielens-100k.csv.
arma::sp_mat dataset;
mlpack::data::Load("movielens-100k.csv", dataset, true);

arma::sp_mat trainData, testData;

// Split the dataset without shuffling.
mlpack::data::Split(dataset, trainData, testData, 0.2, false);

// Print the first point of the dataset and the training set (these will be the
// same because we did not shuffle during splitting).
std::cout << "First point of full dataset:" << std::endl;
std::cout << dataset.col(0).t() << std::endl;

std::cout << "First point of training set:" << std::endl;
std::cout << trainData.col(0).t() << std::endl;

// Print the last point of the dataset and the test set (these will also be the
// same).
std::cout << "Last point of full dataset:" << std::endl;
std::cout << dataset.col(dataset.n_cols - 1).t() << std::endl;

std::cout << "Last point of test set:" << std::endl;
std::cout << testData.col(testData.n_cols - 1).t() << std::endl;

```

---

Perform a stratified sampling of the `covertype` dataset, printing the
percentage of each class in the original dataset and in the split datasets.

```c++
// See https://datasets.mlpack.org/covertype.data.csv.
arma::mat dataset;
mlpack::data::Load("covertype.data.csv", dataset, true);

// See https://datasets.mlpack.org/covertype.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("covertype.labels.csv", labels, true);

arma::mat trainData, testData;
arma::Row<size_t> trainLabels, testLabels;

// Perform a stratified split, keeping 15% of the data as a test set.
mlpack::data::StratifiedSplit(dataset, labels, trainData, testData, trainLabels,
    testLabels, 0.15);

// Now compute the percentage of each label in the dataset.
const size_t numClasses = arma::max(labels) + 1;
arma::vec classPercentages(numClasses);
for (size_t i = 0; i < labels.n_elem; ++i)
  ++classPercentages[(size_t) labels[i]];
classPercentages /= labels.n_elem;

std::cout << "Percentages of each class in the full dataset:" << std::endl;
for (size_t i = 0; i < numClasses; ++i)
{
  std::cout << " - Class " << i << ": " << 100.0 * classPercentages[i] << "%."
      << std::endl;
}

// Now compute the percentage of each label in the training set.
classPercentages.zeros();
for (size_t i = 0; i < trainLabels.n_elem; ++i)
  ++classPercentages[(size_t) trainLabels[i]];
classPercentages /= trainLabels.n_elem;

std::cout << "Percentages of each class in the training set:" << std::endl;
for (size_t i = 0; i < numClasses; ++i)
{
  std::cout << " - Class " << i << ": " << 100.0 * classPercentages[i] << "%."
      << std::endl;
}

// Finally compute the percentage of each label in the test set.
classPercentages.zeros();
for (size_t i = 0; i < testLabels.n_elem; ++i)
  ++classPercentages[(size_t) testLabels[i]];
classPercentages /= testLabels.n_elem;

std::cout << "Percentages of each class in the training set:" << std::endl;
for (size_t i = 0; i < numClasses; ++i)
{
  std::cout << " - Class " << i << ": " << 100.0 * classPercentages[i] << "%."
      << std::endl;
}
```
