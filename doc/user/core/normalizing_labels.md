# Normalizing labels

mlpack classifiers and other algorithms require labels to be in the range `0` to
`numClasses - 1`.  A vector of labels with arbitrary (`size_t`) values can be
normalized to the required range with the
[`NormalizeLabels()`](#datanormalizelabels) function, and reverted to the
original range with the [`RevertLabels()`](#datarevertlabels) function.

---

## `data::NormalizeLabels()`

 * `data::NormalizeLabels(labelsIn, labelsOut, mappings)`
    - Map vector `labelsIn` into the range `0` to `numClasses - 1`, storing as
      `labelsOut` (of type `arma::Row<size_t>`).
      * `numClasses` is automatically detected using the number of unique values
        in `labelsIn`. 

    - The column vector `mappings` will be filled with the reverse mappings to
      convert back to the old labels; this can be used by `RevertLabels()`.

    - `mappings[i]` contains the original class label for the mapped label `i`.

---

## `data::RevertLabels()`

 * `data::RevertLabels(labelsIn, mappings, labelsOut)`
    - Unmap normalized labels `labelsIn` using `mappings` into `labelsOut`.

    - Performs the reverse operation of `NormalizeLabels()`; `mappings` should
      be the same vector output by `NormalizeLabels()`.

---

## Example

Convert labels into `0`, `1`, `2`, learn a model, then convert predictions back
to the original label values.

```c++
// Create a random dataset with 5 points in 10 dimensions.
arma::mat dataset(10, 5, arma::fill::randu);

// Manually assemble labels vector: [3, 7, 3, 3, 5]
arma::Row<size_t> labels = { 3, 7, 3, 3, 5 };

// Note that these labels are not in the range `0` to `2`, and thus cannot be
// used directly by mlpack classifiers!
// We will map them to that range using NormalizeLabels().
arma::Row<size_t> mappedLabels;
arma::Col<size_t> mappings;
mlpack::data::NormalizeLabels(labels, mappedLabels, mappings);
const size_t numClasses = mappedLabels.max() + 1;

// Print the mapped values:
// [3, 7, 3, 3, 5] maps to [0, 1, 0, 0, 2].
// The `mappings` vector will be [3, 7, 5].
std::cout << "Original labels: " << labels;
std::cout << "Mapped labels:   " << mappedLabels;
std::cout << std::endl;
std::cout << "Mappings: " << mappings.t();
std::cout << std::endl << std::endl;

// Learn a model with the mapped labels.
mlpack::DecisionTree d(dataset, mappedLabels, numClasses, 1 /* leaf size */);

// Make predictions on the training dataset.
arma::Row<size_t> mappedPredictions;
d.Classify(dataset, mappedPredictions);

// The predictions use mapped labels (0, 1, 2), which we will need to map back
// to the original labels using RevertLabels().
arma::Row<size_t> predictions;
mlpack::data::RevertLabels(mappedPredictions, mappings, predictions);

// Print the predictions before and after unmapping.
// The mapped predictions will take values 0, 1, or 2; the predictions will take
// values 3, 7, or 5 (like the original data).
std::cout << "Mapped predictions: " << mappedPredictions;
std::cout << "Predictions:        " << predictions;
```
