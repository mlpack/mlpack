## `LocalCoordinateCoding`

The `LocalCoordinateCoding` class implements local coordinate coding, a
variation of [sparse coding](sparse_coding.md) with dictionary learning.  Local
coordinate coding is a form of representation learning, and can be used to
represent each point in a dataset as a linear combination of a few nearby
*atoms* in the learned dictionary.

#### Simple usage example:

```c++
// Create a random dataset with 100 points in 40 dimensions, and then a random
// test dataset with 50 points.
arma::mat data(40, 100, arma::fill::randn);
arma::mat testData(40, 50, arma::fill::randn);

// Perform local coordinate coding with 20 atoms and an L1 penalty of 0.1.
mlpack::LocalCoordinateCoding lcc(20, 0.1); // Step 1: create object.
double objective = lcc.Train(data);         // Step 2: learn dictionary.
arma::mat codes;
lcc.Encode(testData, codes);                // Step 3: encode new data.

// Print some information about the test encoding.
std::cout << "Average density of encoded test data: "
    << 100.0 * arma::mean(arma::sum(codes != 0)) / codes.n_rows << "\%."
    << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `LocalCoordinateCoding` objects.
 * [`Train()`](#training): train model (learn dictionary).
 * [`Encode()`](#encoding): encode points with a trained model.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-template-parameters) for
   advanced functionality: different element types and dictionary initialization
   strategies.

#### See also:

 * [`SparseCoding`](sparse_coding.md)
 * [`LARS`](lars.md) (used internally by `LocalCoordinateCoding`)
 * [mlpack transformations](../transformations.md)
 * [Sparse dictionary learning on Wikipedia](https://en.wikipedia.org/wiki/Sparse_dictionary_learning)
 * [Nonlinear learning using local coordinate coding (pdf)](https://proceedings.neurips.cc/paper_files/paper/2009/file/2afe4567e1bf64d32a5527244d104cea-Paper.pdf)

### Constructors

 * `lcc = LocalCoordinateCoding()`
 * `lcc = LocalCoordinateCoding(atoms=0, lambda=0.0, maxIter=0, tol=0.01)`
   - Create a `LocalCoordinateCoding` object without learning a dictionary on
     data.
   - If `atoms` is set to `0` (the default), it will need to be set to a value
     greater than `0` before `Train()` is called (`lcc.Atoms() = atoms` can be
     used for this).

 * `lcc = LocalCoordinateCoding(data, atoms, lambda=0.0, maxIter=0, tol=0.01)`
   - Create a `LocalCoordinateCoding` object and train the dictionary on the
     given `data`.
   - The dictionary will contain `atoms` elements.

 * `lcc = LocalCoordinateCoding(data, atoms, lambda, maxIter, tol, initializer)`
   - *Advanced constructor*: create a `LocalCoordinateCoding` object that will
     use a custom dictionary initializer and train on the given `data`.
   - The dictionary will contain `atoms` elements.
   - `initializer` will be used to initialize the dictionary; see [Advanced
     Functionality: Different Dictionary Initialization
     Strategies](#dictionaryinitializer-different-dictionary-initialization-strategies)
     for details.

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `atoms` | `size_t` | Number of atoms in dictionary. | _(N/A)_ |
| `lambda` | `double` | L1 regularization penalty.  Used in both `Train()` and `Encode()` steps. | `0.0` |
| `maxIter` | `size_t` | Maximum number of iterations for dictionary learning.  `0` means no limit. | `0` |
| `tol` | `double` | Objective function tolerance for terminating dictionary learning. | `0.01` |

As an alternative to passing `atoms`, `lambda`, `maxIter`, or `tol`, these can
be set with a standalone method.  The following functions can be used before
calling `Train()`:

 * `lcc.Atoms() = a;` will set the number of atoms to use in the dictionary to
   `a`.  Changing this after calling `Train()` will not make a difference to the
   dictionary size.

 * `lcc.Lambda() = l;` will set the L1 regularization penalty to `l1`.  This can
   be set after `Train()` to force sparser encodings when `Encode()` is called.

 * `lcc.MaxIterations() = m;` will set the maximum number of iterations for
   dictionary learning to `m`.  `0` means that the algorithm will run until
   convergence.

 * `lcc.Tolerance() = t;` will set the objective tolerance for convergence of
   the dictionary learning algorithm to `t`.

***Caveats***:

 * Larger settings of `atoms` (i.e. larger dictionary sizes) will be able to
   more accurately represent the data, but may take longer to learn.

 * Larger values of `lambda` will cause the model to use sparser encodings for
   data (e.g. fewer nearby anchor points) when `Train()` and `Encode()` are
   called, but when `lambda` is too large, the codings may be inaccurate
   representations of the original points.

<!-- TODO: indicate that you can get this info with MLPACK_PRINT_INFO and
MLPACK_PRINT_WARN, once those are documented -->

 * If `lambda` is set too large, encodings may be empty (e.g. all zeros).

 * Training is not incremental; a second call to `Train()` will reinitialize the
   dictionary and restart the learning process.

### Training

If training the dictionary is not done as part of the constructor call, it can
be done with one of the following versions of the `Train()` member function:

 * `lcc.Train(data)`
 * `lcc.Train(data, initializer)`
   - Train the local coordinate coding dictionary on the given `data`.
   - Optionally, use the given `initializer` to initialize the dictionary (see
     [`DictionaryInitializer`](#dictionaryinitializer-different-dictionary-initialization-strategies)
     for more details).

### Encoding

Once a `LocalCoordinateCoding` model has a trained dictionary, the `Encode()`
member function can be used to encode new data points.

 * `lcc.Encode(data, codes)`
   - Encode `data` (a [column-major data
     matrix](../matrices.md#representing-data-in-mlpack)) as a sparse set of
     local atoms of the dictionary, storing the result in `codes`.
   - Both `data` and `codes` should be the same matrix type (e.g. `arma::mat`);
     see [Different Element Types](#mattype-different-element-types) for more
     details.
   - `codes` will be set to have `atoms` rows and `data.n_cols` columns.
   - Column `i` of `codes` corresponds to the coding of the `i`'th column of
     `data`.  Each row represents the weight associated with each atom in the
     dictionary.

After encoding, the original data can be recovered (approximately) as
`lcc.Dictionary() * data`.

### Other Functionality

 * A `LocalCoordinateCoding` model can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `lcc.Dictionary()` will return an `arma::mat&` containing the dictionary
   matrix.  The matrix has `data.n_rows` rows and `atoms` columns; each column
   corresponds to an atom in the dictionary.  Dictionary atoms are regularized
   to be close to the manifold that data lie on.

 * `double obj = lcc.Objective(data, codes)` computes the local coordinate
   coding objective function on the given `data` and encodings `codes`.  This
   can be used after `Encode()` to test the quality of the encodings (a smaller
   objective is better).

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial usage
of the `LocalCoordinateCoding` class.

---

Train a local coordinate coding model on the cloud dataset and print the
reconstruction error.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset, true);

mlpack::LocalCoordinateCoding lcc;
lcc.Atoms() = 50;
lcc.Lambda() = 1e-5;
lcc.MaxIterations() = 25;
lcc.Train(dataset);

// Encode the training dataset.
arma::mat codes;
lcc.Encode(dataset, codes);

std::cout << "Input matrix size: " << dataset.n_rows << " x " << dataset.n_cols
    << "." << std::endl;
std::cout << "Codes matrix size: " << codes.n_rows << " x " << codes.n_cols
    << "." << std::endl;

// Reconstruct the original matrix.
arma::mat recon = lcc.Dictionary() * codes;
double error = std::sqrt(arma::norm(dataset - recon, "fro") / dataset.n_elem);
std::cout << "RMSE of reconstructed matrix: " << error << "." << std::endl;
```

---

Train a local coordinate coding model on the iris dataset and save the model to
disk.

```c++
// See https://datasets.mlpack.org/iris.train.csv.
arma::mat dataset;
mlpack::data::Load("iris.train.csv", dataset, true);

// Train the model in the constructor.
mlpack::LocalCoordinateCoding lcc(dataset,
                                  10 /* atoms */,
                                  0.1 /* L1 penalty */);

// Save the model to disk.
mlpack::data::Save("lcc.bin", "lcc", lcc);
```

---

Train a local coordinate coding model on the satellite dataset, trying several
different regularization parameters and checking the objective value on a
held-out test dataset.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat trainData;
mlpack::data::Load("satellite.train.csv", trainData, true);
// See https://datasets.mlpack.org/satellite.test.csv.
arma::mat testData;
mlpack::data::Load("satellite.test.csv", testData, true);

for (double lambdaPow = -6; lambdaPow <= -2; lambdaPow += 1)
{
  const double lambda = std::pow(10.0, lambdaPow);
  mlpack::LocalCoordinateCoding lcc(50 /* atoms */);
  lcc.Lambda() = lambda;
  lcc.MaxIterations() = 25; // Keep iterations low so this runs relatively fast.

  const double trainObj = lcc.Train(trainData);

  // Compute the objective on the test set.
  arma::mat codes;
  lcc.Encode(testData, codes);
  const double testObj = lcc.Objective(testData, codes);

  std::cout << "Lambda: " << std::setfill(' ') << std::setw(3) << lambda
      << "; ";
  std::cout << "training set objective: " << std::setw(6) << trainObj << "; ";
  std::cout << "test set objective: " << std::setw(6) << testObj << "."
      << std::endl;
}
```

### Advanced Functionality: Template Parameters

The `LocalCoordinateCoding` class has one class template parameter that can be
used for custom behavior.  The full signature of the class is:

```
LocalCoordinateCoding<MatType>
```

In addition, the [constructors](#constructors) and [`Train()`
functions](#training) have a template parameter `DictionaryInitializer` that can
be used for custom behavior.

 * `MatType`: the type of the matrix to use (e.g. `arma::mat`, `arma::fmat`,
   etc.).  The given `MatType` must support the Armadillo API and hold a
   floating-point element type (e.g. `float`, `double`, etc.).

 * `DictionaryInitializer`: the strategy used to initialize the dictionary.  By
   default, `DataDependentRandomInitializer` is used.

#### `MatType`: Different Element Types

`MatType` specifies the type of matrix used for training data and internal
representation of the dictionary.  Any matrix type that implements the Armadillo
API can be used.  The example below trains a local coordinate coding model on
32-bit floating point data.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::fmat dataset;
mlpack::data::Load("cloud.csv", dataset, true);

mlpack::LocalCoordinateCoding<arma::fmat> lcc;
lcc.Atoms() = 30;
lcc.Lambda() = 1e-5;
lcc.MaxIterations() = 100;
lcc.Train(dataset);

// Encode the training dataset.
arma::fmat codes;
lcc.Encode(dataset, codes);

std::cout << "Input matrix size: " << dataset.n_rows << " x " << dataset.n_cols
    << "." << std::endl;
std::cout << "Codes matrix size: " << codes.n_rows << " x " << codes.n_cols
    << "." << std::endl;

// Reconstruct the original matrix.
arma::fmat recon = lcc.Dictionary() * codes;
double error = std::sqrt(arma::norm(dataset - recon, "fro") / dataset.n_elem);
std::cout << "RMSE of reconstructed matrix: " << error << "." << std::endl;
```

#### `DictionaryInitializer`: Different Dictionary Initialization Strategies

The `DictionaryInitializer` template class specifies the strategy to be used to
initialize the dictionary when `Train()` is called.

 * The `DataDependentRandomInitalizer` class (the default) uses the average of
   three random points in the dataset to initialize each atom in the dictionary.

 * The `NothingInitializer` class does not modify the dictionary matrix in any
   way, and could be used either to set a specific dictionary before training
   with `sc.Dictionary()`, or to allow incremental training that does not modify
   the existing dictionary when `Train()` is called a second time.

 * The `RandomInitializer` class initializes the dictionary by sampling norm-1
   atoms from a normal distribution.

***Note:*** none of the classes above have any members, and as such it is not
necessary to use the constructor or `Train()` variants that take an initialized
`initializer` object.  That would only be necessary for a custom
`DictionaryInitializer` class that stored internal members.

---

The example below uses `NothingInitializer` to set a specific initial
dictionary.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat trainData;
mlpack::data::Load("satellite.train.csv", trainData, true);

const size_t atoms = 25;
const double lambda = 1e-5;
const size_t maxIterations = 50;

// Use a uniform random matrix as the initial dictionary.
arma::mat initialDictionary(trainData.n_rows, atoms, arma::fill::randu);

mlpack::LocalCoordinateCoding lcc(atoms, lambda, maxIterations);
lcc.Dictionary() = initialDictionary;

const double obj = lcc.Train<mlpack::NothingInitializer>(trainData);
std::cout << "Training set objective: " << obj << "." << std::endl;
```

---

 * An entirely custom class can also be implemented.  The class must implement
   one method, `Initialize()`:

```c++
// You can use this as a starting point for implementation.
class CustomDictionaryInitializer
{
 public:
  // Initialize the dictionary to have the given number of atoms, given the
  // dataset.  MatType will be the matrix type used by the local coordinate
  // coding model (e.g. `arma::mat`, `arma::fmat`, etc.).
  template<typename MatType>
  void Initialize(const MatType& data,
                  const size_t atoms,
                  MatType& dictionary);
};
```
