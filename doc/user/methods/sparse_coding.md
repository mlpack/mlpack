## `SparseCoding`

The `SparseCoding` class implements sparse coding with dictionary learning using
L1 regularization or elastic net (L1+L2) regularization.  Sparse coding is a
form of representation learning, and can be used to represent each point in a
dataset as a sparse combination of *atoms* in the dictionary.

#### Simple usage example:

```c++
// Create a random dataset with 100 points in 40 dimensions, and then a random
// test dataset with 50 points.
arma::mat data(40, 100, arma::fill::randu);
arma::mat testData(40, 50, arma::fill::randu);

// Perform sparse coding with 20 atoms and an L1 penalty of 0.5.
mlpack::SparseCoding sc(20, 0.5);  // Step 1: create object.
double objective = sc.Train(data); // Step 2: learn dictionary.
arma::mat codes;
sc.Encode(testData, codes);        // Step 3: encode new data.

// Print some information about the test encoding.
std::cout << "Average density of encoded test data: "
    << 100.0 * arma::mean(arma::sum(codes != 0)) / codes.n_rows << "\%."
    << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `SparseCoding` objects.
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

 * [`LocalCoordinateCoding`](local_coordinate_coding.md)
 * [`LARS`](lars.md) (used internally by `SparseCoding`)
 * [mlpack transformations](../transformations.md)
 * [Sparse dictionary learning on Wikipedia](https://en.wikipedia.org/wiki/Sparse_dictionary_learning)
 * [Efficient sparse coding algorithms (pdf)](https://proceedings.neurips.cc/paper/2006/file/2d71b2ae158c7c5912cc0bbde2bb9d95-Paper.pdf)

### Constructors

 * `sc = SparseCoding()`
 * `sc = SparseCoding(atoms=0, lambda1=0.0, lambda2=0.0, maxIter=0, objTol=0.01, newtonTol=1e-6)`
   - Create a `SparseCoding` object without learning a dictionary on data.
   - If `atoms` is set to `0` (the default), it will need to be set to a value
     greater than `0` before `Train()` is called (`sc.Atoms() = atoms` can be
     used for this).

 * `sc = SparseCoding(data, atoms, lambda1=0.0, lambda2=0.0, maxIter=0, objTol=0.01, newtonTol=1e-6)`
   - Create a `SparseCoding` object and train the dictionary on the given
     `data`.
   - The dictionary will contain `atoms` elements.

 * `sc = SparseCoding(data, atoms, lambda1, lambda2, maxIter, objTol, newtonTol, initializer)`
   - *Advanced constructor*: create a `SparseCoding` object that will use a
     custom dictionary initializer and train on the given `data`.
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
| `lambda1` | `double` | L1 regularization penalty.  Used in both `Train()` and `Encode()` steps. | `0.0` |
| `lambda2` | `double` | L2 regularization penalty (for elastic net regularization).  Used in both `Train()` and `Encode()` steps. | `0.0` |
| `maxIter` | `size_t` | Maximum number of iterations for dictionary learning.  `0` means no limit. | `0` |
| `objTol` | `double` | Objective function tolerance for terminating dictionary learning. | `0.01` |
| `newtonTol` | `double` | Tolerance for the Newton's method dictionary optimization step. | `1e-6` |

As an alternative to passing `atoms`, `lambda1`, `lambda2`, `maxIter`, `objTol`,
or `newtonTol`, these can be set with a standalone method.  The following
functions can be used before calling `Train()`:

 * `sc.Atoms() = a;` will set the number of atoms to use in the dictionary to
   `a`.  Changing this after calling `Train()` will not make a difference to the
   dictionary size.

 * `sc.Lambda1() = l1;` will set the L1 regularization penalty to `l1`.  This
   can be set after `Train()` to force sparser encodings when `Encode()` is
   called.

 * `sc.Lambda2() = l2;` will set the L2 regularization penalty to `l2`.  This
   can be set after `Train()` to increase the regularization when `Encode()` is
   called.

 * `sc.MaxIterations() = m;` will set the maximum number of iterations for
   dictionary learning to `m`.  `0` means that the algorithm will run until
   convergence.

 * `sc.ObjTolerance() = ot;` will set the objective tolerance for convergence of
   the dictionary learning algorithm to `ot`.

 * `sc.NewtonTolerance() = nt;` will set the tolerance for the Newton's method
   dictionary optimization step to `nt`.

***Caveats***:

 * Larger settings of `atoms` (i.e. larger dictionary sizes) will be able to
   more accurately represent the data, but may take longer to learn.

 * Larger values of `lambda1` will cause the model to use sparser codings for
   data when `Train()` and `Encode()` are called, but codings that are too
   sparse may be inaccurate representations of the original points.

 * Training is not incremental; a second call to `Train()` will reinitialize the
   dictionary and restart the learning process.

### Training

If training the dictionary is not done as part of the constructor call, it can
be done with one of the following versions of the `Train()` member function:

 * `sc.Train(data)`
 * `sc.Train(data, initializer)`
   - Train the sparse coding dictionary on the given `data`.
   - Optionally, use the given `initializer` to initialize the dictionary (see
     [`DictionaryInitializer`](#dictionaryinitializer-different-dictionary-initialization-strategies)
     for more details).

### Encoding

Once a `SparseCoding` model has a trained dictionary, the `Encode()` member
function can be used to encode new data points.

 * `sc.Encode(data, codes)`
   - Encode `data` (a
     [column-major data matrix](../matrices.md#representing-data-in-mlpack))
     as a set of sparse codes of the dictionary, storing the result in `codes`.
   - Both `data` and `codes` should be the same matrix type (e.g. `arma::mat`);
     see [Different Element Types](#mattype-different-element-types) for more
     details.
   - `codes` will be set to have `atoms` rows and `data.n_cols` columns.
   - Column `i` of `codes` corresponds to the sparse coding of the `i`'th column
     of `data`.  Each row represents the weight associated with each atom in
     the dictionary.

After encoding, the original data can be recovered (approximately) as
`sc.Dictionary() * data`.

### Other Functionality

 * A `SparseCoding` model can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `sc.Dictionary()` will return an `arma::mat&` containing the dictionary
   matrix.  The matrix has `data.n_rows` rows and `atoms` columns; each column
   corresponds to an atom in the dictionary.

 * `double obj = sc.Objective(data, codes)` computes the sparse coding objective
   function on the given `data` and encodings `codes`.  This can be used after
   `Encode()` to test the quality of the encodings (a smaller objective is
   better).

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial usage
of the `SparseCoding` class.

---

Train a sparse coding model on the cloud dataset and print the reconstruction
error.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset, true);

mlpack::SparseCoding sc;
sc.Atoms() = 50;
sc.Lambda1() = 0.1;
sc.Lambda2() = 0.001;
sc.MaxIterations() = 25;
sc.Train(dataset);

// Encode the training dataset.
arma::mat codes;
sc.Encode(dataset, codes);

std::cout << "Input matrix size: " << dataset.n_rows << " x " << dataset.n_cols
    << "." << std::endl;
std::cout << "Codes matrix size: " << codes.n_rows << " x " << codes.n_cols
    << "." << std::endl;

// Reconstruct the original matrix.
arma::mat recon = sc.Dictionary() * codes;
double error = std::sqrt(arma::norm(dataset - recon, "fro") / dataset.n_elem);
std::cout << "RMSE of reconstructed matrix: " << error << "." << std::endl;
```

---

Train a sparse coding model on the iris dataset and save the model to disk.

```c++
// See https://datasets.mlpack.org/iris.train.csv.
arma::mat dataset;
mlpack::data::Load("iris.train.csv", dataset, true);

// Train the model in the constructor.
mlpack::SparseCoding sc(dataset, 10 /* atoms */, 0.1 /* L1 penalty */);

// Save the model to disk.
mlpack::data::Save("sc.bin", "sc", sc);
```

---

Load a sparse coding model from disk and encode some new points from the
iris dataset.

```c++
// Load model from disk.
mlpack::SparseCoding sc;
mlpack::data::Load("sc.bin", "sc", sc);

// See https://datasets.mlpack.org/iris.test.csv.
arma::mat dataset;
mlpack::data::Load("iris.test.csv", dataset, true);

// Encode the test points.
arma::mat codes;
sc.Encode(dataset, codes);

// Compute the sparse coding objective on the test points.
const double obj = sc.Objective(dataset, codes);
std::cout << "Sparse coding objective on test set: " << obj << "." << std::endl;
```

---

Train a sparse coding model on the satellite dataset, trying several different
dictionary sizes and checking the objective value on a held-out test dataset.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat trainData;
mlpack::data::Load("satellite.train.csv", trainData, true);
// See https://datasets.mlpack.org/satellite.test.csv.
arma::mat testData;
mlpack::data::Load("satellite.test.csv", testData, true);

for (size_t atoms = 20; atoms < 100; atoms += 10)
{
  mlpack::SparseCoding sc(atoms);
  sc.Lambda1() = 0.1;
  sc.MaxIterations() = 20; // Keep iterations low so this runs relatively fast.

  const double trainObj = sc.Train(trainData);

  // Compute the objective on the test set.
  arma::mat codes;
  sc.Encode(testData, codes);
  const double testObj = sc.Objective(testData, codes);

  std::cout << "Atoms: " << std::setfill(' ') << std::setw(3) << atoms << "; ";
  std::cout << "training set objective: " << std::setw(6) << trainObj << "; ";
  std::cout << "test set objective: " << std::setw(6) << testObj << "."
      << std::endl;
}
```

### Advanced Functionality: Template Parameters

The `SparseCoding` class has one class template parameter that can be used for
custom behavior.  The full signature of the class is:

```
SparseCoding<MatType>
```

In addition, the [constructors](#constructors) and
[`Train()` functions](#training) have a template parameter
`DictionaryInitializer` that can be used for custom behavior.

 * `MatType`: the type of the matrix to use (e.g. `arma::mat`, `arma::fmat`,
   etc.).  The given `MatType` must support the Armadillo API and hold a
   floating-point element type (e.g. `float`, `double`, etc.).

 * `DictionaryInitializer`: the strategy used to initialize the dictionary.  By
   default, `DataDependentRandomInitializer` is used.

#### `MatType`: Different Element Types

`MatType` specifies the type of matrix used for training data and internal
representation of the dictionary.  Any matrix type that implements the Armadillo
API can be used.  The example below trains a sparse coding model on 32-bit
floating point data.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::fmat dataset;
mlpack::data::Load("cloud.csv", dataset, true);

mlpack::SparseCoding<arma::fmat> sc;
sc.Atoms() = 30;
sc.Lambda1() = 0.15;
sc.Lambda2() = 0.001;
sc.MaxIterations() = 100;
// Note: a looser tolerance is often required when using floats instead of
// doubles.
sc.ObjTolerance() = 0.01;
sc.Train(dataset);

// Encode the training dataset.
arma::fmat codes;
sc.Encode(dataset, codes);

std::cout << "Input matrix size: " << dataset.n_rows << " x " << dataset.n_cols
    << "." << std::endl;
std::cout << "Codes matrix size: " << codes.n_rows << " x " << codes.n_cols
    << "." << std::endl;

// Reconstruct the original matrix.
arma::fmat recon = sc.Dictionary() * codes;
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
const double lambda1 = 0.1;
const double lambda2 = 0.0;
const size_t maxIterations = 50;

// Use a uniform random matrix as the initial dictionary.
arma::mat initialDictionary(trainData.n_rows, atoms, arma::fill::randu);

mlpack::SparseCoding sc(atoms, lambda1, lambda2, maxIterations);
sc.Dictionary() = initialDictionary;

const double obj = sc.Train<mlpack::NothingInitializer>(trainData);
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
  // dataset.  MatType will be the matrix type used by the sparse coding model
  // (e.g. `arma::mat`, `arma::fmat`, etc.).
  template<typename MatType>
  void Initialize(const MatType& data,
                  const size_t atoms,
                  MatType& dictionary);
};
```
