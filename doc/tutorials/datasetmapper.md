# DatasetMapper tutorial

`DatasetMapper` is a class which holds information about a dataset. This can be
used when dataset contains categorical non-numeric features which should be
mapped to numeric features. A simple example can be

```
7,5,True,3
6,3,False,4
4,8,False,2
9,3,True,3
```

The above dataset will be represented as

```
7,5,0,3
6,3,1,4
4,8,1,2
9,3,0,3
```

Here the mappings are

- `True` mapped to `0`
- `False` mapped to `1`

**Note**: `DatasetMapper` converts non-numeric values in the order in which it
encounters them in the dataset. Therefore there is a chance that `True` might
get mapped to `0` if it encounters `True` before `False`.  This `0` and `1` are
not to be confused with C++ `bool` notations. These are mapping created by
`mlpack::DatasetMapper`.

`DatasetMapper` provides an easy API to load such data and stores all the
necessary information of the dataset.

## Loading data

To use `DatasetMapper` we have to call a specific overload of the `data::Load()`
function.

```c++
using namespace mlpack;

arma::mat data;
data::DatasetMapper info;
data::Load("dataset.csv", data, info);
```

Dataset:

```
7, 5, True, 3
6, 3, False, 4
4, 8, False, 2
9, 3, True, 3
```

## Dimensionality

There are two ways to initialize a DatasetMapper object.

* The first is to initialize the object and set each property yourself.

* The second is to pass the object to `Load()` without initialization, and
  mlpack will populate the object. If we use the latter option then the
  dimensionality will be same as what's in the data file.

```c++
std::cout << info.Dimensionality();
```

```
4
```

## Type of each dimension

Each dimension can be of either of the two types:

  - `data::Datatype::numeric`
  - `data::Datatype::categorical`

The function `Type(size_t dimension)` takes an argument dimension which is the
row number for which you want to know the type

This will return an enum `data::Datatype`, which is cast to `size_t` when we
print them using `std::cout`.

  - `0` represents `data::Datatype::numeric`
  - `1` represents `data::Datatype::categorical`

```c++
std::cout << info.Type(0) << "\n";
std::cout << info.Type(1) << "\n";
std::cout << info.Type(2) << "\n";
std::cout << info.Type(3) << "\n";
```

This produces:

```
0
0
1
0
```

## Number of mappings

If the type of a dimension is `data::Datatype::categorical`, then during
loading, each unique token in that dimension will be mapped to an integer
starting with `0`.

`NumMappings(size_t dimension)` takes `dimension` as an argument and returns the
number of mappings in that dimension, if the dimension is numeric, or there are
no mappings, then it will return 0.

```c++
std::cout << info.NumMappings(0) << "\n";
std::cout << info.NumMappings(1) << "\n";
std::cout << info.NumMappings(2) << "\n";
std::cout << info.NumMappings(3) << "\n";
```

will print:

```
0
0
2
0
```

## Checking mappings

There are two ways to check the mappings.

  - Enter the string to get mapped integer
  - Enter the mapped integer to get string

### `UnmapString()`

The `UnmapString()` function has the full signature `UnmapString(int value,
size_t dimension, size_t unmappingIndex = 0UL)`.

  - `value` is the integer for which you want to find the mapped value
  - `dimension` is the dimension in which you want to check the mappings

```c++
std::cout << info.UnmapString(0, 2) << "\n";
std::cout << info.UnmapString(1, 2) << "\n";
```

This will print:

```
T
F
```

### `UnmapValue()`

The `UnmapValue()` function has the signature `UnmapValue(const std::string
&input, size_t dimension)`.

  - `input` is the mapped value for which you want to find mapping
  - `dimension` is the dimension in which you want to find the mapped value

```c++
std::cout << info.UnmapValue("T", 2) << "\n";
std::cout << info.UnmapValue("F", 2) << "\n";
```

will produce:

```
0
1
```

## Further documentation

For further documentation on `DatasetMapper` and its uses, see the comments in
the source code in `src/mlpack/core/data/`, as well as its uses in the [examples
repository](https://github.com/mlpack/examples).
