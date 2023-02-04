# EMST Tutorial

The Euclidean Minimum Spanning Tree problem is widely used in machine learning
and data mining applications.  Given a set `S` of points in `R^d`, our task is
to compute lowest weight spanning tree in the complete graph on `S` with edge
weights given by the Euclidean distance between points.

Among other applications, the EMST can be used to compute hierarchical
clusterings of data.  A *single-linkage clustering* can be obtained from the
EMST by deleting all edges longer than a given cluster length.  This technique
is also referred to as a *Friends-of-Friends* clustering in the astronomy
literature.

mlpack includes an implementation of ***Dual-Tree Boruvka*** which uses
`kd`-trees by default; this is the empirically and theoretically fastest EMST
algorithm.  In addition, the implementation supports the use of different trees
via templates.  For more details, see the following paper:

```
@inproceedings{march2010fast,
  title={Fast {E}uclidean minimum spanning tree: algorithm, analysis, and
applications},
  author={March, William B. and Ram, Parikshit and Gray, Alexander G.},
  booktitle={Proceedings of the 16th ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining (KDD '10)},
  pages={603--612},
  year={2010},
  organization={ACM}
}
```

mlpack provides:

 - a simple command-line executable to compute the EMST of a given data set
 - a simple C++ interface to compute the EMST

## Command-line `mlpack_emst`

The `mlpack_emst` program in mlpack will compute the EMST of a given set
of points and store the resulting edge list to a file.  Note that mlpack also
has bindings to other languages, and so there also exists, e.g., an `emst()`
function in Python and other similar functions in other languages.  Although
these examples are written for the command-line `mlpack_emst` program, it is
easy to adapt each of these to another language.

The output file contains an edge list representation of the MST in an `(n - 1) x
3` matrix, where the first and second columns are labels of points and the third
column is the edge weight.  The edges are sorted in order of increasing weight.

Below are several examples of simple usage (and the resultant output).  The `-v`
option is used so that verbose output is given.  Further documentation on each
individual option can be found by typing

```sh
$ mlpack_emst --help
```

```sh
$ mlpack_emst --input_file=dataset.csv --output_file=edge_list.csv -v
[INFO ] Reading in data.
[INFO ] Loading 'dataset.csv' as CSV data.
[INFO ] Data read, building tree.
[INFO ] Tree built, running algorithm.
[INFO ] 4 edges found so far.
[INFO ] 5 edges found so far.
[INFO ] Total spanning tree length: 1002.45
[INFO ] Saving CSV data to 'edge_list.csv'.
[INFO ]
[INFO ] Execution parameters:
[INFO ]   help: false
[INFO ]   info: ""
[INFO ]   input_file: dataset.csv
[INFO ]   leaf_size: 1
[INFO ]   naive: false
[INFO ]   output_file: edge_list.csv
[INFO ]   verbose: true
[INFO ]
[INFO ] Program timers:
[INFO ]   emst/mst_computation: 0.000179s
[INFO ]   emst/tree_building: 0.000061s
[INFO ]   total_time: 0.052641s
```

The code performs at most `log N` iterations for `N` data points.  It will print
an update on the number of MST edges found after each iteration.  Convenient
program timers are given for different parts of the calculation at the bottom of
the output, as well as the parameters the simulation was run with.

```sh
$ cat dataset.csv
0, 0
1, 1
3, 3
0.5, 0
1000, 0
1001, 0

$ cat edge_list.csv
0.0000000000e+00,3.0000000000e+00,5.0000000000e-01
4.0000000000e+00,5.0000000000e+00,1.0000000000e+00
1.0000000000e+00,3.0000000000e+00,1.1180339887e+00
1.0000000000e+00,2.0000000000e+00,2.8284271247e+00
2.0000000000e+00,4.0000000000e+00,9.9700451353e+02
```

The input points are labeled 0-5.  The output tells us that the MST connects
point 0 to point 3, point 4 to point 5, point 1 to point 3, point 1 to point 2,
and point 2 to point 4, with the corresponding edge weights given in the third
column.  The total length of the MST is also given in the verbose output.

Note that it is also possible to compute the EMST using a naive (`O(N^2)`)
algorithm for timing and comparison purposes, using the `--naive` option.

## The `DualTreeBoruvka` class

The `DualTreeBoruvka` class contains our implementation of the Dual-Tree Boruvka
algorithm.

The class has two constructors: the first takes the data set, constructs the
tree (where the type of tree constructed is the TreeType template parameter),
and computes the MST.  The second takes data set and an already constructed
tree.

The class provides one method that performs the MST computation:

```c++
void ComputeMST(const arma::mat& results);
```

This method stores the computed MST in the matrix results in the format given
above.

## Further documentation

For further documentation on the `DualTreeBoruvka` class, consult the comments
in the source code, in `mlpack/methods/emst/dtb.hpp`.
