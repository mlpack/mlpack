/**
 * @file dbscan_main.cpp
 * @author Ryan Curtin
 *
 * Implementation of program to run DBSCAN.
 */
// #include "dbscan.hpp"
#include "hdbscan.hpp"

#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/rectangle_tree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>

using namespace mlpack;
using namespace mlpack::metric;
using namespace mlpack::tree;
using namespace std;

PROGRAM_INFO("DBSCAN clustering",
    "This program implements the DBSCAN algorithm for clustering.");

PARAM_STRING_IN_REQ("input_file", "Input dataset to cluster.", "i");
PARAM_STRING_OUT("assignments_file", "Output file for assignments of each "
    "point.", "a");
PARAM_STRING_OUT("centroids_file", "File to save output centroids to.", "C");

PARAM_DOUBLE_IN("epsilon", "Radius of each range search.", "e", 1.0);
PARAM_INT_IN("min_size", "Minimum number of points for a cluster.", "m", 5);

PARAM_STRING_IN("tree_type", "If using single-tree or dual-tree search, the "
    "type of tree to use ('kd', 'r', 'r-star', 'x', 'hilbert-r', 'r-plus', "
    "'r-plus-plus', 'cover', 'ball').", "t", "kd");
PARAM_FLAG("single_mode", "If set, single-tree range search (not dual-tree) "
    "will be used.", "S");
PARAM_FLAG("naive", "If set, brute-force range search (not tree-based) "
    "will be used.", "N");


int main(int argc, char** argv)
{
  
}
