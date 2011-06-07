/**
 * @file allknn_test.cc
 *
 * Test file for AllkNN class
 */

#include <fastlib/fastlib.h>
#include <armadillo>
#include "allknn.h"

#define BOOST_TEST_MODULE AllkNN Test
#include <boost/test/unit_test.hpp>

using namespace mlpack::allknn;

/***
 * Simple nearest-neighbors test with small, synthetic dataset.  This is an
 * exhaustive test, which checks that each method for performing the calculation
 * (dual-tree, single-tree, naive) produces the correct results.  An
 * eleven-point dataset and the ten nearest neighbors are taken.  The dataset is
 * in one dimension for simplicity -- the correct functionality of distance
 * functions is not tested here.
 */
BOOST_AUTO_TEST_CASE(exhaustive_synthetic_test) {
  // Set up our data.
  arma::mat data(1, 11);
  data[0] = 0.05; // Row addressing is unnecessary (they are all 0).
  data[1] = 0.35;
  data[2] = 0.15;
  data[3] = 1.25;
  data[4] = 5.05;
  data[5] = -0.20;
  data[6] = -2.00;
  data[7] = -1.30;
  data[8] = 0.45;
  data[9] = 0.90;
  data[10] = 1.00;

  // We will loop through three times, one for each method of performing the
  // calculation.
  for (int i = 0; i < 3; i++) {
    AllkNN* allknn;
    arma::mat data_mutable = data;
    switch(i) {
      case 0: // Use the dual-tree method.
        allknn = new AllkNN(data_mutable, 20, 10);
        break;
      case 1: // Use the single-tree method.
        allknn = new AllkNN(data_mutable, 20, 10, AllkNN::MODE_SINGLE);
        break;
      case 2: // Use the naive method.
        allknn = new AllkNN(data_mutable, 1, 10, AllkNN::NAIVE);
        break;
    }

    // Now perform the actual calculation.
    arma::Col<index_t> neighbors;
    arma::vec distances;
    allknn->ComputeNeighbors(neighbors, distances);

    // Now the exhaustive check for correctness.  This will be long.  We must
    // also remember that the distances returned are squared distances.  As a
    // result, distance comparisons are written out as (distance * distance) for
    // readability.

    // Neighbors of point 0.
    BOOST_REQUIRE(neighbors[0] == 2);
    BOOST_REQUIRE_CLOSE(distances[0], (0.10 * 0.10), 1e-5);
    BOOST_REQUIRE(neighbors[1] == 5);
    BOOST_REQUIRE_CLOSE(distances[1], (0.25 * 0.25), 1e-5);
    BOOST_REQUIRE(neighbors[2] == 1);
    BOOST_REQUIRE_CLOSE(distances[2], (0.30 * 0.30), 1e-5);
    BOOST_REQUIRE(neighbors[3] == 8);
    BOOST_REQUIRE_CLOSE(distances[3], (0.40 * 0.40), 1e-5);
    BOOST_REQUIRE(neighbors[4] == 9);
    BOOST_REQUIRE_CLOSE(distances[4], (0.85 * 0.85), 1e-5);
    BOOST_REQUIRE(neighbors[5] == 10);
    BOOST_REQUIRE_CLOSE(distances[5], (0.95 * 0.95), 1e-5);
    BOOST_REQUIRE(neighbors[6] == 3);
    BOOST_REQUIRE_CLOSE(distances[6], (1.20 * 1.20), 1e-5);
    BOOST_REQUIRE(neighbors[7] == 7);
    BOOST_REQUIRE_CLOSE(distances[7], (1.35 * 1.35), 1e-5);
    BOOST_REQUIRE(neighbors[8] == 6);
    BOOST_REQUIRE_CLOSE(distances[8], (2.05 * 2.05), 1e-5);
    BOOST_REQUIRE(neighbors[9] == 4);
    BOOST_REQUIRE_CLOSE(distances[9], (5.00 * 5.00), 1e-5);
    
    // Neighbors of point 1.
    BOOST_REQUIRE(neighbors[10] == 8);
    BOOST_REQUIRE_CLOSE(distances[10], (0.10 * 0.10), 1e-5);
    BOOST_REQUIRE(neighbors[11] == 2);
    BOOST_REQUIRE_CLOSE(distances[11], (0.20 * 0.20), 1e-5);
    BOOST_REQUIRE(neighbors[12] == 0);
    BOOST_REQUIRE_CLOSE(distances[12], (0.30 * 0.30), 1e-5);
    BOOST_REQUIRE(neighbors[13] == 9);
    BOOST_REQUIRE_CLOSE(distances[13], (0.55 * 0.55), 1e-5);
    BOOST_REQUIRE(neighbors[14] == 5);
    BOOST_REQUIRE_CLOSE(distances[14], (0.55 * 0.55), 1e-5);
    BOOST_REQUIRE(neighbors[15] == 10);
    BOOST_REQUIRE_CLOSE(distances[15], (0.65 * 0.65), 1e-5);
    BOOST_REQUIRE(neighbors[16] == 3);
    BOOST_REQUIRE_CLOSE(distances[16], (0.90 * 0.90), 1e-5);
    BOOST_REQUIRE(neighbors[17] == 7);
    BOOST_REQUIRE_CLOSE(distances[17], (1.65 * 1.65), 1e-5);
    BOOST_REQUIRE(neighbors[18] == 6);
    BOOST_REQUIRE_CLOSE(distances[18], (2.35 * 2.35), 1e-5);
    BOOST_REQUIRE(neighbors[19] == 4);
    BOOST_REQUIRE_CLOSE(distances[19], (4.70 * 4.70), 1e-5);

    // Neighbors of point 2.
    BOOST_REQUIRE(neighbors[20] == 0);
    BOOST_REQUIRE_CLOSE(distances[20], (0.10 * 0.10), 1e-5);
    BOOST_REQUIRE(neighbors[21] == 1);
    BOOST_REQUIRE_CLOSE(distances[21], (0.20 * 0.20), 1e-5);
    BOOST_REQUIRE(neighbors[22] == 8);
    BOOST_REQUIRE_CLOSE(distances[22], (0.30 * 0.30), 1e-5);
    BOOST_REQUIRE(neighbors[23] == 5);
    BOOST_REQUIRE_CLOSE(distances[23], (0.35 * 0.35), 1e-5);
    BOOST_REQUIRE(neighbors[24] == 9);
    BOOST_REQUIRE_CLOSE(distances[24], (0.75 * 0.75), 1e-5);
    BOOST_REQUIRE(neighbors[25] == 10);
    BOOST_REQUIRE_CLOSE(distances[25], (0.85 * 0.85), 1e-5);
    BOOST_REQUIRE(neighbors[26] == 3);
    BOOST_REQUIRE_CLOSE(distances[26], (1.10 * 1.10), 1e-5);
    BOOST_REQUIRE(neighbors[27] == 7);
    BOOST_REQUIRE_CLOSE(distances[27], (1.45 * 1.45), 1e-5);
    BOOST_REQUIRE(neighbors[28] == 6);
    BOOST_REQUIRE_CLOSE(distances[28], (2.15 * 2.15), 1e-5);
    BOOST_REQUIRE(neighbors[29] == 4);
    BOOST_REQUIRE_CLOSE(distances[29], (4.90 * 4.90), 1e-5);

    // Neighbors of point 3.
    BOOST_REQUIRE(neighbors[30] == 10);
    BOOST_REQUIRE_CLOSE(distances[30], (0.25 * 0.25), 1e-5);
    BOOST_REQUIRE(neighbors[31] == 9);
    BOOST_REQUIRE_CLOSE(distances[31], (0.35 * 0.35), 1e-5);
    BOOST_REQUIRE(neighbors[32] == 8);
    BOOST_REQUIRE_CLOSE(distances[32], (0.80 * 0.80), 1e-5);
    BOOST_REQUIRE(neighbors[33] == 1);
    BOOST_REQUIRE_CLOSE(distances[33], (0.90 * 0.90), 1e-5);
    BOOST_REQUIRE(neighbors[34] == 2);
    BOOST_REQUIRE_CLOSE(distances[34], (1.10 * 1.10), 1e-5);
    BOOST_REQUIRE(neighbors[35] == 0);
    BOOST_REQUIRE_CLOSE(distances[35], (1.20 * 1.20), 1e-5);
    BOOST_REQUIRE(neighbors[36] == 5);
    BOOST_REQUIRE_CLOSE(distances[36], (1.45 * 1.45), 1e-5);
    BOOST_REQUIRE(neighbors[37] == 7);
    BOOST_REQUIRE_CLOSE(distances[37], (2.55 * 2.55), 1e-5);
    BOOST_REQUIRE(neighbors[38] == 6);
    BOOST_REQUIRE_CLOSE(distances[38], (3.25 * 3.25), 1e-5);
    BOOST_REQUIRE(neighbors[39] == 4);
    BOOST_REQUIRE_CLOSE(distances[39], (3.80 * 3.80), 1e-5);
    
    // Neighbors of point 4.
    BOOST_REQUIRE(neighbors[40] == 3);
    BOOST_REQUIRE_CLOSE(distances[40], (3.80 * 3.80), 1e-5);
    BOOST_REQUIRE(neighbors[41] == 10);
    BOOST_REQUIRE_CLOSE(distances[41], (4.05 * 4.05), 1e-5);
    BOOST_REQUIRE(neighbors[42] == 9);
    BOOST_REQUIRE_CLOSE(distances[42], (4.15 * 4.15), 1e-5);
    BOOST_REQUIRE(neighbors[43] == 8);
    BOOST_REQUIRE_CLOSE(distances[43], (4.60 * 4.60), 1e-5);
    BOOST_REQUIRE(neighbors[44] == 1);
    BOOST_REQUIRE_CLOSE(distances[44], (4.70 * 4.70), 1e-5);
    BOOST_REQUIRE(neighbors[45] == 2);
    BOOST_REQUIRE_CLOSE(distances[45], (4.90 * 4.90), 1e-5);
    BOOST_REQUIRE(neighbors[46] == 0);
    BOOST_REQUIRE_CLOSE(distances[46], (5.00 * 5.00), 1e-5);
    BOOST_REQUIRE(neighbors[47] == 5);
    BOOST_REQUIRE_CLOSE(distances[47], (5.25 * 5.25), 1e-5);
    BOOST_REQUIRE(neighbors[48] == 7);
    BOOST_REQUIRE_CLOSE(distances[48], (6.35 * 6.35), 1e-5);
    BOOST_REQUIRE(neighbors[49] == 6);
    BOOST_REQUIRE_CLOSE(distances[49], (7.05 * 7.05), 1e-5);
    
    // Neighbors of point 5.
    BOOST_REQUIRE(neighbors[50] == 0);
    BOOST_REQUIRE_CLOSE(distances[50], (0.25 * 0.25), 1e-5);
    BOOST_REQUIRE(neighbors[51] == 2);
    BOOST_REQUIRE_CLOSE(distances[51], (0.35 * 0.35), 1e-5);
    BOOST_REQUIRE(neighbors[52] == 1);
    BOOST_REQUIRE_CLOSE(distances[52], (0.55 * 0.55), 1e-5);
    BOOST_REQUIRE(neighbors[53] == 8);
    BOOST_REQUIRE_CLOSE(distances[53], (0.65 * 0.65), 1e-5);
    BOOST_REQUIRE(neighbors[54] == 9);
    BOOST_REQUIRE_CLOSE(distances[54], (1.10 * 1.10), 1e-5);
    BOOST_REQUIRE(neighbors[55] == 7);
    BOOST_REQUIRE_CLOSE(distances[55], (1.10 * 1.10), 1e-5);
    BOOST_REQUIRE(neighbors[56] == 10);
    BOOST_REQUIRE_CLOSE(distances[56], (1.20 * 1.20), 1e-5);
    BOOST_REQUIRE(neighbors[57] == 3);
    BOOST_REQUIRE_CLOSE(distances[57], (1.45 * 1.45), 1e-5);
    BOOST_REQUIRE(neighbors[58] == 6);
    BOOST_REQUIRE_CLOSE(distances[58], (1.80 * 1.80), 1e-5);
    BOOST_REQUIRE(neighbors[59] == 4);
    BOOST_REQUIRE_CLOSE(distances[59], (5.25 * 5.25), 1e-5);
    
    // Neighbors of point 6.
    BOOST_REQUIRE(neighbors[60] == 7);
    BOOST_REQUIRE_CLOSE(distances[60], (0.70 * 0.70), 1e-5);
    BOOST_REQUIRE(neighbors[61] == 5);
    BOOST_REQUIRE_CLOSE(distances[61], (1.80 * 1.80), 1e-5);
    BOOST_REQUIRE(neighbors[62] == 0);
    BOOST_REQUIRE_CLOSE(distances[62], (2.05 * 2.05), 1e-5);
    BOOST_REQUIRE(neighbors[63] == 2);
    BOOST_REQUIRE_CLOSE(distances[63], (2.15 * 2.15), 1e-5);
    BOOST_REQUIRE(neighbors[64] == 1);
    BOOST_REQUIRE_CLOSE(distances[64], (2.35 * 2.35), 1e-5);
    BOOST_REQUIRE(neighbors[65] == 8);
    BOOST_REQUIRE_CLOSE(distances[65], (2.45 * 2.45), 1e-5);
    BOOST_REQUIRE(neighbors[66] == 9);
    BOOST_REQUIRE_CLOSE(distances[66], (2.90 * 2.90), 1e-5);
    BOOST_REQUIRE(neighbors[67] == 10);
    BOOST_REQUIRE_CLOSE(distances[67], (3.00 * 3.00), 1e-5);
    BOOST_REQUIRE(neighbors[68] == 3);
    BOOST_REQUIRE_CLOSE(distances[68], (3.25 * 3.25), 1e-5);
    BOOST_REQUIRE(neighbors[69] == 4);
    BOOST_REQUIRE_CLOSE(distances[69], (7.05 * 7.05), 1e-5);
    
    // Neighbors of point 7.
    BOOST_REQUIRE(neighbors[70] == 6);
    BOOST_REQUIRE_CLOSE(distances[70], (0.70 * 0.70), 1e-5);
    BOOST_REQUIRE(neighbors[71] == 5);
    BOOST_REQUIRE_CLOSE(distances[71], (1.10 * 1.10), 1e-5);
    BOOST_REQUIRE(neighbors[72] == 0);
    BOOST_REQUIRE_CLOSE(distances[72], (1.35 * 1.35), 1e-5);
    BOOST_REQUIRE(neighbors[73] == 2);
    BOOST_REQUIRE_CLOSE(distances[73], (1.45 * 1.45), 1e-5);
    BOOST_REQUIRE(neighbors[74] == 1);
    BOOST_REQUIRE_CLOSE(distances[74], (1.65 * 1.65), 1e-5);
    BOOST_REQUIRE(neighbors[75] == 8);
    BOOST_REQUIRE_CLOSE(distances[75], (1.75 * 1.75), 1e-5);
    BOOST_REQUIRE(neighbors[76] == 9);
    BOOST_REQUIRE_CLOSE(distances[76], (2.20 * 2.20), 1e-5);
    BOOST_REQUIRE(neighbors[77] == 10);
    BOOST_REQUIRE_CLOSE(distances[77], (2.30 * 2.30), 1e-5);
    BOOST_REQUIRE(neighbors[78] == 3);
    BOOST_REQUIRE_CLOSE(distances[78], (2.55 * 2.55), 1e-5);
    BOOST_REQUIRE(neighbors[79] == 4);
    BOOST_REQUIRE_CLOSE(distances[79], (6.35 * 6.35), 1e-5);
    
    // Neighbors of point 8.
    BOOST_REQUIRE(neighbors[80] == 1);
    BOOST_REQUIRE_CLOSE(distances[80], (0.10 * 0.10), 1e-5);
    BOOST_REQUIRE(neighbors[81] == 2);
    BOOST_REQUIRE_CLOSE(distances[81], (0.30 * 0.30), 1e-5);
    BOOST_REQUIRE(neighbors[82] == 0);
    BOOST_REQUIRE_CLOSE(distances[82], (0.40 * 0.40), 1e-5);
    BOOST_REQUIRE(neighbors[83] == 9);
    BOOST_REQUIRE_CLOSE(distances[83], (0.45 * 0.45), 1e-5);
    BOOST_REQUIRE(neighbors[84] == 10);
    BOOST_REQUIRE_CLOSE(distances[84], (0.55 * 0.55), 1e-5);
    BOOST_REQUIRE(neighbors[85] == 5);
    BOOST_REQUIRE_CLOSE(distances[85], (0.65 * 0.65), 1e-5);
    BOOST_REQUIRE(neighbors[86] == 3);
    BOOST_REQUIRE_CLOSE(distances[86], (0.80 * 0.80), 1e-5);
    BOOST_REQUIRE(neighbors[87] == 7);
    BOOST_REQUIRE_CLOSE(distances[87], (1.75 * 1.75), 1e-5);
    BOOST_REQUIRE(neighbors[88] == 6);
    BOOST_REQUIRE_CLOSE(distances[88], (2.45 * 2.45), 1e-5);
    BOOST_REQUIRE(neighbors[89] == 4);
    BOOST_REQUIRE_CLOSE(distances[89], (4.60 * 4.60), 1e-5);
    
    // Neighbors of point 9.
    BOOST_REQUIRE(neighbors[90] == 10);
    BOOST_REQUIRE_CLOSE(distances[90], (0.10 * 0.10), 1e-5);
    BOOST_REQUIRE(neighbors[91] == 3);
    BOOST_REQUIRE_CLOSE(distances[91], (0.35 * 0.35), 1e-5);
    BOOST_REQUIRE(neighbors[92] == 8);
    BOOST_REQUIRE_CLOSE(distances[92], (0.45 * 0.45), 1e-5);
    BOOST_REQUIRE(neighbors[93] == 1);
    BOOST_REQUIRE_CLOSE(distances[93], (0.55 * 0.55), 1e-5);
    BOOST_REQUIRE(neighbors[94] == 2);
    BOOST_REQUIRE_CLOSE(distances[94], (0.75 * 0.75), 1e-5);
    BOOST_REQUIRE(neighbors[95] == 0);
    BOOST_REQUIRE_CLOSE(distances[95], (0.85 * 0.85), 1e-5);
    BOOST_REQUIRE(neighbors[96] == 5);
    BOOST_REQUIRE_CLOSE(distances[96], (1.10 * 1.10), 1e-5);
    BOOST_REQUIRE(neighbors[97] == 7);
    BOOST_REQUIRE_CLOSE(distances[97], (2.20 * 2.20), 1e-5);
    BOOST_REQUIRE(neighbors[98] == 6);
    BOOST_REQUIRE_CLOSE(distances[98], (2.90 * 2.90), 1e-5);
    BOOST_REQUIRE(neighbors[99] == 4);
    BOOST_REQUIRE_CLOSE(distances[99], (4.15 * 4.15), 1e-5);

    // Neighbors of point 10.
    BOOST_REQUIRE(neighbors[100] == 9);
    BOOST_REQUIRE_CLOSE(distances[100], (0.10 * 0.10), 1e-5);
    BOOST_REQUIRE(neighbors[101] == 3);
    BOOST_REQUIRE_CLOSE(distances[101], (0.25 * 0.25), 1e-5);
    BOOST_REQUIRE(neighbors[102] == 8);
    BOOST_REQUIRE_CLOSE(distances[102], (0.55 * 0.55), 1e-5);
    BOOST_REQUIRE(neighbors[103] == 1);
    BOOST_REQUIRE_CLOSE(distances[103], (0.65 * 0.65), 1e-5);
    BOOST_REQUIRE(neighbors[104] == 2);
    BOOST_REQUIRE_CLOSE(distances[104], (0.85 * 0.85), 1e-5);
    BOOST_REQUIRE(neighbors[105] == 0);
    BOOST_REQUIRE_CLOSE(distances[105], (0.95 * 0.95), 1e-5);
    BOOST_REQUIRE(neighbors[106] == 5);
    BOOST_REQUIRE_CLOSE(distances[106], (1.20 * 1.20), 1e-5);
    BOOST_REQUIRE(neighbors[107] == 7);
    BOOST_REQUIRE_CLOSE(distances[107], (2.30 * 2.30), 1e-5);
    BOOST_REQUIRE(neighbors[108] == 6);
    BOOST_REQUIRE_CLOSE(distances[108], (3.00 * 3.00), 1e-5);
    BOOST_REQUIRE(neighbors[109] == 4);
    BOOST_REQUIRE_CLOSE(distances[109], (4.05 * 4.05), 1e-5);

    // Clean the memory.
    delete allknn;
  }
}

/***
 * Test the dual-tree nearest-neighbors method with the naive method.  This
 * uses both a query and reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(dual_tree_vs_naive_1) {
  arma::mat data_for_tree_;

  // Hard-coded filename: bad!
  if (data::Load("test_data_3_1000.csv", data_for_tree_) != SUCCESS_PASS)
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Set up matrices to work with.
  arma::mat dual_query(data_for_tree_);
  arma::mat dual_references(data_for_tree_);
  arma::mat naive_query(data_for_tree_);
  arma::mat naive_references(data_for_tree_);

  AllkNN allknn_(dual_query, dual_references, 20, 5);
  AllkNN naive_(naive_query, naive_references, 1 /* leaf_size ignored */, 5,
      AllkNN::NAIVE);
 
  arma::Col<index_t> resulting_neighbors_tree;
  arma::vec distances_tree;
  allknn_.ComputeNeighbors(resulting_neighbors_tree, distances_tree);

  arma::Col<index_t> resulting_neighbors_naive;
  arma::vec distances_naive;
  naive_.ComputeNeighbors(resulting_neighbors_naive, distances_naive);

  for (index_t i = 0; i < resulting_neighbors_tree.n_elem; i++) {
    BOOST_REQUIRE(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
    BOOST_REQUIRE_CLOSE(distances_tree[i], distances_naive[i], 1e-5);
  }
}

/***
 * Test the dual-tree nearest-neighbors method with the naive method.  This uses
 * only a reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(dual_tree_vs_naive_2) {
  arma::mat data_for_tree_;

  // Hard-coded filename: bad!
  // Code duplication: also bad!
  if (data::Load("test_data_3_1000.csv", data_for_tree_) != SUCCESS_PASS)
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Set up matrices to work with (may not be necessary with no ALIAS_MATRIX?).
  arma::mat dual_query(data_for_tree_);
  arma::mat naive_query(data_for_tree_);

  AllkNN allknn_(dual_query, 20, 1);
  AllkNN naive_(naive_query, 1 /* leaf_size ignored with naive */, 1,
      AllkNN::NAIVE);

  arma::Col<index_t> resulting_neighbors_tree;
  arma::vec distances_tree;
  allknn_.ComputeNeighbors(resulting_neighbors_tree, distances_tree);

  arma::Col<index_t> resulting_neighbors_naive;
  arma::vec distances_naive;
  naive_.ComputeNeighbors(resulting_neighbors_naive, distances_naive);

  for (index_t i = 0; i < resulting_neighbors_tree.n_elem; i++) {
    BOOST_REQUIRE(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
    BOOST_REQUIRE_CLOSE(distances_tree[i], distances_naive[i], 1e-5);
  }
}

/***
 * Test the single-tree nearest-neighbors method with the naive method.  This
 * uses only a reference dataset.
 *
 * Errors are produced if the results are not identical.
 */
BOOST_AUTO_TEST_CASE(single_tree_vs_naive) {
  arma::mat data_for_tree_;

  // Hard-coded filename: bad!
  // Code duplication: also bad!
  if (data::Load("test_data_3_1000.csv", data_for_tree_) != SUCCESS_PASS)
    BOOST_FAIL("Cannot load test dataset test_data_3_1000.csv!");

  // Set up matrices to work with (may not be necessary with no ALIAS_MATRIX?).
  arma::mat single_query(data_for_tree_);
  arma::mat naive_query(data_for_tree_);

  AllkNN allknn_(single_query, 20, 5, AllkNN::MODE_SINGLE);
  AllkNN naive_(naive_query, 1 /* leaf_size ignored with naive */, 5,
      AllkNN::NAIVE);

  arma::Col<index_t> resulting_neighbors_tree;
  arma::vec distances_tree;
  allknn_.ComputeNeighbors(resulting_neighbors_tree, distances_tree);

  arma::Col<index_t> resulting_neighbors_naive;
  arma::vec distances_naive;
  naive_.ComputeNeighbors(resulting_neighbors_naive, distances_naive);

  for (index_t i = 0; i < resulting_neighbors_tree.n_elem; i++) {
    BOOST_REQUIRE(resulting_neighbors_tree[i] == resulting_neighbors_naive[i]);
    BOOST_REQUIRE_CLOSE(distances_tree[i], distances_naive[i], 1e-5);
  }
}
