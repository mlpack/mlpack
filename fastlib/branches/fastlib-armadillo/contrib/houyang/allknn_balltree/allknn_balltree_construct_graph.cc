/**
 * @allknn_balltree_construct_graph.cc
 * Construct and store the All-k-Nearest-Neighbor Graph of a data set.
 * A pre-step for learnig a ball tree.
 */

#include "allknn_balltree.h"
#include "fastlib/base/test.h"
#include <stdio.h>
#include <stdlib.h>

// Parameters
const bool TREE_LEARNING= false; // true: learning tree, false: no learning, just normal tree
const int TREE_DUAL_SINGLE= 2; // 2: dual tree, 1: single tree
const index_t K_NN= 1; // want to find the K_NN th nearest neighbor, excluding the point itself


// compare the order of two 3-by-1 columns according to the first two rows
int ColumnCompare(const void *col_a, const void *col_b) {
  const double *col_a_ptr = (double*)col_a;
  const double *col_b_ptr = (double*)col_b;

  if (*col_a_ptr< *col_b_ptr)
    return -1;
  else if (*col_a_ptr> *col_b_ptr)
    return 1;
  else // (*col_a_ptr== *col_b_ptr)
    if (*(col_a_ptr+1) < *(col_b_ptr+1))
      return -1;
    else if (*(col_a_ptr+1) > *(col_b_ptr+1))
      return 1;
    else // (*(col_a_ptr+1) == *(col_b_ptr+1))
      return 0;
}

int main(int argc, char *argv[]) {
  fx_init(argc, argv, NULL);
  // ConstructTreeGraph
  ArrayList<index_t> result_neighbors;
  ArrayList<double> result_distances;

  index_t n_data, i, j, n_pair;
  Matrix degree; // Diagonal of the degree matrix D, dimension; n_data-by-1
  Matrix affinity; // Sparse (also symmetric) affinity matrix A
  // 1st row: adjacency[0 i]: for sample i, how many other samples are adjacent to it;
  // 2nd row: adjacency[1 i]: start positions of sample i in affinity matrix
  Matrix adjacency;
  
  AllkNNBallTree tree_allknn_;
  tree_allknn_.Init(TREE_DUAL_SINGLE, TREE_LEARNING); // use dual tree, no learning
  Matrix data_for_tree_;

  //data::Load("data_3_1000.csv", &data_for_tree_);
  //data::Load("pen_digi.csv", &data_for_tree_);
  //data::Load("pen_digi_sorted_merged.csv", &data_for_tree_);
  //data::Load("UCI_letter.csv", &data_for_tree_);
  //data::Load("UCI_letter_sorted_merged.csv", &data_for_tree_);
  //data::Load("UCI_magic_sorted_merged.csv", &data_for_tree_);
  
  // Minimum size of the leaf node, if small than this, do not split the node
  index_t LEAF_SIZE= fx_param_int_req(NULL,"leaf_size");
  data::Load(fx_param_str_req(NULL, "ref_data"), &data_for_tree_);
  

fx_timer_start(NULL, "construct_all_kNN_graph");


  // set Query and Reference data to 'data_for_tree_' and 'data_for_tree_'
  tree_allknn_.TreeInit(data_for_tree_, data_for_tree_,  LEAF_SIZE, K_NN+1); // not K_NN! since the first-NN is the data sample itself
  // all-kNN using Dual BallTrees
  tree_allknn_.TreeAllkNN(&result_neighbors, &result_distances, false);

fx_timer_stop(NULL, "construct_all_kNN_graph");
  
  n_data = data_for_tree_.n_cols();
  
  degree.Init(n_data, 1);
  //degree.SetAll(K_NN);
  degree.SetZero();

  n_pair = 2*K_NN*n_data;

  affinity.Init(3, n_pair);
  affinity.SetZero();
  adjacency.Init(2, n_data);
  adjacency.SetZero();
  /*
  Matrix matrix_knn;
  matrix_knn.Init(n_data, K_NN);
  
  // Consturct all-kNN graph
  index_t idx;
  for (i=0; i<n_data; i++) {
    for (j=0; j<K_NN; j++) {
      // NN relationship
      idx = result_neighbors[i*(K_NN+1)+ j+ 1]; // skip the first NN which is the data sample itself
      matrix_knn.set(i, j, idx);
      // Diagonal elements of the the degree matrix D 
      degree.set(idx, 0, degree.get(idx,0)+1);
    }
  }
  // Store digonal of degrees matrix D
  data::Save("data_3_1000_graph_degree.csv", degree);
  
  // Store NN relations(first KNN-1 columns)
  data::Save("data_3_1000_graph_knn.csv", matrix_knn);
  
  // Store distances
  for (i=0; i<n_data; i++)
    for (j=0; j<K_NN; j++)
      matrix_knn.set(i, j, result_distances[i*(K_NN+1)+ j+ 1]); // skip the first NN which is the data sample itself
  data::Save("data_3_1000_graph_dist.csv", matrix_knn);
  */
  
  // List the 2kN affinity relations; set each column as [col#; row#; aff_value], O(kN)
  index_t tmp_idx;
  for (i=0; i<n_data; i++) {
    for (j=0; j<K_NN; j++) {
      tmp_idx = result_neighbors[i*(K_NN+1)+ j + 1]; // skip the first NN which is the data sample itself
      affinity.set(0, 2*i*K_NN+2*j, i);
      affinity.set(1, 2*i*K_NN+2*j, tmp_idx);
      affinity.set(2, 2*i*K_NN+2*j, 1.0); // 0/1 weigths
      affinity.set(0, 2*i*K_NN+2*j+1, tmp_idx);
      affinity.set(1, 2*i*K_NN+2*j+1, i);
      affinity.set(2, 2*i*K_NN+2*j+1, 1.0); // 0/1 weights
    }
  }
  // data::Save("data_3_1000_graph_affunsort.csv", affinity);

  // Fill in degree info
  for (i=0; i<n_pair; i++){
    index_t s= (index_t)affinity.get(0,i);
    degree.set(s, 0, degree.get(s,0)+1.0);
  }
  // Store digonal of degrees matrix D
  data::Save("data_3_1000_graph_degree.csv", degree);

  // Sort affinity relations according to the first two rows [col#;row#], O(kN log(kN))
  qsort(affinity.ptr(), n_pair, 3*sizeof(double), ColumnCompare);
  
  // Combine affinity relations and fill the sparse A, O(kN); Get adjacency info
  index_t idx_ct = 0;
  index_t idx_adj = 0;
  index_t n_adj_ct = 1;
  for (i=0; i<n_pair-1; i++) {
    if (affinity.get(0,i) == affinity.get(0,i+1)){
      if (affinity.get(1,i) == affinity.get(1,i+1)) {
	affinity.set(2,idx_ct, affinity.get(2,idx_ct)+1);
	// handle the last line
	if (i==n_pair-2)
	  affinity.set(2,idx_ct, affinity.get(2,idx_ct)+1);
      }
      else{
	idx_ct++; // 1419
	n_adj_ct++;
	affinity.set(0, idx_ct, affinity.get(0,i+1));
	affinity.set(1, idx_ct, affinity.get(1,i+1));
	affinity.set(2, idx_ct, affinity.get(2,i+1));
	// handle the last line
	if (i==n_pair-2) {
	  idx_ct++;
	  affinity.set(0, idx_ct, affinity.get(0,i+1));
	  affinity.set(1, idx_ct, affinity.get(1,i+1));
	  affinity.set(2, idx_ct, affinity.get(2,i+1));
	  // 1st row: adjacency[0 i]: for sample i, how many other samples are adjacent to it;
	  adjacency.set(0, idx_adj, n_adj_ct);
	}
      }
    }
    else {
      idx_ct++; // 1419

      affinity.set(0, idx_ct, affinity.get(0,i+1));
      affinity.set(1, idx_ct, affinity.get(1,i+1));
      affinity.set(2, idx_ct, affinity.get(2,i+1));
      // 1st row: adjacency[0 i]: for sample i, how many other samples are adjacent to it;
      adjacency.set(0, idx_adj, n_adj_ct);

      idx_adj++; // 999
      n_adj_ct = 1;

      // handle the last line
      if (i==n_pair-2) {
	idx_ct++;
	affinity.set(0, idx_ct, affinity.get(0,i+1));
	affinity.set(1, idx_ct, affinity.get(1,i+1));
	affinity.set(2, idx_ct, affinity.get(2,i+1));
	// 1st row: adjacency[0 i]: for sample i, how many other samples are adjacent to it;
	adjacency.set(0, idx_adj, n_adj_ct);
      }
    }
  }
  // 2nd row: adjacency[1 i]: start positions of sample i in affinity matrix
  for (i=1; i<= idx_adj; i++)
    adjacency.set(1, i, adjacency.get(1, i-1)+ adjacency.get(0, i-1));
  
  // Store the adjacency information
  data::Save("data_3_1000_graph_adjacency.csv", adjacency);

  // Check and store the sparse affinity matrix A (only save the first idx_ct nubmer of columns of affinity)
  Matrix affinity_part;
  affinity_part.Alias(affinity.ptr(), 3, idx_ct);

  /*  // check whether A is symmetric
  index_t opt_pos;
  for (i=0; i<idx_ct; i++){
    index_t left= (index_t)affinity_part.get(0,i);
    index_t right= (index_t)affinity_part.get(1,i);
    double value= affinity_part.get(2,i);
    index_t left_chk;
    double value_chk;

    for (index_t k=0; k<(index_t)adjacency.get(0, right); k++) {
      opt_pos = (index_t)adjacency.get(1, right) + k;
      if((index_t)affinity.get(1,opt_pos) == left){
	left_chk = (index_t)affinity.get(1,opt_pos);
	value_chk = affinity.get(2,opt_pos);
	break;
      }
    }
    
    DEBUG_ASSERT(left==left_chk);
    DEBUG_ASSERT(value==value_chk);
  }
  */
  data::Save("data_3_1000_graph_affinity.csv", affinity_part);

    
  NOTIFY("kNN Graph constructed and stored. TREE_DUAL_SINGLE= %d; K_NN=%d; LEAF_SIZE=%d\n", TREE_DUAL_SINGLE, K_NN, LEAF_SIZE);
  fx_done(NULL);
}
