/**
 * @allknn_kdtree_test.cc
 *
 * kd-tree and brute force search for all-nearest neighbor search
 *
 */

#include "allknn_kdtree.h"
#include "fastlib/base/test.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv, NULL);

  // Minimum size of the leaf node, if smaller than this, do not split the node
  // default: 20
  index_t LEAF_SIZE= fx_param_int(NULL,"leaf_size", 20); 
  
  // the k of kNN; default: 1-nearest neighbor
  index_t K_NN= fx_param_int(NULL, "k_nn", 1);

  // Whether use dual tree or single tree for kd tree. 2: dual tree, 1: single tree
  // default: use dual tree
  String str_dual_single = fx_param_str(NULL,"dual_single", "dual");
  int TREE_DUAL_SINGLE;
  if (str_dual_single == "dual") {
    TREE_DUAL_SINGLE = 2; // use dual tree
    printf("Using dual trees...\n");
  }
  else {
    TREE_DUAL_SINGLE = 1; // use single tree
    printf("Using single trees...\n");
  }

  // Load reference data (required parameter)
  Matrix *ref_data;
  ref_data = new Matrix();
  String ref_data_filename = fx_param_str_req(NULL, "ref_data");
  data::Load(ref_data_filename, ref_data);
  // Load query data (optional parameter)
  Matrix *query_data;
  query_data = new Matrix();
  String query_data_filename = fx_param_str(NULL, "query_data", ref_data_filename);
  data::Load(query_data_filename, query_data);
  
  // Tree type (kd/bf), required parameter
  String tree_name = fx_param_str_req(NULL,"tree");

  // Whether need to check the correctness of kd tree; default: do not check
  bool check_kd_correctness = fx_param_bool(NULL, "check_correct", 0);


  ArrayList<index_t> resulting_neighbors_kd;
  ArrayList<double> distances_kd;
  ArrayList<index_t> resulting_neighbors_bf;
  ArrayList<double> distances_bf;


  if (tree_name == "kd" || check_kd_correctness==1) { // kd tree search
    AllkNNkdTree *kd_allknn_;
    kd_allknn_ = new AllkNNkdTree();
    kd_allknn_->Init(TREE_DUAL_SINGLE); // for brute force searech, we still use a dual tree

    fx_timer_start(NULL, "kdTree_Build");
    kd_allknn_->kdTreeInit(*query_data, *ref_data, LEAF_SIZE, K_NN);
    fx_timer_stop(NULL, "kdTree_Build");

    fx_timer_start(NULL, "kdTree_Query");
    kd_allknn_->kdTreeAllkNN(&resulting_neighbors_kd, &distances_kd);
    fx_timer_stop(NULL, "kdTree_Query");

    printf("Number of distance computations for kd Tree:%d\n", kd_allknn_->ct_dist_comp);
    delete kd_allknn_;
  }
  
  if (tree_name == "bf" || check_kd_correctness==1) { // brute force search
    AllkNNkdTree *bf_allknn_;
    bf_allknn_ = new AllkNNkdTree();
    bf_allknn_->Init(2); // for brute force searech, we still use a dual tree

    fx_timer_start(NULL, "BF_Build");
    bf_allknn_->BruteForceInit(*query_data, *ref_data, K_NN);
    fx_timer_stop(NULL, "BF_Build");

    fx_timer_start(NULL, "BF_Query");
    bf_allknn_->BruteForceAllkNN(&resulting_neighbors_bf, &distances_bf);
    fx_timer_stop(NULL, "BF_Query");

    printf("Number of distance computations for BruteForce:%d\n", bf_allknn_->ct_dist_comp);
    delete bf_allknn_;
  }
  
  // Check correctness of kd tree's results
  if (check_kd_correctness) {
    for(index_t i=0; i<resulting_neighbors_kd.size(); i++) {
      TEST_ASSERT(resulting_neighbors_kd[i] == resulting_neighbors_bf[i]);
      TEST_DOUBLE_APPROX(distances_kd[i], distances_bf[i], 1e-5);
    }
    NOTIFY("BF v.s. kd Tree test passed. TREE_DUAL_SINGLE= %d; K_NN=%d; LEAF_SIZE=%d\n", TREE_DUAL_SINGLE, K_NN, LEAF_SIZE);
  }

  
  delete ref_data;
  delete query_data;
 
  fx_done(NULL);
  return 0;
}
