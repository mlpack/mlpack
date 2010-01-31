/**
 * @allknn_balltree_test.cc
 *
 * Test file for Ball Tree based All-kNN
 *
 * Original Ball Trees and Min-Cut Tree based learning ball trees are implemented.
 * For Min-Cut-Tree (MCT) Ball Trees, need to construct a kNN graph first. See allknn_balltree_construct_graph.cc
 *
 */

#include "allknn_balltree.h"
#include "fastlib/base/test.h"


int main(int argc, char *argv[]) {
  fx_init(argc, argv, NULL);

  // Minimum size of the leaf node, if smaller than this, do not split the node
  // default: 20
  index_t LEAF_SIZE= fx_param_int(NULL,"leaf_size", 20); 

  // the k of kNN; default: 1-nearest neighbor
  index_t K_NN= fx_param_int(NULL, "k_nn", 1);

  // Whether use dual tree or single tree for ball trees. 2: dual tree, 1: single tree
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

  // Tree type (ball tree/MCT ball tree), required parameter
  String tree_name = fx_param_str_req(NULL,"tree");

  // Whether need to check the correctness of MCT ball tree and non-learning ball tree; default: do not check
  bool check_bt_correctness = fx_param_bool(NULL, "check_correct", 0);

  ArrayList<index_t> resulting_neighbors_tree_learn;
  ArrayList<double> distances_tree_learn;
  ArrayList<index_t> resulting_neighbors_tree_ball;
  ArrayList<double> distances_tree_ball;
  ArrayList<index_t> resulting_neighbors_bf;
  ArrayList<double> distances_bf;
  

  if (tree_name == "MCT_bt" || check_bt_correctness==1) { // MCT Learning Ball Tree
    AllkNNBallTree *tree_allknn_learn_;
    tree_allknn_learn_ = new AllkNNBallTree();
    tree_allknn_learn_->Init(TREE_DUAL_SINGLE, true); // true: learning
    printf("Begin MCT Ball Tree Building...\n");
    fx_timer_start(NULL, "MCT_Ball_Tree_Build");
    tree_allknn_learn_->TreeInit(*query_data, *ref_data, LEAF_SIZE, K_NN);
    fx_timer_stop(NULL, "MCT_Ball_Tree_Build");
    printf("Begin MCT Ball Tree Query...\n");
    fx_timer_start(NULL, "MCT_Ball_Tree_Query");
    tree_allknn_learn_->TreeAllkNN(&resulting_neighbors_tree_learn, &distances_tree_learn, true);
    fx_timer_stop(NULL, "MCT_Ball_Tree_Query");
    printf("Number of distance computations for MCT Ball Tree:%d\n", tree_allknn_learn_->ct_dist_comp);
    delete tree_allknn_learn_; 
  }

  if (tree_name == "bt" || check_bt_correctness==1) { // Non-learning Ball tree
    AllkNNBallTree *tree_allknn_ball_;
    tree_allknn_ball_ = new AllkNNBallTree();
    tree_allknn_ball_->Init(TREE_DUAL_SINGLE, false); // false: non-learning
    printf("Begin Ball Tree Building...\n");
    fx_timer_start(NULL, "Ball_Tree_NO_LEARN_Build");
    tree_allknn_ball_->TreeInit(*query_data, *ref_data,  LEAF_SIZE, K_NN);
    fx_timer_stop(NULL, "Ball_Tree_NO_LEARN_Build");
    printf("Begin Ball Tree Query...\n");
    fx_timer_start(NULL, "Ball_Tree_NO_LEARN_Query");
    tree_allknn_ball_->TreeAllkNN(&resulting_neighbors_tree_ball, &distances_tree_ball, false);
    fx_timer_stop(NULL, "Ball_Tree_NO_LEARN_Query");
    printf("Number of distance computations for Ball Tree:%d\n", tree_allknn_ball_->ct_dist_comp);
    delete tree_allknn_ball_; 
  }
  
  if (tree_name == "bf" || check_bt_correctness==1) { // brute force search
    AllkNNBallTree *bf_allknn_;
    bf_allknn_ = new AllkNNBallTree();
    bf_allknn_->Init(2, false); // for brute force searech, we still use a dual tree
    printf("Begin Brute Force Building...\n");
    fx_timer_start(NULL, "BF_build");
    bf_allknn_->BruteForceInit(*query_data, *ref_data, K_NN);
    fx_timer_stop(NULL, "BF_build");
    printf("Begin Brute Force Query...\n");
    fx_timer_start(NULL, "BF_query");
    bf_allknn_->BruteForceAllkNN(&resulting_neighbors_bf, &distances_bf);
    fx_timer_stop(NULL, "BF_query");
    printf("Number of distance computations for BruteForce:%d\n", bf_allknn_->ct_dist_comp);
    delete bf_allknn_;
  }

  if (check_bt_correctness && tree_name == "bt") {
    // Check correctness of Non-learning Ball Tree results  
    for(index_t i=0; i<resulting_neighbors_tree_ball.size(); i++) {
      TEST_ASSERT(resulting_neighbors_tree_ball[i] == resulting_neighbors_bf[i]);
      TEST_DOUBLE_APPROX(distances_tree_ball[i], distances_bf[i], 1e-5);
    }
    NOTIFY("BF v.s. Non-Learning BallTree test passed. TREE_DUAL_SINGLE= %d; K_NN=%d; LEAF_SIZE=%d\n", TREE_DUAL_SINGLE, K_NN, LEAF_SIZE);
  }
  else if (check_bt_correctness && tree_name == "MCT_bt") {
    // Check correctness of MCT Ball Tree results  
    for(index_t i=0; i<resulting_neighbors_tree_learn.size(); i++) {
      TEST_ASSERT(resulting_neighbors_tree_learn[i] == resulting_neighbors_bf[i]);
      TEST_DOUBLE_APPROX(distances_tree_learn[i], distances_bf[i], 1e-5);
    }
    NOTIFY("BF v.s. MCT Ball Tree test passed. TREE_DUAL_SINGLE= %d; K_NN=%d; LEAF_SIZE=%d\n", TREE_DUAL_SINGLE, K_NN, LEAF_SIZE);
  }
  
  delete ref_data;
  delete query_data;

  fx_done(NULL);
  return 0;
}
