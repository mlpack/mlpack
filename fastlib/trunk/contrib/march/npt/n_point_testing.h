/*
 *  n_point_testing.h
 *  
 *
 *  Created by William March on 4/5/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "fastlib/fastlib.h"
#include "n_point.h"
#include "n_point_nodes.h"

class NPointTester {
    
public:
  
  void TestCounting() {
   
    Matrix data;
    const char* data_file = "test_counting_points.csv";
    data::Load(data_file, &data);
    
    Vector weights;
    weights.Init(data.n_rows());
    weights.SetAll(1.0);
    
    int n = 3;
  
    Matrix upper_bounds;
    upper_bounds.Init(3, 3);
    
    Matrix lower_bounds;
    lower_bounds.Init(3, 3);
    
    fx_module* test_mod = fx_submodule(NULL, "test");
    
    NPointAlg alg;
    alg.Init(data, weights, lower_bounds, upper_bounds, n, test_mod);
    
    ArrayList<NPointNode*> test_nodes;
    test_nodes.Init(n);
    
    alg.tree_->Print();
    
    test_nodes[0] = alg.tree_;
    test_nodes[1] = alg.tree_->left();
    test_nodes[2] = alg.tree_->left()->right();
    
    int count = alg.CountTuples_(test_nodes);
    printf("Count: %d\n", count);
    
  }
  
  void TestInvalidIndices() {
   
    Matrix data;
    const char* data_file = "test_invalid_indices.csv";
    data::Load(data_file, &data);
    
    ArrayList<index_t> old_from_new;
    NPointNode* tree = tree::MakeKdTreeMidpoint<NPointNode>(data, 1, 
                                                            &old_from_new, NULL);
    
    ArrayList<NPointNode*> node_list;
    node_list.Init(5);
    node_list[0] = tree->left();
    node_list[1] = tree->left()->right();
    node_list[2] = tree->right()->left();
    node_list[3] = tree->right()->right();
    node_list[4] = tree->right()->right()->right();
    
    NodeTuple tuple;
    tuple.Init(node_list);
    
    ArrayList<index_t> bad_inds;
    bad_inds.Init();
    
    tuple.ind_to_split_ = 4;
    tuple.FindInvalidIndices_(&bad_inds);
    
    ot::Print(bad_inds);
    
  }
  
  
}; // NPointTester
