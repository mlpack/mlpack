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
    
    ArrayList<NPointAlg::NPointNode *> test_nodes;
    test_nodes.Init(n);
    
    alg.tree_->Print();
    
    test_nodes[0] = alg.tree_;
    test_nodes[1] = alg.tree_->left();
    test_nodes[2] = alg.tree_->left()->right();
    
    int count = alg.CountTuples_(test_nodes);
    printf("Count: %d\n", count);
    
  }
  
  
}; // NPointTester
