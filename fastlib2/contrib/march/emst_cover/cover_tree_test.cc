/*
 *  cover_tree_test.cc
 *  
 *
 *  Created by William March on 9/10/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */


#include "ctree.h"

class TestStat {
  
public:
  double test_dist;
  
  void Init() {
    test_dist = 0.0;
  }
  
};

int main(int argc, char* argv[]) {
  
  fx_module* root = fx_init(argc, argv, NULL);
  
  fx_module* mod = fx_submodule(root, "test");
  
  Matrix queries;
  queries.Init(2,4);
  queries.set(0, 0, 0.0);
  queries.set(1, 0, 0.0);
  queries.set(0, 1, 1.0);
  queries.set(1, 1, 0.0);
  queries.set(0, 2, 2.0);
  queries.set(1, 2, 0.0);
  queries.set(0, 3, 3.0);
  queries.set(1, 3, 0.0);
  
  
  // what does this mean?
  double base = 1.3;

  typedef CoverTreeNode<TestStat, double> TestTree;
  
  TestTree* tree = ctree::MakeCoverTree<TestTree, double>(queries, base, mod);
  
  ctree::PrintTree<TestTree>(tree);
  
  fx_done(root);
  
  return 0;
  
} // main()
