/*
 *  multi_matcher.h
 *  
 *
 *  Created by William March on 3/30/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef MULTI_MATCHER_H
#define MULTI_MATCHER_H

#include "n_point_results.h"
#include "fastlib/fastlib.h"

class MultiMatcher {
  
private:
  
    
  ////////// variables /////////////
  
  ArrayList<double> ranges_;
  
  Permutations perms_;
  
public:

  
  
  void TestNodes_(const DHrectBound<2>& box1, const DHrectBound<2>& box2,
                  ResultsTensor& results);
  
}; // class MultiMatcher


#endif 

