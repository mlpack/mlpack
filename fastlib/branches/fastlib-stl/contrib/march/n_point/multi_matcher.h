/*
 *  multi_matcher.h
 *  
 *
 *  Created by William March on 6/6/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


namespace npt {
  
  class MultiMatcher {
    
  private:
    // for now, I'm assuming a single, global thickness for each dimension of 
    // the matcher
    double bandwidth_;
    
    // a matcher is a set of (n choose 2) of these
    // they may be linearly or logarithmically spaced
    std::vector<double> matcher_components_;
    
    
    
  public:
    
  }; // class
  
} // namespace
