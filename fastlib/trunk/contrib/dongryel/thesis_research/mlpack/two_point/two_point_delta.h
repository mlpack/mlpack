/*
 *  two_point_delta.h
 *
 *
 *  Stores the (possible) result of a node-node computation at this stage
 *  This result is examined elsewhere to see if pruning is appropriate.
 *
 *  Created by William March on 9/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef TWO_POINT_DELTA_H
#define TWO_POINT_DELTA_H

// I think this can just be empty

namespace mlpack {
namespace two_point {

class TwoPointDelta {

  public:

    TwoPointDelta() {

    } // constructor

    void SetZero() {

    }

    template<typename MetricType, typename GlobalType, typename TreeType>
    void DeterministicCompute(
      const MetricType &metric,
      const GlobalType &global, TreeType *qnode, TreeType *rnode,
      bool qnode_and_rnode_are_equal,
      const core::math::Range &squared_distance_range) {



    } // DeterministicCompute()


    // members are public



}; // class

} // namespace
}// namespace




#endif
