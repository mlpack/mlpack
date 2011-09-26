/*
 *  two_point_summary.h
 *
 *
 *  Created by William March on 9/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef MLPACK_TWO_POINT_TWO_POINT_SUMMARY_H
#define MLPACK_TWO_POINT_TWO_POINT_SUMMARY_H


namespace mlpack {
namespace two_point {

class TwoPointSummary {

  private:

    // TODO: do I need to worry about serialization?



  public:

    template < typename GlobalType, typename DeltaType, typename TreeType,
             typename ResultType >
    bool CanSummarize(const GlobalType &global, DeltaType &delta,
                      const core::math::Range &squared_distance_range,
                      TreeType *qnode, int qnode_rank,
                      TreeType *rnode, int rnode_rank,
                      bool qnode_and_rnode_are_equal,
                      ResultType *query_results) {


      
      if ((qnode_rank == rnode_rank) 
          && (rnode->end() <= qnode->begin())) {
        //printf("symmetry prune \n");
        return true;
      }
      
      // I think this is taken care of in the gnp code

      return((squared_distance_range.lo > global.upper_bound_sqr())
             || (squared_distance_range.hi < global.lower_bound_sqr()));
      //return false;
      
    } // CanSummarize

    TwoPointSummary() {

    } // constructor

    TwoPointSummary(TwoPointSummary& other) {

    } // copy constructor

    template<typename TwoPointPostponedType>
    void ApplyPostponed(const TwoPointPostponedType &postponed_in) {

    } // ApplyPostponed

    void ApplyDelta(const TwoPointDelta &delta_in) {

    } // ApplyDelta

    template < typename GlobalType, typename DeltaType, typename TreeType,
             typename ResultType >
    bool CanSummarize(
      const GlobalType &global, DeltaType &delta,
      const core::math::Range &squared_distance_range,
      TreeType *qnode, TreeType *rnode, ResultType *query_results) const {

      // we're assuming the "query" node is the first one in the tuple
      /*
      if (global.is_monochromatic() && (qnode->end() <= rnode->begin())) {
        return true;
      }
       */

      return((squared_distance_range.lo > global.matcher_upper_bound_sq())
             || (squared_distance_range.hi < global.matcher_lower_bound_sq()));

      //return false;

    } // CanSummarize


    void StartReaccumulate() {


    } // StartReaccumulate

    template<typename GlobalType, typename ResultType>
    void Accumulate(
      const GlobalType &global, const ResultType &results, int q_index) {


    } // Accumulate

    template<typename GlobalType, typename TwoPointPostponedType>
    void Accumulate(
      const GlobalType &global, const TwoPointSummary &summary_in,
      const TwoPointPostponedType &postponed_in) {

    } // Accumulate

    void Copy(const TwoPointSummary& summary_in) {

    }

    template < typename MetricType, typename GlobalType,
             typename PostponedType, typename DeltaType,
             typename TreeType, typename ResultType >
    bool CanProbabilisticSummarize(
      const MetricType &metric,
      GlobalType &global,
      const PostponedType &postponed, DeltaType &delta,
      const core::math::Range &squared_distance_range,
      TreeType *qnode, int qnode_rank,
      TreeType *rnode, int rnode_rank,
      bool qnode_and_rnode_are_equal,
      double failure_probability, ResultType *query_results) const {

      return false;

    }

}; // class

} // namespace
} // namespace

#endif
