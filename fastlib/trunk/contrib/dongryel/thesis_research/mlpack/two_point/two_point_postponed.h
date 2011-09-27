/*
 *  two_point_postponed.h
 *
 *
 *  Created by William March on 9/14/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef TWO_POINT_POSTPONED_H
#define TWO_POINT_POSTPONED_H

namespace mlpack {
namespace two_point {

// TODO: does this need to be templated by an ExpansionType?
class TwoPointPostponed {

  private:

    int num_tuples_;

    double weighted_num_tuples_;

    bool is_monochromatic_;

  public:

    int num_tuples() const {
      return num_tuples_;
    }

    double weighted_num_tuples() const {
      return weighted_num_tuples_;
    }

    TwoPointPostponed() {

      num_tuples_ = 0;
      weighted_num_tuples_ = 0.0;

    }

    void SetZero() {

      num_tuples_ = 0;
      weighted_num_tuples_ = 0.0;

    } // SetZero

    void Copy(const TwoPointPostponed& post_in) {

      num_tuples_ = post_in.num_tuples_;

    } // Copy

    template<typename GlobalType, typename MetricType>
    void ApplyContribution() {

    } // ApplyContribution

    void ApplyPostponed(const TwoPointPostponed &other_postponed) {

    } // ApplyPostponed

    void Init() {

    } // Init()

    template<typename GlobalType>
    void Init(const GlobalType& global_in) {

    }

    template<typename GlobalType, typename TreeType>
    void Init(
      const GlobalType &global_in, TreeType *qnode, TreeType *rnode,
      bool qnode_and_rnode_are_equal) {

      SetZero();
      is_monochromatic_ = (qnode_and_rnode_are_equal);
      
    }

    template < typename TreeType, typename GlobalType,
             typename TwoPointDelta, typename ResultType >
    void ApplyDelta(
      TreeType *qnode, TreeType *rnode,
      const GlobalType &global, const TwoPointDelta &delta_in,
      ResultType *query_results) {

    } // ApplyDelta


    template<typename GlobalType>
    void FinalApplyPostponed(const GlobalType &global,
                             TwoPointPostponed &other_postponed) {

    }

    template<typename GlobalType, typename MetricType>
    void ApplyContribution(const GlobalType &global,
                           const MetricType &metric,
                           const arma::vec &query_point,
                           int query_point_rank,
                           int query_point_dfs_index,
                           double query_weight,
                           const arma::vec &reference_point,
                           int reference_point_rank,
                           int reference_point_dfs_index,
                           double reference_weight) {

      /*
      if (query_and_reference_points_are_equal) {
      printf("query_point[0]: %g\n", query_point(0));
      printf("ref_point[0]: %g\n", reference_point(0));
      printf("are_equal: %d\n", query_and_reference_points_are_equal);
      printf("\n");
      
      
      }
     */
      
      // make sure we don't count a point with itself
      if(!(query_point_rank == reference_point_rank 
           && reference_point_dfs_index <= query_point_dfs_index)) {

        
        double dist_sq = metric.DistanceSq(query_point, reference_point);

        printf("ranks: (q %d, r %d), tuple: (%d, %d)\n", query_point_rank,
               reference_point_rank,
               query_point_dfs_index,
               reference_point_dfs_index);

        
        if(dist_sq <= global.upper_bound_sqr() &&
            dist_sq >= global.lower_bound_sqr()) {

          
                    
          
          num_tuples_++;
          weighted_num_tuples_ += (query_weight * reference_weight);

        }

      }

    } // ApplyContribution

    void FinalSetZero() {

    }

}; // class

} // namespace
} // namespace

#endif
