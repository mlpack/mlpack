/*
 *  2pt_result.h
 *  
 *
 *  Created by William March on 9/12/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef MLPACK_TWO_POINT_RESULT_H
#define MLPACK_TWO_POINT_RESULT_H

#include "core/parallel/map_vector.h"


namespace mlpack {
  namespace two_point {
    
    class TwoPointResult {
      
    //private:
    public:
      
      int num_tuples_;
      
      double weighted_num_tuples_;
      
      int num_prunes_;
      
    //public:
      
      // has public members
      
      /*
      int num_tuples() const {
        return num_tuples_;
      }
      
      double weighted_num_tuples() const {
        return weighted_num_tuples_;
      }
      
      int num_prunes() const {
        return num_prunes_;
      }
      */
       
      void SetZero() {
        num_tuples_ = 0;
        weighted_num_tuples_ = 0.0;
        num_prunes_ = 0;
      }
      
      TwoPointResult() {
        SetZero();
      }
      
      void Init(int num_points) {
        SetZero();
      }
      
      template<typename GlobalType>
      void Init(const GlobalType& global_in, int num_points) {
        this->Init(num_points);
      }
      
      template<typename MetricType, typename GlobalType>
      void PostProcess(const MetricType &metric,
                       const arma::vec &qpoint,
                       int q_index,
                       double q_weight,
                       const GlobalType &global,
                       const bool is_monochromatic) {
        
      
      } // PostProcess
      
      
      template<typename GlobalType, typename TreeType, typename DeltaType>
      void ApplyProbabilisticDelta(GlobalType &global, TreeType *qnode, 
                                   double failure_probability,
                                   const DeltaType &delta_in) {

      } // ApplyProbabalisticDelta
      
      
      void Seed(int qpoint_index, double initial_pruned_in) {
        
      }
      
      template<typename KdePostponedType>
      void ApplyPostponed(int q_index, const KdePostponedType &postponed_in) {
   
        num_tuples_ += postponed_in.num_tuples();
        weighted_num_tuples_ += postponed_in.weighted_num_tuples();
        
      }
      
      template<typename GlobalType, typename KdePostponedType>
      void FinalApplyPostponed(const GlobalType &global,
                               const arma::vec &qpoint,
                               int q_index,
                               const KdePostponedType &postponed_in) {
        
        //ApplyPostponed(q_index, postponed_in);
        
      } // FinalApplyPostponed
      
    }; // class
    
  } // namespace
} // namespace

#endif