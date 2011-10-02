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
      
      // For BOOST serialization.
      friend class boost::serialization::access;

      
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
        // put malloc here 
        SetZero();
      }
      
      template<typename GlobalType>
      void Init(const GlobalType& global_in, int num_points) {
        this->Init(num_points);
      }
      
      void Accumulate(const TwoPointResult &result_in) {
        
        num_tuples_ += result_in.num_tuples_;
        weighted_num_tuples_ += result_in.weighted_num_tuples_;
        num_prunes_ += result_in.num_prunes_;
        
      }
      
      template<typename MetricType, typename GlobalType>
      void PostProcess(const MetricType &metric,
                       const arma::vec &qpoint,
                       int q_index,
                       double q_weight,
                       const GlobalType &global,
                       const bool is_monochromatic) {
      

        
      } // PostProcess
      
      
      template<typename DistributedTableType>
      void PostProcess(boost::mpi::communicator &world,
                       DistributedTableType *distributed_table_in) {

	int total_num_tuples;
	boost::mpi::all_reduce( 
          world, num_tuples_, total_num_tuples, std::plus<int>() );
	num_tuples_ = total_num_tuples;
      }
      
      
      template<typename GlobalType, typename TreeType, typename DeltaType>
      void ApplyProbabilisticDelta(GlobalType &global, TreeType *qnode, 
                                   double failure_probability,
                                   const DeltaType &delta_in) {

      } // ApplyProbabalisticDelta
      
      
      void Seed(int qpoint_index, double initial_pruned_in) {
        
      }
      
      /** @brief Aliases a subset of the given result.
       */
      template<typename TreeIteratorType>
      void Alias(TreeIteratorType &it) {
        // TODO: should this be blank?
      }
      
      /** @brief Aliases a subset of the given result.
       */
      template<typename TreeIteratorType>
      void Alias(const TwoPointResult &result_in, TreeIteratorType &it) {
        // TODO: is this right?
        num_tuples_ = result_in.num_tuples_;
        weighted_num_tuples_ = result_in.weighted_num_tuples_;
        num_prunes_ = result_in.num_prunes_;
      }
      
      /** @brief Aliases another result.
       */
      void Alias(const TwoPointResult &result_in) {
        num_tuples_ = result_in.num_tuples_;
        weighted_num_tuples_ = result_in.weighted_num_tuples_;
        num_prunes_ = result_in.num_prunes_;
      }
      
      void Copy(const TwoPointResult& result_in) {
        num_tuples_ = result_in.num_tuples_;
        weighted_num_tuples_ = result_in.weighted_num_tuples_;
        num_prunes_ = result_in.num_prunes_;        
      }
      
      /** @brief Serialize the KDE result object.
       */
      template<class Archive>
      void serialize(Archive &ar, const unsigned int version) {
        ar & num_tuples_;
        ar & weighted_num_tuples_;
        ar & num_prunes_;
      }
      
      
      template<typename TwoPointPostponedType>
      void ApplyPostponed(int q_index, const TwoPointPostponedType &postponed_in) {
   
        //printf("adding postponed to result.\n");
        num_tuples_ += postponed_in.num_tuples();
        weighted_num_tuples_ += postponed_in.weighted_num_tuples();
        
      }
      
      template<typename GlobalType, typename TwoPointPostponedType>
      void FinalApplyPostponed(const GlobalType &global,
                               const arma::vec &qpoint,
                               int q_index,
                               const TwoPointPostponedType &postponed_in) {
        
        
        //ApplyPostponed(q_index, postponed_in);
        
      } // FinalApplyPostponed
      
    }; // class
    
  } // namespace
} // namespace

#endif
