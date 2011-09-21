/*
 *  two_point_statistic.h
 *  
 *
 *  Created by William March on 9/14/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef TWO_POINT_STATISTIC_H
#define TWO_POINT_STATISTIC_H

// I think this just needs to be empty

namespace mlpack {
  namespace two_point {
    
    class TwoPointStatistic {
      
      // For BOOST serialization.
      friend class boost::serialization::access;

      
    private:
      
      
      
    public:
      
      TwoPointPostponed postponed_;
      
      TwoPointSummary summary_;
      
      
      TwoPointStatistic() {
        
      }
      
      void SetZero() {
      
      }
      
      void Copy(const TwoPointStatistic& other) {
        postponed_.Copy(other.postponed_);
        summary_.Copy(other.summary_);
      }
      
      
      template<class Archive>
      void serialize(Archive &ar, const unsigned int version) {
      }
      
      template<typename GlobalType, typename TreeType>
      void Init(const GlobalType& global, TreeType *node) {}
      
      template<typename GlobalType, typename TreeType>
      void Init(const GlobalType& global, TreeType *node,
                TwoPointStatistic& left_stat,
                TwoPointStatistic& right_stat) {
        
      }
      
      void Seed(double pruned_in) {}
      
      
    }; // class
    
  } // namespace
}// namespace

#endif