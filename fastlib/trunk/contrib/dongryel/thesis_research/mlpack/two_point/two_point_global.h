/*
 *  two_point_global.h
 *  
 *
 *  Created by William March on 9/14/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef TWO_POINT_GLOBAL_H
#define TWO_POINT_GLOBAL_H

// I think this is done for now, unless ConsiderExtinsicPruneTrait is 
// used in the gnp code somewhere

namespace mlpack {
  namespace two_point {
    
    template<typename IncomingTableType>
    class TwoPointGlobal {
      
    public:
      
      typedef IncomingTableType TableType;
      
      bool ConsiderExtrinsicPrune(
        const core::math::Range &squared_distance_range) const {
        
        return ((squared_distance_range.lo > matcher_upper_bound_sqr_)
                || (squared_distance_range.hi < matcher_lower_bound_sqr_));
        
      } // ConsiderExtrinsicPrune
      
      TwoPointGlobal() {
        
      } // constructor
      
      ~TwoPointGlobal() {
        
      } // destructor
      
      void Init(TableType* table_1, TableType* table_2,
                double matcher_dist, double matcher_thick,
                bool is_mono) {
        
        points_table_1_ = table_1;
        points_table_2_ = table_2;
        
        matcher_distance_ = matcher_dist;
        matcher_thickness_ = matcher_thick;
        is_monochromatic_ = is_mono;
        
        matcher_lower_bound_sqr_ = (matcher_distance_ - 0.5 * matcher_thickness_) 
                                * (matcher_distance_ - 0.5 * matcher_thickness_);
        matcher_upper_bound_sqr_ = (matcher_distance_ + 0.5 * matcher_thickness_) 
                                * (matcher_distance_ + 0.5 * matcher_thickness_);
        
      } // Init
      
      double matcher_distance() const {
        return matcher_distance_;
      }
      
      double matcher_thickness() const {
        return matcher_thickness_;
      }
      
      bool is_monochromatic() const {
        return is_monochromatic_;
      }
      
      TableType* points_table_1() {
        return points_table_1_;
      }
      
      TableType* points_table_2() {
        return points_table_2_;
      }
      
      double lower_bound_sqr() {
        return matcher_lower_bound_sqr_;
      } 

      double upper_bound_sqr() {
        return matcher_upper_bound_sqr_;
      }
      
      double lower_bound_sqr() const {
        return matcher_lower_bound_sqr_;
      } 
      
      double upper_bound_sqr() const {
        return matcher_upper_bound_sqr_;
      }
      
      double probability() const {
        return 1.0;
      }
      
    private:
      
      double matcher_distance_;
      double matcher_thickness_;
      
      double matcher_lower_bound_sqr_;
      double matcher_upper_bound_sqr_;
      
      bool is_monochromatic_;
      
      TableType* points_table_1_;
      TableType* points_table_2_;
      
    }; // class
    
  } // namespace
} // namespace



#endif