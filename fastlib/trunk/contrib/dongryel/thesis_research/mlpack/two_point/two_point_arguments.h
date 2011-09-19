/*
 *  2pt_arguments.h
 *  
 *
 *  Created by William March on 9/12/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TWO_POINT_ARGUMENTS_H
#define TWO_POINT_ARGUMENTS_H


#include "core/table/table.h"
#include "core/metric_kernels/lmetric.h"

namespace mlpack {
  namespace two_point {
    
    template<typename TableType>
    class TwoPointArguments {
      
    public:
      
      double matcher_distance_;
      double matcher_thickness_;
      
      core::metric_kernels::LMetric<2>* metric_;
      
      int leaf_size_;
      
      TableType* points_table_1_;
      TableType* points_table_2_;
      
      bool tables_are_aliased_;
      
      std::string counts_out;
      
      template<typename GlobalType>
      void Init(TableType* points_table_1_in, TableType* points_table_2_in,
                GlobalType& global_in)
      {
        
        matcher_distance_ = global_in.matcher_distance();
        matcher_thickness_ = global_in.matcher_thickness();
        
        points_table_1_ = points_table_1_in;
        points_table_2_ = points_table_2_in;
        
        tables_are_aliased_ = true;
        
        // what about metric_?
        
        
      } // Init()
      
      template<typename GlobalType>
      void Init(GlobalType &global_in)
      {
      
        matcher_distance_ = global_in.matcher_distance();
        matcher_thickness_ = global_in.matcher_thickness();
        
        points_table_1_ = global_in.points_table_1()->local_table();
        points_table_2_ = global_in.points_table_2()->local_table();
        
        tables_are_aliased_ = true;
        
      } // Init()
      
      TwoPointArguments() {
        
        leaf_size_ = 0;
        points_table_1_ = NULL;
        points_table_2_ = NULL;
        metric_ = NULL;
        tables_are_aliased_ = false;
        
        
      } // constructor
      
      ~TwoPointArguments() {
        
        if (!tables_are_aliased_) {
        
          if (points_table_1_ == points_table_2_) {
            if (points_table_1_ != NULL) {
              delete points_table_1_;
            }
          }
          else {
            if (points_table_1_ != NULL) {
              delete points_table_1_;
            }
            if (points_table_2_ != NULL) {
              delete points_table_2_;
            }
            
          }
           
        } // delete points
        
        points_table_1_ = NULL;
        points_table_2_ = NULL;
        
        if (metric_ != NULL) {
          delete metric_;
          metric_ = NULL;
        }
        
      } // destructor
      
      
    }; // class
    
    
  } // namespace
} // namespace



#endif