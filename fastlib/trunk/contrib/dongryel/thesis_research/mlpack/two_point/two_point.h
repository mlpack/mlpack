/*
 *  two_point.h
 *  
 *
 *  Created by William March on 9/12/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include <boost/program_options.hpp>
#include "core/table/table.h"
#include "mlpack/two_point/two_point_arguments.h"
#include "mlpack/two_point/two_point_dualtree.h"

namespace mlpack {
  namespace two_point {
    
    class TwoPointArgumentParser {
      
    private:
      
      
      
    public:
      
      template<typename TableType>
      static bool ParseArguments(
         boost::program_options::variables_map &vm,
         mlpack::two_point::TwoPointArguments<TableType> *arguments_out);
    
      static bool ConstructBoostVariableMap(
        const std::vector<std::string> &args,
        boost::program_options::variables_map *vm);

      static bool ConstructBoostVariableMap(
        int argc,
        char *argv[],
        boost::program_options::variables_map *vm);

    
    }; // ArgumentParser

    // Note: Kde is also templated by a KernelAuxType
    template<typename IncomingTableType>
    class TwoPoint {
      
    public:
      
      // KernelAuxType - don't think I need
      
      // ExpansionType (as in series expansion) - don't need
      
      // PostponedType 
      // ApplyDelta(qnode, rnode, global, delta, query_results)
      // ApplyContribution(global, metric, q_col, q_weight, r_col, r_weight)
      // stores the local expansion - don't need
      
      // GlobalType - something like is_monochromatic
      // needs is_monochromatic()
      // This holds the matcher info
      
      // ResultType - needs a ContainerType? 
      // actually just stores the count (and the weighted_count)
      // also num_prunes
      
      // DeltaType - used to make pruning decisions - is this where the matcher
      // goes?
      // DeteriministicCompute(metric, global, qnode, rnode, distance_range)
      // This computes the approximation
      
      // SummaryType
      // CanSummarize_(global, delta, distance_range, qnode, rnode, query_results)
      // StartReaccumulate()
      // Accumulate(global, query_results, q_index)
      // This makes the pruning decision based on the matcher in the GlobalType
      
      // StatisticType - this is the node statistic - don't need it
        // just uses the summary in KDE?
      // why does this have a summary?
      
      typedef IncomingTableType TableType;
      
      // don't need
      //typedef IncomingKernelAuxType KernelAuxType;
      
      // don't need
      //static const
      //enum mlpack::series_expansion::CartesianExpansionType ExpansionType =
      //KernelAuxType::ExpansionType;
      
      typedef mlpack::two_point::TwoPointPostponed PostponedType;
      
      typedef mlpack::two_point::TwoPointGlobal<TableType> GlobalType;
      
      typedef mlpack::two_point::TwoPointResult ResultType;
      
      typedef mlpack::two_point::TwoPointDelta DeltaType;
      
      typedef mlpack::two_point::TwoPointSummary SummaryType;
      
      typedef mlpack::two_point::TwoPointStatistic StatisticType;
      
      
      //////////// Methods ///////////////
      
      template<typename IncomingGlobalType>
      void Init(mlpack::two_point::TwoPointArguments<TableType> &arguments_in,
                IncomingGlobalType *global_in);
      
      
      void Compute(
         const mlpack::two_point::TwoPointArguments<TableType> &arguments_in,
         ResultType *result_out);
      
      
      TableType* query_table() {
        return points_table_1_;
      }
      
      TableType* reference_table() {
        return points_table_2_;
      }

      GlobalType& global() {
        return global_;
      }
      
      bool is_monochromatic() const {
        return is_monochromatic_;
      }
      
    private:
      
      TableType* points_table_1_;
      TableType* points_table_2_;
      
      bool is_monochromatic_;
      
      GlobalType global_;
      
      
    }; // class
    
    
  } // two_point
} // mlpack

