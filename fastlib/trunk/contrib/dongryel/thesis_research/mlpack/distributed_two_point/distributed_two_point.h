/*
 *  distributed_two_point.h
 *  
 *
 *  Created by William March on 9/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef MLPACK_DISTRIBUTED_TWO_POINT_H
#define MLPACK_DISTRIBUTED_TWO_POINT_H

#include <boost/program_options.hpp>
#include <boost/mpi/communicator.hpp>
#include "core/table/distributed_table.h"

#include "mlpack/two_point/two_point_dev.h"
#include "mlpack/two_point/two_point_dualtree.h"
#include "mlpack/two_point/two_point_arguments.h"
#include "mlpack/distributed_two_point/distributed_two_point_arguments.h"


namespace mlpack {
  namespace distributed_two_point {

    class DistributedTwoPointArgumentParser {
    public:
      
      template<typename DistributedTableType>
      static bool ParseArguments(boost::mpi::communicator &world,
                                 boost::program_options::variables_map &vm,
                                 mlpack::distributed_two_point::DistributedTwoPointArguments <
                                 DistributedTableType > *arguments_out);
      
      static bool ConstructBoostVariableMap(
                    boost::mpi::communicator &world,
                    const std::vector<std::string> &args,
                    boost::program_options::variables_map *vm);

      static bool ConstructBoostVariableMap(
                    boost::mpi::communicator &world,
                    int argc,
                    char *argv[],
                    boost::program_options::variables_map *vm);
    
      
    }; // class
    
    
    template<typename IncomingDistributedTableType>
    class DistributedTwoPoint {
    public:
      
      typedef IncomingDistributedTableType DistributedTableType;
      
      typedef typename DistributedTableType::TableType TableType;
      
      typedef mlpack::two_point::TwoPointPostponed PostponedType;
      
      typedef mlpack::two_point::TwoPointGlobal <DistributedTableType> GlobalType;
      
      typedef mlpack::two_point::TwoPointResult ResultType;
      
      typedef mlpack::two_point::TwoPointDelta DeltaType;
      
      typedef mlpack::two_point::TwoPointSummary SummaryType;
      
      typedef mlpack::two_point::TwoPointStatistic StatisticType;
      
      typedef mlpack::two_point::TwoPointArguments<TableType> ArgumentType;
      
      typedef mlpack::two_point::TwoPoint<TableType> ProblemType;
      
    public:
      
      /** @brief The default constructor.
       */
      DistributedTwoPoint() {
        world_ = NULL;
      }
      
      /** @brief returns a pointer to the query table
       */
      DistributedTableType *reference_table() {
        return points_table_1_;
      }
      
      /** @brief returns a pointer to the reference table
       */
      DistributedTableType *query_table() {
        return points_table_2_;
      }
      
      /** @brief returns a GlobalType structure that has the
       *         normalization statistics
       */
      GlobalType &global() {
        return global_;
      }
      
      /** @brief False if we're doing a DR-type count
       */
      bool is_monochromatic() const {
        return is_monochromatic_;
      }
      
      /** @brief Initialize a two point engine with the arguments.
       */
      void Init(boost::mpi::communicator &world_in,
                mlpack::distributed_two_point::DistributedTwoPointArguments <
                DistributedTableType > &arguments_in);
      
      void Compute(const mlpack::distributed_two_point::DistributedTwoPointArguments <
                   DistributedTableType > &arguments_in,
                   ResultType *result_out);
      
    private:
      
      boost::mpi::communicator *world_;
      
      /** @brief The distributed points table.
       */
      DistributedTableType *points_table_1_;
      
      /** @brief The distributed randoms (or other data) table.
       */
      DistributedTableType *points_table_2_;
      
      /** @brief The relevant global variables for the distributed two point
       *         computation.
       */
      GlobalType global_;
      
      /** @brief The flag that tells whether the computation is
       *         monochromatic.
       */
      bool is_monochromatic_;
    };
    
    
    
  } // namespace
} // namespace


#endif