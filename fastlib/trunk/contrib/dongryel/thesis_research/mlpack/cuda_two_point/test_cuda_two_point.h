/*
 *  test_two_point.h
 *  
 *
 *  Created by William March on 9/22/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef MLPACK_TWO_POINT_TEST_TWO_POINT_H
#define MLPACK_TWO_POINT_TEST_TWO_POINT_H

#include <boost/test/unit_test.hpp>
#include "core/parallel/random_dataset_generator.h"
#include "core/math/math_lib.h"
#include "mlpack/two_point/two_point_dev.h"
#include "mlpack/cuda_two_point/cuda_two_point.h"

namespace mlpack {
  namespace cuda_two_point {
    namespace test_cuda_two_point {
      extern int num_dimensions_;
      extern int num_points_;
    }
    
    class TestCudaTwoPoint {
      
    private:
      
      // add weights later
      bool CheckAccuracy_(int query_num_tuples,
                          const int naive_num_tuples) {
        
        printf("naive num_tuples: %d, tree num tuples: %d\n",
               naive_num_tuples, query_num_tuples);
        
        return naive_num_tuples == query_num_tuples;

      } // CheckAccuracy
      
    public:
      
      template<typename MetricType, typename TableType>
      static int UltraNaive(const MetricType &metric_in,
                             TableType &query_table, TableType &reference_table,
                             bool is_monochromatic,
                             double matcher_distance,
                             double matcher_thickness) {
        
        //printf("naive dist: %g, thick: %g\n", matcher_distance, matcher_thickness);
        
        int num_tuples = 0;
        
        double matcher_lower_bound_sqr = (matcher_distance - 0.5*matcher_thickness)
                              * (matcher_distance - 0.5*matcher_thickness);
        double matcher_upper_bound_sqr = (matcher_distance + 0.5*matcher_thickness)
        * (matcher_distance + 0.5*matcher_thickness);
        
        for(int i = 0; i < query_table.n_entries(); i++) {
          arma::vec query_point;
          query_table.get(i, &query_point);
          
          //query_point.print();
          //printf("\n");
          
          int ref_start = is_monochromatic ? i+1 : 0;
          
          for(int j = ref_start; j < reference_table.n_entries(); j++) {
            arma::vec reference_point;
            reference_table.get(j, &reference_point);
            
            // By default, monochromaticity is assumed in the test -
            // this will be addressed later for general bichromatic
            // test.
            //if(i == j) {
            //  continue;
            //}
            
            double squared_distance =
            metric_in.DistanceSq(query_point, reference_point);
            
            if (matcher_lower_bound_sqr <= squared_distance 
                && squared_distance <= matcher_upper_bound_sqr) {
              
              // printf("tuple: (%d, %d)\n", i, j);
              
              num_tuples++;
              
            }
            
            
          }
          
        }
        
        return num_tuples;
        
        //printf("Ultra naive num_tuples: %d\n", *ultra_naive_num_tuples);
        
      } // UltraNaive
      
      int StressTestMain() {
        
        int retval;
        for(int i = 0; i < 20; i++) {
          
          // Randomly choose the number of dimensions and the points.
          mlpack::cuda_two_point::test_cuda_two_point::num_dimensions_ = 3;
          mlpack::cuda_two_point::test_cuda_two_point::num_points_ = core::math::RandInt(500, 1001);
        
          //mlpack::two_point::test_two_point::num_dimensions_ = 3;
          //mlpack::two_point::test_two_point::num_points_ = 20;
          
          
          retval = StressTest();
          
          if (retval) {
            break;
          }
          
                      
        }
        return retval;
      }
      
      int StressTest() {
        
        typedef core::table::Table <core::tree::GenMetricTree <
                                      mlpack::two_point::TwoPointStatistic> ,
                              mlpack::two_point::TwoPointResult > TableType;
        
        // The list of arguments.
        std::vector< std::string > args;
        
        // Push in the reference dataset name.
        std::string references_in("random.csv");
        args.push_back(std::string("--data_in=") + references_in);
        
        // Push in the densities output file name.
        args.push_back(std::string("--counts_out=counts.txt"));
        
        std::cout << "\n==================\n";
        std::cout << "Test trial begin\n";
        std::cout << "Number of dimensions: " <<
        mlpack::cuda_two_point::test_cuda_two_point::num_dimensions_ << "\n";
        std::cout << "Number of points: " <<
        mlpack::cuda_two_point::test_cuda_two_point::num_points_ << "\n";
        
        
        // Push in the randomly generated matcher_distance.
        double matcher_distance =
        core::math::Random(0.1 * sqrt(mlpack::cuda_two_point::test_cuda_two_point::num_dimensions_),
                           0.5 * sqrt(mlpack::cuda_two_point::test_cuda_two_point::num_dimensions_));
        std::stringstream matcher_distance_sstr;
        matcher_distance_sstr << "--matcher_distance=" << matcher_distance;
        args.push_back(matcher_distance_sstr.str());

        // Push in the randomly generated matcher_thickness.
        double matcher_thickness =
        core::math::Random(0.01 * sqrt(mlpack::cuda_two_point::test_cuda_two_point::num_dimensions_),
                           0.025 * sqrt(mlpack::cuda_two_point::test_cuda_two_point::num_dimensions_));
        std::stringstream matcher_thickness_sstr;
        matcher_thickness_sstr << "--matcher_thickness=" << matcher_thickness;
        args.push_back(matcher_thickness_sstr.str());
        
        // Generate the random dataset and save it.
        TableType random_table;
        core::parallel::RandomDatasetGenerator::Generate(
               mlpack::cuda_two_point::test_cuda_two_point::num_dimensions_,
               mlpack::cuda_two_point::test_cuda_two_point::num_points_, 0, 
               std::string("none"),
               false, &random_table);
        random_table.Save(references_in);
        
        // Parse the two point arguments.
        mlpack::two_point::TwoPointArguments<TableType> two_point_arguments;
        boost::program_options::variables_map vm;
        mlpack::two_point::TwoPointArgumentParser::ConstructBoostVariableMap(args, &vm);
        
        if(mlpack::two_point::TwoPointArgumentParser::ParseArguments(vm, 
                                                     &two_point_arguments)) {
          return 1;
        }
        //std::cout << "Matcher distance value " << matcher_distance << "\n";
        //std::cout << "Matcher thickness value " << matcher_thickness << "\n";
        
        mlpack::two_point::TwoPointResult two_point_result;
        
        StartComputation(two_point_arguments, &two_point_result);
        
        // Call the ultra-naive.
        int ultra_naive_two_point_result;
        
        double naive_match = two_point_arguments.matcher_distance_;
        double naive_thick = two_point_arguments.matcher_thickness_;
        //ultra_naive_two_point_result = UltraNaive(*(two_point_arguments.metric_), 
        //                                          *(two_point_arguments.points_table_1_),
        //                                          *(two_point_arguments.points_table_1_),
        //                                          matcher_distance, matcher_thickness);
        ultra_naive_two_point_result = UltraNaive(*(two_point_arguments.metric_), 
                                                  *(two_point_arguments.points_table_1_),
                                                  *(two_point_arguments.points_table_1_),
                                                  true,
                                                  naive_match, naive_thick);
        
        
        if(CheckAccuracy_(two_point_result.num_tuples_,
                          ultra_naive_two_point_result) == false) {
          std::cerr << "There is a problem!\n";
          
          
          return 1;
        }
        
        return 0;
      };
    };
  }
}
#endif
