/** @file random_dataset_generator.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_RANDOM_DATASET_GENERATOR_H
#define CORE_PARALLEL_RANDOM_DATASET_GENERATOR_H

#include <iostream>
#include <omp.h>
#include "core/math/math_lib.h"
#include "core/table/transform.h"

namespace core {
namespace parallel {
class RandomDatasetGenerator {
  public:
    template<typename TableType>
    static void Generate(
      int num_dimensions, int num_points,
      const std::string &prescale_option, int num_threads_in,
      TableType *random_dataset) {

      random_dataset->Init(num_dimensions, num_points);

#pragma omp parallel for
      for(int j = 0; j < num_points; j++) {
        core::table::DensePoint point;
        random_dataset->get(j, &point);
        for(int i = 0; i < num_dimensions; i++) {
          point[i] = core::math::Random(0.1, 1.0);
        }

        // Set the weight to the random one.
        random_dataset->weights().set(
          0, j, core::math::Random(1.0, 5.0));
      }

      // Scale the dataset.
      if(prescale_option == "hypercube") {
        core::table::UnitHypercube::Transform(random_dataset, num_threads_in);
      }
      else if(prescale_option == "standardize") {
        core::table::Standardize::Transform(random_dataset, num_threads_in);
      }

      // Now, make sure that all coordinates are non-negative.
      if(prescale_option != "hypercube") {
        core::table::TranslateToNonnegative::Transform(
          random_dataset, num_threads_in);
      }

      std::cout << "Scaled the dataset with the option: " <<
                prescale_option << "\n";
    }
};
}
}

#endif
