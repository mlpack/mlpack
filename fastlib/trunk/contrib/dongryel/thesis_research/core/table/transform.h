/** @file transform.h
 *
 *  @brief Transforms the table such that it is standardized or scaled to
 *         a unit hypercube.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_TRANSFORM_H
#define CORE_TABLE_TRANSFORM_H

#include <omp.h>

namespace core {
namespace table {

class TranslateToNonnegative {

  public:

    template<typename TableType>
    static void Transform(TableType *table_in) {

      // Get the minimum coordinates of each dimension.
      std::vector<double> mins(
        table_in->n_attributes(), std::numeric_limits<double>::max());

      #pragma omp parallel
      {
        std::vector<double> local_mins(
          table_in->n_attributes(), std::numeric_limits<double>::max());

        #pragma omp for
        for(int i = 0; i < table_in->n_entries(); i++) {
          arma::vec point;
          table_in->get(i, &point);
          for(int d = 0; d < table_in->n_attributes(); d++) {
            local_mins[d] = std::min(local_mins[d], point[d]);
          }
        }

        // The final reduction.
        #pragma omp critical
        {
          for(int d = 0; d < table_in->n_attributes(); d++) {
            mins[d] = std::min(local_mins[d], mins[d]);
          }
        }
      } // end of parallel region.

      // Now shift by the minimum along each coordinate.
      #pragma omp parallel for
      for(int i = 0; i < table_in->n_entries(); i++) {
        arma::vec point;
        table_in->get(i, &point);
        for(unsigned int d = 0; d < point.n_elem; d++) {
          if(mins[d] < 0.0) {
            point[d] -= mins[d];
          }
        }
      }
    }
};

class Standardize {
  public:

    template<typename TableType>
    static void Transform(TableType *table_in) {

      // Means and standard deviations of each dimension.
      std::vector<double> means(table_in->n_attributes(), 0.0);
      std::vector<double> standard_deviations(table_in->n_attributes(), 0.0);
      int accum_count = 0;

      #pragma omp parallel
      {
        // Means and standard deviations of each dimension.
        std::vector<double> local_means(table_in->n_attributes(), 0.0);
        std::vector<double> local_standard_deviations(
          table_in->n_attributes(), 0.0);

        // Loop through each point in the table and compute
        // means/standard_deviations for each dimension.
        int local_count = 0;
        #pragma omp for
        for(int i = 0; i < table_in->n_entries(); i++) {
          arma::vec point;
          table_in->get(i, &point);
          for(int d = 0; d < table_in->n_attributes(); d++) {
            double delta = point[d] - local_means[d];
            local_means[d] =
              local_means[d] + delta / static_cast<double>(local_count + 1);
            local_standard_deviations[d] =
              local_standard_deviations[d] + delta *
              (point[d] - local_means[d]);
            local_count++;
          }
        }

        #pragma omp critical
        {
          for(int d = 0; d < table_in->n_attributes(); d++) {
            double delta = local_means[d] - means[d];
            means[d] += delta * local_count /
                        static_cast<double>(local_count + accum_count);
            standard_deviations[d] +=
              local_standard_deviations[d] + delta * delta *
              accum_count * local_count /
              static_cast<double>(local_count + accum_count);
          }
          accum_count += local_count;
        }
      } // end of parallel region.

      for(int d = 0; d < table_in->n_attributes(); d++) {
        standard_deviations[d] =
          sqrt(
            standard_deviations[d] /
            static_cast<double>(table_in->n_entries() - 1));
      }

      // Now transform each point in parallel.
      #pragma omp parallel for
      for(int i = 0; i < table_in->n_entries(); i++) {
        arma::vec point;
        table_in->get(i, &point);
        for(int d = 0; d < table_in->n_attributes(); d++) {
          if(standard_deviations[d] > 0 && (! isnan(standard_deviations[d])) &&
              (! isinf(standard_deviations[d]))) {
            point[d] = (point[d] - means[d]) / standard_deviations[d] ;
          }
        }
      }
    }
};

class UnitHypercube {
  public:

    template<typename TableType>
    static void Transform(TableType *table_in) {

      // The minimum and maximum values of each dimension.
      std::vector<double> minimums(
        table_in->n_attributes(), std::numeric_limits<double>::max());
      std::vector<double> maximums(
        table_in->n_attributes(), -std::numeric_limits<double>::max());

      // Loop through each point in the table and compute
      // mins/maxs for each dimension.
      #pragma omp parallel
      {
        std::vector<double> local_minimums(
          table_in->n_attributes(), std::numeric_limits<double>::max());
        std::vector<double> local_maximums(
          table_in->n_attributes(), -std::numeric_limits<double>::max());

        #pragma omp for
        for(int i = 0; i < table_in->n_entries(); i++) {
          arma::vec point;
          table_in->get(i, &point);
          for(int d = 0; d < table_in->n_attributes(); d++) {
            local_minimums[d] = std::min(local_minimums[d], point[d]);
            local_maximums[d] = std::max(local_maximums[d], point[d]);
          }
        }

        #pragma omp critical
        {
          for(int d = 0; d < table_in->n_attributes(); d++) {
            minimums[d] = std::min(local_minimums[d], minimums[d]);
            maximums[d] = std::max(local_maximums[d], maximums[d]);
          }
        }
      } // end of the parallel region.

      // Now transform each point in parallel.
      #pragma omp parallel for
      for(int i = 0; i < table_in->n_entries(); i++) {
        arma::vec point;
        table_in->get(i, &point);
        for(int d = 0; d < table_in->n_attributes(); d++) {
          double length = maximums[d] - minimums[d];
          if(length > 0.0) {
            point[d] = (point[d] - minimums[d]) / length;
          }
        }
      }
    }
};
}
}

#endif
