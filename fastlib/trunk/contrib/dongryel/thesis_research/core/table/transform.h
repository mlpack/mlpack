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
      for(int i = 0; i < table_in->n_entries(); i++) {
        arma::vec point;
        table_in->get(i, &point);
        for(int d = 0; d < table_in->n_attributes(); d++) {
          mins[d] = std::min(mins[d], point[d]);
        }
      }

      // Now shift by the minimum along each coordinate.
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

      // Loop through each point in the table and compute
      // means/standard_deviations for each dimension.
      for(int i = 0; i < table_in->n_entries(); i++) {
        arma::vec point;
        table_in->get(i, &point);
        for(int d = 0; d < table_in->n_attributes(); d++) {
          double delta = point[d] - means[d];
          means[d] = means[d] + delta / static_cast<double>(i + 1);
          standard_deviations[d] = standard_deviations[d] + delta *
                                   (point[d] - means[d]);
        }
      }
      for(int d = 0; d < table_in->n_attributes(); d++) {
        standard_deviations[d] =
          sqrt(
            standard_deviations[d] /
            static_cast<double>(table_in->n_entries() - 1));
      }
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
      // means/standard_deviations for each dimension.
      for(int i = 0; i < table_in->n_entries(); i++) {
        arma::vec point;
        table_in->get(i, &point);
        for(int d = 0; d < table_in->n_attributes(); d++) {
          minimums[d] = std::min(minimums[d], point[d]);
          maximums[d] = std::max(maximums[d], point[d]);
        }
      }
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
