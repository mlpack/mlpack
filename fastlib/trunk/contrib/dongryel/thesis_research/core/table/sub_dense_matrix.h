/** @file sub_dense_matrix.h
 *
 *  A helper class for serializing the points under a given set of
 *  nodes.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_SUB_DENSE_MATRIX_H
#define CORE_TABLE_SUB_DENSE_MATRIX_H

#include "core/table/dense_matrix.h"
#include <boost/serialization/serialization.hpp>

namespace core {
namespace table {
template<typename SubTableType>
class SubDenseMatrix {
  private:
    friend class boost::serialization::access;

    /** @brief The pointer to the dense matrix.
     */
    core::table::DenseMatrix *matrix_;

    const std::vector< typename SubTableType::PointSerializeFlagType >
    *serialize_points_per_terminal_node_;

  public:

    bool is_alias() const {
      return matrix_->is_alias();
    }

    SubDenseMatrix() {
      matrix_ = NULL;
      serialize_points_per_terminal_node_ = NULL;
    }

    void Init(
      core::table::DenseMatrix *matrix_in,
      const std::vector< typename SubTableType::PointSerializeFlagType >
      &serialize_points_per_terminal_node_in) {
      matrix_ = matrix_in;
      serialize_points_per_terminal_node_ =
        &serialize_points_per_terminal_node_in;
    }

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // Save the dimensionality.
      int n_rows = matrix_->n_rows();
      int n_cols = matrix_->n_cols();
      ar & n_rows;
      ar & n_cols;

      for(unsigned int j = 0;
          j < serialize_points_per_terminal_node_->size(); j++) {
        for(int i = (*serialize_points_per_terminal_node_)[j].begin_;
            i < (*serialize_points_per_terminal_node_)[j].end(); i++) {
          const double *column_ptr = matrix_->GetColumnPtr(i);
          for(int k = 0; k < matrix_->n_rows(); k++) {
            ar & column_ptr[k];
          }
        }
      }
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the dimensionality.
      int n_rows;
      int n_cols;
      ar & n_rows;
      ar & n_cols;

      if(this->is_alias() == false) {
        matrix_->Init(n_rows, n_cols);
      }

      for(unsigned int j = 0;
          j < serialize_points_per_terminal_node_->size(); j++) {
        for(int i = (*serialize_points_per_terminal_node_)[j].begin_;
            i < (*serialize_points_per_terminal_node_)[j].end(); i++) {
          double *column_ptr = const_cast<double *>(matrix_->GetColumnPtr(i));
          for(int k = 0; k < matrix_->n_rows(); k++) {
            ar & column_ptr[k];
          }
        }
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
};
};

#endif
