/** @file sub_dense_matrix.h
 *
 *  A helper class for serializing the points under a given set of
 *  nodes.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_SUB_DENSE_MATRIX_H
#define CORE_TABLE_SUB_DENSE_MATRIX_H

#include <boost/serialization/serialization.hpp>
#include "core/table/dense_matrix.h"

namespace core {
namespace table {

/** @brief A class representing a subset of a dense matrix.
 */
template<typename SubTableType>
class SubDenseMatrix {
  private:

    // For boost serialization.
    friend class boost::serialization::access;

    /** @brief The pointer to the dense matrix. The loading/unloading
     *         is done from here.
     */
    core::table::DenseMatrix *matrix_;

    /** @brief The list of begin/count pairs to serialize from the
     *         dense matrix.
     */
    const std::vector< typename SubTableType::PointSerializeFlagType >
    *serialize_points_per_terminal_node_;

  public:

    /** @brief Returns whether the sub dense matrix is an alias or
     *         not.
     */
    bool is_alias() const {
      return matrix_->is_alias();
    }

    /** @brief The default constructor.
     */
    SubDenseMatrix() {
      matrix_ = NULL;
      serialize_points_per_terminal_node_ = NULL;
    }

    /** @brief Initialize a sub dense matrix class for
     *         serialization/unserialization.
     */
    void Init(
      core::table::DenseMatrix *matrix_in,
      const std::vector< typename SubTableType::PointSerializeFlagType >
      &serialize_points_per_terminal_node_in) {
      matrix_ = matrix_in;
      serialize_points_per_terminal_node_ =
        &serialize_points_per_terminal_node_in;
    }

    /** @brief Serialize a subset of a dense matrix.
     */
    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // Save the dimensionality.
      int n_rows = matrix_->n_rows();
      int n_cols = matrix_->n_cols();
      ar & n_rows;
      ar & n_cols;

      // Since we are extracting an already well-formed matrix, we use
      // the direct mapping.
      for(unsigned int j = 0;
          j < serialize_points_per_terminal_node_->size(); j++) {
        for(int i = (*serialize_points_per_terminal_node_)[j].begin();
            i < (*serialize_points_per_terminal_node_)[j].end(); i++) {
          const double *column_ptr = matrix_->GetColumnPtr(i);
          for(int k = 0; k < matrix_->n_rows(); k++) {
            ar & column_ptr[k];
          }
        }
      }
    }

    /** @brief Unserialize a subset of a dense matrix.
     */
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

      // Serialize onto a consecutive block of memory.
      int index = 0;
      for(unsigned int j = 0;
          j < serialize_points_per_terminal_node_->size(); j++) {
        for(int i = (*serialize_points_per_terminal_node_)[j].begin();
            i < (*serialize_points_per_terminal_node_)[j].end(); i++, index++) {
          double *column_ptr =
            const_cast<double *>(matrix_->GetColumnPtr(index));
          for(int k = 0; k < matrix_->n_rows(); k++) {
            ar & column_ptr[k];
          }
        }
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
}
}

#endif
