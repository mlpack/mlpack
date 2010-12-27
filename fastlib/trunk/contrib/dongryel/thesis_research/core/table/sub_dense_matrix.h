/** @file sub_dense_matrix.h
 *
 *  A helper class for serializing the points under a given set of
 *  nodes.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_OFFSET_DENSE_MATRIX_H
#define CORE_TABLE_OFFSET_DENSE_MATRIX_H

#include "core/table/dense_matrix.h"
#include <boost/serialization/serialization.hpp>

namespace core {
namespace table {
template<typename TreeType>
class SubDenseMatrix {
  private:
    friend class boost::serialization::access;

    /** @brief The pointer to the dense matrix.
     */
    const core::table::DenseMatrix *matrix_;

    const std::vector<TreeType *> *nodes_;

    const std::vector<bool> *serialize_points_per_terminal_node_;

  public:

    SubDenseMatrix() {
      matrix_ = NULL;
    }

    void Init(
      const core::table::DenseMatrix *matrix_in,
      const std::vector<TreeType *> &nodes_in,
      const std::vector<bool> &serialze_points_per_terminal_node_in) {
      matrix_ = matrix_in;
      nodes_ = &nodes_in;
      serialize_points_per_terminal_nodes_ =
        &serialize_points_per_terminal_node_in;
    }

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      for(unsigned int j = 0; j < nodes_->size(); j++) {
        if(nodes_[j] != NULL && nodes_[j]->is_leaf()) {
          for(int i = nodes_[j]->begin(); i < nodes_[j]->end(); i++) {
            arma::vec column_ptr;
            matrix_->GetColumnPtr(i, &column_ptr);
            for(int k = 0; k < matrix_->n_attributes(); k++) {
              ar & column_ptr[k];
            }
          }
        }
      }
    }
};
};
};

#endif
