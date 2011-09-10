/** @file sub_dense_matrix.h
 *
 *  A helper class for serializing the points under a given set of
 *  nodes.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_SUB_DENSE_MATRIX_H
#define CORE_TABLE_SUB_DENSE_MATRIX_H

#include <armadillo>
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

    std::map<int, int> *id_to_position_map_;

    /** @brief The pointer to the dense matrix. The loading/unloading
     *         is done from here.
     */
    arma::mat *matrix_;

    std::map<int, int> *position_to_id_map_;

    /** @brief The list of begin/count pairs to serialize from the
     *         dense matrix.
     */
    std::vector< typename SubTableType::PointSerializeFlagType >
    *serialize_points_per_terminal_node_;

  public:

    /** @brief The default constructor.
     */
    SubDenseMatrix() {
      id_to_position_map_ = NULL;
      matrix_ = NULL;
      position_to_id_map_ = NULL;
      serialize_points_per_terminal_node_ = NULL;
    }

    /** @brief Initialize a sub dense matrix class for
     *         serialization/unserialization.
     */
    void Init(
      arma::mat *matrix_in,
      std::vector< typename SubTableType::PointSerializeFlagType >
      *serialize_points_per_terminal_node_in,
      std::map<int, int> &id_to_position_map_in,
      std::map<int, int> &position_to_id_map_in) {
      id_to_position_map_ = &id_to_position_map_in;
      position_to_id_map_ = &position_to_id_map_in;
      matrix_ = matrix_in;
      serialize_points_per_terminal_node_ =
        serialize_points_per_terminal_node_in;
    }

    /** @brief Serialize a subset of a dense matrix.
     */
    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // Save the dimensionality.
      int n_rows = matrix_->n_rows;
      int n_cols = 0;
      for(unsigned int j = 0;
          j < serialize_points_per_terminal_node_->size(); j++) {
        n_cols += ((*serialize_points_per_terminal_node_)[j]).count();
      }
      ar & n_rows;
      ar & n_cols;

      for(unsigned int j = 0;
          j < serialize_points_per_terminal_node_->size(); j++) {
        for(int i = (*serialize_points_per_terminal_node_)[j].begin();
            i < (*serialize_points_per_terminal_node_)[j].end(); i++) {
          int translated_index = i;
          typename std::map<int, int>::const_iterator it =
            id_to_position_map_->find(translated_index);
          if(it != id_to_position_map_->end()) {
            translated_index = it->second;
          }
          const double *column_ptr =
            core::table::GetColumnPtr(*matrix_, translated_index);
          for(unsigned int k = 0; k < matrix_->n_rows; k++) {
            ar & column_ptr[k];
          }

          // Save the ID.
          ar & i;
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

      if(matrix_->n_elem == 0) {
        matrix_->set_size(n_rows, n_cols);
      }

      // Serialize onto a consecutive block of memory.
      int current_begin = 0;
      int previous_index = 0;
      for(int index = 0; index < n_cols; index++) {
        double *column_ptr = core::table::GetColumnPtr(*matrix_, index);
        for(unsigned int k = 0; k < matrix_->n_rows; k++) {
          ar & column_ptr[k];
        }

        // Load the ID and complete the mapping.
        int id;
        ar & id;
        (*id_to_position_map_)[id] = index;
        (*position_to_id_map_)[index] = id;
        if(index == 0) {
          current_begin = id;
        }
        // Add to the list of begin/count pairs if there is a break in
        // continuity.
        else if(serialize_points_per_terminal_node_ != NULL &&
                previous_index + 1 != id) {
          serialize_points_per_terminal_node_->push_back(
            typename SubTableType::PointSerializeFlagType(
              current_begin, previous_index - current_begin + 1));
          current_begin = id;
        }
        previous_index = id;
      }

      // Add the last one to the list.
      if(serialize_points_per_terminal_node_ != NULL) {
        serialize_points_per_terminal_node_->push_back(
          typename SubTableType::PointSerializeFlagType(
            current_begin, previous_index - current_begin + 1));
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
}
}

#endif
