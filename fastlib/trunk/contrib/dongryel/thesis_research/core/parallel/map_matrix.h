/** @file map_matrix.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_MAP_MATRIX_H
#define CORE_PARALLEL_MAP_MATRIX_H

#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <map>

namespace core {
namespace parallel {

template<typename T>
class MapMatrix {
  private:

    // For BOOST serialization.
    friend class boost::serialization::access;

  private:
    boost::shared_ptr< std::map<int, int> > id_to_position_map_;

    boost::shared_ptr< std::map<int, int> > position_to_id_map_;

    std::vector<int> indices_to_save_;

    int num_cols_;

    int num_rows_;

    boost::shared_array<T> matrix_;

  public:

    class iterator {
      private:
        T *ptr_;

        std::map<int, int> *position_to_id_map_;

        int current_pos_;

        int num_rows_;

        int num_cols_;

      public:

        std::map<int, int> *position_to_id_map() {
          return position_to_id_map_;
        }

        int current_id() const {
          return position_to_id_map_->find(current_pos_)->second;
        }

        int n_cols() const {
          return num_cols_;
        }

        int n_rows() const {
          return num_rows_;
        }

        int current_pos() const {
          return current_pos_;
        }

        const T *ptr() const {
          return ptr_;
        }

        T *ptr() {
          return ptr_;
        }

        T *operator->() {
          return &(ptr_[current_pos_]);
        }

        void operator=(const iterator &it_in) {
          ptr_ = const_cast<iterator &>(it_in).ptr();
          position_to_id_map_ =
            const_cast<iterator &>(it_in).position_to_id_map();
          current_pos_ = const_cast<iterator &>(it_in).current_pos();
          num_rows_ = const_cast<iterator &>(it_in).n_rows();
          num_cols_ = const_cast<iterator &>(it_in).n_cols();
        }

        iterator(const iterator &it_in) {
          this->operator=(it_in);
        }

        iterator(
          const T *ptr_in, const std::map<int, int> *position_to_id_map_in,
          int current_pos_in, int num_rows_in, int num_cols_in) {
          ptr_ = const_cast<T *>(ptr_in);
          position_to_id_map_ =
            const_cast< std::map<int, int> * >(position_to_id_map_in);
          current_pos_ = current_pos_in;
          num_rows_ = num_rows_in;
          num_cols_ = num_cols_in;
        }

        iterator() {
          position_to_id_map_ = NULL;
          ptr_ = NULL;
          current_pos_ = 0;
          num_rows_ = 0;
          num_cols_ = 0;
        }

        bool HasNext() const {
          return current_pos_ < num_cols_;
        }

        void operator++(int) {
          current_pos_ += num_rows_;
        }

        bool operator !=(const iterator &it_in) {
          return ptr_ != it_in.ptr() ||
                 current_pos_ != it_in.current_pos() ||
                 num_rows_ != it_in.num_rows() ||
                 num_cols_ != it_in.num_cols();
        }
    };

  public:

    const boost::shared_ptr <
    std::map<int, int> > &id_to_position_map() const {
      return id_to_position_map_;
    }

    const boost::shared_ptr <
    std::map<int, int> > &position_to_id_map() const {
      return position_to_id_map_;
    }

    const boost::shared_array<T> &matrix() const {
      return matrix_;
    }

    template<typename TreeIteratorType>
    void Alias(
      const core::parallel::MapMatrix<T> &source_in,
      TreeIteratorType &it) {
      id_to_position_map_ = source_in.id_to_position_map();
      position_to_id_map_ = source_in.position_to_id_map();
      num_rows_ = source_in.num_rows();
      num_cols_ = source_in.num_cols();
      matrix_ = source_in.matrix();
      this->set_indices_to_save(it);
    }

    /** @brief Copies another map matrix onto this matrix. Assumes
     *         that the mappings of the source and the destination are
     *         the same.
     */
    void Copy(const core::parallel::MapMatrix<T> &source_in) {
      iterator source_it = source_in.get_iterator();
      while(source_it.HasNext()) {
        T *destination_start = this->col(source_it.current_id());
        const T *source_start = source_it.ptr();
        for(int j = 0; j < num_rows_; j++) {
          destination_start[j] = source_start[j];
        }
        source_it++;
      }
    }

    /** @brief Gets an iterator of the current map matrix object.
     */
    iterator get_iterator() const {
      return iterator(
               matrix_.get(), position_to_id_map_.get(), 0,
               num_rows_, num_cols_);
    }

    /** @brief Returns the number of rows stored in the map matrix.
     */
    int n_rows() const {
      return num_rows_;
    }

    /** @brief Returns the number of columns stored in the map matrix.
     */
    int n_cols() const {
      return num_cols_;
    }

    /** @brief The default constructor.
     */
    MapMatrix() {
      num_rows_ = 0;
      num_cols_ = 0;
      indices_to_save_.resize(0);
    }

    /** @brief Call this function before serializing a subset of
     *         vector.
     */
    template<typename TreeIteratorType>
    void set_indices_to_save(TreeIteratorType &it) {
      indices_to_save_.resize(0);
      this->add_indices_to_save(it);
    }

    /** @brief Adds an additional set of indices to save.
     */
    template<typename TreeIteratorType>
    void add_indices_to_save(TreeIteratorType &it) {
      while(it.HasNext()) {
        int point_id;
        it.Next(&point_id);
        indices_to_save_.push_back(point_id);
      }
    }

    /** @brief Save the object.
     */
    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // Save the number of rows/columns being saved.
      ar & num_rows_;
      int num_columns_to_save = indices_to_save_.size();
      ar & num_columns_to_save;

      // Save each element and its mapping.
      for(int i = 0; i < num_columns_to_save; i++) {
        int translated_position = indices_to_save_[i];
        const T *column = this->col(translated_position) ;
        for(int j = 0; j < num_rows_; j++) {
          ar & column[ j ];
        }
        ar & indices_to_save_[i];
      }
    }

    /** @brief Load the object.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the number of rows/columns.
      ar & num_rows_;
      ar & num_cols_;
      boost::shared_array<T> tmp_array(new T[ num_rows_ * num_cols_ ]);
      matrix_.swap(tmp_array);
      boost::shared_ptr <
      std::map<int, int> > tmp_id_to_position_map(new std::map<int, int>());
      id_to_position_map_.swap(tmp_id_to_position_map);
      boost::shared_ptr <
      std::map<int, int> > tmp_position_to_id_map(new std::map<int, int>());
      position_to_id_map_.swap(tmp_position_to_id_map);

      // Load each element and its mapping.
      T * column = matrix_.get();
      for(int i = 0; i < num_cols_; i++, column += num_rows_) {
        int original_index;
        for(int j = 0; j < num_rows_; j++) {
          ar & column[j];
        }
        ar & original_index;
        (*id_to_position_map_)[ original_index ] = i;
        (*position_to_id_map_)[ i ] = original_index;
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /** @brief Initializes the map vector for a given number of rows
     *         and columns.
     */
    void Init(int num_rows_in, int num_cols_in) {
      boost::shared_array<T> tmp_vector(new T[num_rows_in * num_cols_in]);
      matrix_.swap(tmp_vector);
      boost::shared_ptr <
      std::map<int, int> > tmp_id_to_position_map(new std::map<int, int>());
      id_to_position_map_.swap(tmp_id_to_position_map);
      boost::shared_ptr <
      std::map<int, int> > tmp_position_to_id_map(new std::map<int, int>());
      position_to_id_map_.swap(tmp_position_to_id_map);
      num_rows_ = num_rows_in;
      num_cols_ = num_cols_in;
    }

    /** @brief Returns a const pointer to the object with the given ID
     *         for the column.
     */
    const T *col(int i) const {
      int col_start = i;
      if(id_to_position_map_->size() > 0) {
        col_start = (id_to_position_map_->find(col_start)->second);
      }
      col_start *= n_rows_;
      return matrix_.get() + col_start;
    }

    /** @brief Returns a modifiable reference to the object with the
     *         given ID.
     */
    T *col(int i) {
      int col_start = i;
      if(id_to_position_map_->size() > 0) {
        col_start = (id_to_position_map_->find(col_start)->second);
      }
      col_start *= n_rows_;
      return matrix_.get() + col_start;
    }
};
}
}

#endif
