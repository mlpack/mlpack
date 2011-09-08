/** @file map_matrix.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_MAP_MATRIX_H
#define CORE_PARALLEL_MAP_MATRIX_H

#include <boost/intrusive_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/serialization/serialization.hpp>
#include <map>

namespace boost {
template<typename T>
class MapMatrixInternal {
  private:

    int n_rows_;

    int n_cols_;

    std::map<int, int> id_to_position_map_;

    std::map<int, int> position_to_id_map_;

    boost::scoped_array<T> matrix_;

  public:
    long num_references_;

  public:

    std::map<int, int> &id_to_position_map() {
      return id_to_position_map_;
    }

    std::map<int, int> &position_to_id_map() {
      return position_to_id_map_;
    }

    T *matrix() {
      return matrix_.get();
    }

    int n_rows() const {
      return n_rows_;
    }

    int n_cols() const {
      return n_cols_;
    }

    MapMatrixInternal() {
      n_rows_ = 0;
      n_cols_ = 0;
      num_references_ = 0;
    }

    void Init(int num_rows_in, int num_cols_in) {
      n_rows_ = num_rows_in;
      n_cols_ = num_cols_in;
      boost::scoped_array<T> tmp_matrix(new T[n_rows_ * n_cols_]);
      matrix_.swap(tmp_matrix);
    }

    /** @brief Returns a const pointer to the object with the given ID
     *         for the column.
     */
    const T *col(int i) const {
      int col_start = i;
      if(id_to_position_map_.size() > 0) {
        col_start = (id_to_position_map_.find(col_start)->second);
      }
      col_start *= n_rows_;
      return matrix_.get() + col_start;
    }

    /** @brief Returns a modifiable reference to the object with the
     *         given ID.
     */
    T *col(int i) {
      int col_start = i;
      if(id_to_position_map_.size() > 0) {
        col_start = (id_to_position_map_.find(col_start)->second);
      }
      col_start *= n_rows_;
      return matrix_.get() + col_start;
    }
};
}

namespace core {
namespace parallel {

template<typename T>
class MapMatrix {
  private:

    // For BOOST serialization.
    friend class boost::serialization::access;

  private:

    boost::intrusive_ptr< boost::MapMatrixInternal<T> > internal_;

    std::vector<int> indices_to_save_;

  public:

    class iterator {
      private:
        T *ptr_;

        std::map<int, int> *position_to_id_map_;

        int current_index_;

        int current_pos_;

        int num_rows_;

        int num_cols_;

      public:

        std::map<int, int> *position_to_id_map() {
          return position_to_id_map_;
        }

        int current_id() const {
          return position_to_id_map_->find(current_index_)->second;
        }

        int n_cols() const {
          return num_cols_;
        }

        int n_rows() const {
          return num_rows_;
        }

        int current_index() const {
          return current_index_;
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
          current_index_ = it_in.current_index();
          current_pos_ = it_in.current_pos();
          num_rows_ = it_in.n_rows();
          num_cols_ = it_in.n_cols();
        }

        iterator(const iterator &it_in) {
          this->operator=(it_in);
        }

        iterator(
          const T *ptr_in, const std::map<int, int> *position_to_id_map_in,
          int num_rows_in, int num_cols_in) {
          ptr_ = const_cast<T *>(ptr_in);
          position_to_id_map_ =
            const_cast< std::map<int, int> * >(position_to_id_map_in);
          current_index_ = 0;
          current_pos_ = 0;
          num_rows_ = num_rows_in;
          num_cols_ = num_cols_in;
        }

        iterator() {
          position_to_id_map_ = NULL;
          ptr_ = NULL;
          current_index_ = 0;
          current_pos_ = 0;
          num_rows_ = 0;
          num_cols_ = 0;
        }

        bool HasNext() const {
          return current_index_ < num_cols_;
        }

        void operator++(int) {
          current_index_++;
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

    template<typename TreeIteratorType>
    void Alias(
      const core::parallel::MapMatrix<T> &source_in,
      TreeIteratorType &it) {
      internal_ = source_in.internal();
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
        for(int j = 0; j < this->n_rows(); j++) {
          destination_start[j] = source_start[j];
        }
        source_it++;
      }
    }

    /** @brief Gets an iterator of the current map matrix object.
     */
    iterator get_iterator() const {
      return iterator(
               internal_->matrix(), & internal_->position_to_id_map(),
               internal_->n_rows(), internal_->n_cols());
    }

    /** @brief Returns the number of rows stored in the map matrix.
     */
    int n_rows() const {
      return internal_->n_rows();
    }

    /** @brief Returns the number of columns stored in the map matrix.
     */
    int n_cols() const {
      return internal_->n_cols();
    }

    /** @brief The default constructor.
     */
    MapMatrix() {
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
      int num_rows = internal_->n_rows();
      ar & num_rows;
      int num_columns_to_save = indices_to_save_.size();
      ar & num_columns_to_save;

      // Save each element and its mapping.
      for(int i = 0; i < num_columns_to_save; i++) {
        int translated_position = indices_to_save_[i];
        const T *column = this->col(translated_position) ;
        for(int j = 0; j < num_rows; j++) {
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
      int num_rows, num_cols;
      ar & num_rows;
      ar & num_cols;
      this->Init(num_rows, num_cols);
      std::map<int, int> &id_to_position_map =
        internal_->id_to_position_map();
      std::map<int, int> &position_to_id_map =
        internal_->position_to_id_map();

      // Load each element and its mapping.
      T * column = internal_->matrix();
      for(int i = 0; i < num_cols; i++, column += num_rows) {
        int original_index;
        for(int j = 0; j < num_rows; j++) {
          ar & column[j];
        }
        ar & original_index;
        id_to_position_map[ original_index ] = i;
        position_to_id_map[ i ] = original_index;
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /** @brief Initializes the map vector for a given number of rows
     *         and columns.
     */
    void Init(int num_rows_in, int num_cols_in) {
      boost::intrusive_ptr <
      boost::MapMatrixInternal<T> > tmp_internal(
        new boost::MapMatrixInternal<T>());
      internal_.swap(tmp_internal);
      internal_->Init(num_rows_in, num_cols_in);
    }

    /** @brief Returns a const pointer to the object with the given ID
     *         for the column.
     */
    const T *col(int i) const {
      return internal_->col(i);
    }

    /** @brief Returns a modifiable reference to the object with the
     *         given ID.
     */
    T *col(int i) {
      return internal_->col(i);
    }
};
}
}

namespace boost {
template<typename T>
inline void intrusive_ptr_add_ref(boost::MapMatrixInternal<T> *ptr) {
  ptr->num_references_++;
}

template<typename T>
inline void intrusive_ptr_release(boost::MapMatrixInternal<T> *ptr) {
  ptr->num_references_--;
  if(ptr->num_references_ == 0) {
    if(core::table::global_m_file_) {
      core::table::global_m_file_->DestroyPtr(ptr);
    }
    else {
      delete ptr;
    }
  }
}
}

#endif
