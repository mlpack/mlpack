/** @file map_vector.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_MAP_VECTOR_H
#define CORE_PARALLEL_MAP_VECTOR_H

#include <boost/scoped_array.hpp>
#include <boost/serialization/serialization.hpp>
#include <map>

namespace core {
namespace parallel {

template<typename T>
class MapVector {
  private:

    // For BOOST serialization.
    friend class boost::serialization::access;

  private:
    std::map<int, int> id_to_position_map_;

    std::map<int, int> position_to_id_map_;

    std::vector<int> indices_to_save_;

    unsigned int num_elements_;

    boost::scoped_array<T> vector_;

  public:

    class iterator {
      private:
        T *ptr_;

        std::map<int, int> *position_to_id_map_;

        int current_pos_;

        int num_elements_;

      public:

        int current_id() const {
          return ((*position_to_id_map_)[ current_pos_ ]);
        }

        int num_elements() const {
          return num_elements_;
        }

        int current_pos() const {
          return current_pos_;
        }

        T *ptr() {
          return ptr_;
        }

        T &operator*() {
          return ptr_[current_pos_];
        }

        T *operator->() {
          return &(ptr_[current_pos_]);
        }

        void operator=(const iterator &it_in) {
          ptr_ = const_cast<iterator &>(it_in).ptr();
          current_pos_ = const_cast<iterator &>(it_in).current_pos();
          num_elements_ = const_cast<iterator &>(it_in).num_elements();
        }

        iterator(const iterator &it_in) {
          this->operator=(it_in);
        }

        iterator(
          const T *ptr_in, const std::map<int, int> *position_to_id_map_in,
          int current_pos_in, int num_elements_in) {
          ptr_ = const_cast<T *>(ptr_in);
          position_to_id_map_ =
            const_cast< std::map<int, int> * >(position_to_id_map_in);
          current_pos_ = current_pos_in;
          num_elements_ = num_elements_in;
        }

        iterator() {
          position_to_id_map_ = NULL;
          ptr_ = NULL;
          current_pos_ = 0;
          num_elements_ = 0;
        }

        bool HasNext() const {
          return current_pos_ < num_elements_;
        }

        void operator++(int) {
          current_pos_++;
        }

        bool operator !=(const iterator &it_in) {
          return ptr_ != it_in.ptr() ||
                 current_pos_ != it_in.current_pos() ||
                 num_elements_ != it_in.num_elements();
        }
    };

  public:

    iterator get_iterator() const {
      return iterator(vector_.get(), &position_to_id_map_, 0, num_elements_);
    }

    unsigned int size() const {
      return num_elements_;
    }

    MapVector() {
      num_elements_ = 0;
      indices_to_save_.resize(0);
    }

    template<typename TreeIteratorType>
    void set_indices_to_save(TreeIteratorType &it) {
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

      // The pointer to the modifiable self.
      core::parallel::MapVector<T> *modifiable_self =
        const_cast< core::parallel::MapVector<T> * >(this);

      // Save the number of elements being saved.
      unsigned int num_elements_to_save = indices_to_save_.size();
      ar & num_elements_to_save;

      // Save each element and its mapping.
      for(unsigned int i = 0; i < indices_to_save_.size(); i++) {
        ar & vector_[ indices_to_save_[i] ];
        ar & indices_to_save_[i];
      }

      // Clear the indices to be saved after done.
      modifiable_self->indices_to_save_.resize(0);
    }

    /** @brief Load the object.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the number of elements.
      ar & num_elements_;

      // Load each element and its mapping.
      for(int i = 0; i < num_elements_; i++) {
        T element;
        int original_index;
        ar & element;
        vector_.push_back(element);
        ar & original_index;
        id_to_position_map_[ original_index ] = i;
        position_to_id_map_[ i ] = original_index;
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    void Init(int num_elements_in) {
      boost::scoped_array<T> tmp_vector(new T[num_elements_in]);
      vector_.swap(tmp_vector);
      num_elements_ = num_elements_in;
    }

    const T &operator[](int i) const {
      if(id_to_position_map_.size() == 0) {
        return vector_[i];
      }
      else {
        return vector_[
                 const_cast<
                 core::parallel::MapVector<T> * >(this)->id_to_position_map_[i] ];
      }
    }

    T &operator[](int i) {
      if(id_to_position_map_.size() == 0) {
        return vector_[i];
      }
      else {
        return vector_[ id_to_position_map_[i] ];
      }
    }
};
}
}

#endif
