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

        std::map<int, int> *position_to_id_map() {
          return position_to_id_map_;
        }

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
          position_to_id_map_ =
            const_cast<iterator &>(it_in).position_to_id_map();
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

    /** @brief Copies another map vector onto this vector. Assumes
     *         that the mappings of the source and the destination are
     *         the same.
     */
    void Copy(const core::parallel::MapVector<T> &source_in) {
      iterator source_it = source_in.get_iterator();
      while(source_it.HasNext()) {
        vector_[ source_it.current_id()] = *source_it;
        source_it++;
      }
    }

    /** @brief Gets an iterator of the current map vector object.
     */
    iterator get_iterator() const {
      return iterator(vector_.get(), &position_to_id_map_, 0, num_elements_);
    }

    /** @brief Returns the number of elements stored in the map
     *         vector.
     */
    unsigned int size() const {
      return num_elements_;
    }

    /** @brief The default constructor.
     */
    MapVector() {
      num_elements_ = 0;
      indices_to_save_.resize(0);
    }

    /** @brief Call this function before serializing a subset of
     *         vector.
     */
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
        int translated_position = indices_to_save_[i];
        if(id_to_position_map_.size() > 0) {
          translated_position =
            modifiable_self->id_to_position_map_[ translated_position ];
        }
        ar & vector_[ translated_position ];
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
      boost::scoped_array<T> tmp_array(new T[ num_elements_]);
      vector_.swap(tmp_array);

      // Load each element and its mapping.
      for(unsigned int i = 0; i < num_elements_; i++) {
        int original_index;
        ar & vector_[i];
        ar & original_index;
        id_to_position_map_[ original_index ] = i;
        position_to_id_map_[ i ] = original_index;
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /** @brief Initializes the map vector for a given number of
     *         elements.
     */
    void Init(int num_elements_in) {
      boost::scoped_array<T> tmp_vector(new T[num_elements_in]);
      vector_.swap(tmp_vector);
      num_elements_ = num_elements_in;
    }

    /** @brief Returns a const reference to the object with the given
     *         ID.
     */
    const T &operator[](int i) const {
      if(id_to_position_map_.size() == 0) {
        return vector_[i];
      }
      else {
        return
          vector_[
            const_cast<
            core::parallel::MapVector<T> * >(this)->id_to_position_map_[i] ];
      }
    }

    /** @brief Returns a modifiable reference to the object with the
     *         given ID.
     */
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
