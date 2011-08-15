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

    std::vector<int> indices_to_save_;

    unsigned int num_elements_;

    boost::scoped_array<T> vector_;

  public:

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
        id_to_position_map_[ original_index ] = i;
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
