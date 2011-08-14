/** @file map_vector.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_MAP_VECTOR_H
#define CORE_PARALLEL_MAP_VECTOR_H

#include <boost/scoped_array.hpp>
#include <boost/serialization/serialization.hpp>
#include <hash_map>

namespace core {
namespace parallel {

template<typename T>
class MapVector {
  private:

    // For BOOST serialization.
    friend class boost::serialization::access;

  private:
    std::hash_map<int, int> id_to_position_map_;

    boost::scoped_array<T> vector_;

    std::vector<int> indices_to_save_;

  public:

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
        const_cast< core::paralell::MapVector<T> * >(this);

      // Save the number of elements being saved.
      int num_elements_to_save = indices_to_save_.size();
      ar & num_elements_to_save;

      // Clear the indices to be saved after done.
      modifiable_self->indices_to_save_.clear();
    }

    /** @brief Load the object.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    void Init(int num_elements_in) {
      boost::scoped_array<T> tmp_vector(new T[num_elements_in]);
      vector_.swap(tmp_vector);
    }

    const T &operator[](int i) const {
      if(id_to_position_map_.size() == 0) {
        return vector_[i];
      }
      else {
        return vector_[ id_to_position_map_[i] ];
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
