/** @file map_vector.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_PARALLEL_MAP_VECTOR_H
#define CORE_PARALLEL_MAP_VECTOR_H

#include <boost/intrusive_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <boost/serialization/serialization.hpp>
#include <map>

namespace boost {
template<typename T>
class MapVectorInternal {
  private:

    int num_elements_;

    std::map<int, int> id_to_position_map_;

    std::map<int, int> position_to_id_map_;

    boost::scoped_array<T> vector_;

  public:
    long num_references_;

  public:

    std::map<int, int> &id_to_position_map() {
      return id_to_position_map_;
    }

    std::map<int, int> &position_to_id_map() {
      return position_to_id_map_;
    }

    T *vector() {
      return vector_.get();
    }

    int size() const {
      return num_elements_;
    }

    MapVectorInternal() {
      num_elements_ = 0;
      num_references_ = 0;
    }

    void Init(int num_elements_in) {
      num_elements_ = num_elements_in;
      boost::scoped_array<T> tmp_vector(new T[num_elements_]);
      vector_.swap(tmp_vector);
    }

    /** @brief Returns a const reference to the object with the given
     *         ID.
     */
    const T &operator[](int i) const {
      int translated_index = i;
      if(id_to_position_map_.size() > 0) {
        translated_index =
          id_to_position_map_.find(translated_index)->second;
      }
      return  vector_[ translated_index ];
    }

    /** @brief Returns a modifiable reference to the object with the
     *         given ID.
     */
    T &operator[](int i) {
      int translated_index = i;
      if(id_to_position_map_.size() > 0) {
        translated_index =
          id_to_position_map_.find(translated_index)->second;
      }
      return  vector_[ translated_index ];
    }
};
}

namespace core {
namespace parallel {

template<typename T>
class MapVector {
  private:

    std::vector<int> indices_to_save_;

    boost::intrusive_ptr< boost::MapVectorInternal<T> > internal_;

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
          return position_to_id_map_->find(current_pos_)->second;
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

    const boost::intrusive_ptr< boost::MapVectorInternal<T> > &internal() const {
      return internal_;
    }

    void Alias(const core::parallel::MapVector<T> &source_in) {
      internal_ = source_in.internal();
    }

    template<typename TreeIteratorType>
    void Alias(
      const core::parallel::MapVector<T> &source_in,
      TreeIteratorType &it) {
      internal_ = source_in.internal();
      this->set_indices_to_save(it);
    }

    /** @brief Copies another map vector onto this vector. Assumes
     *         that the mappings of the source and the destination are
     *         the same.
     */
    void Copy(const core::parallel::MapVector<T> &source_in) {
      iterator source_it = source_in.get_iterator();
      while(source_it.HasNext()) {
        this->operator[](source_it.current_id()) = *source_it;
        source_it++;
      }
    }

    /** @brief Gets an iterator of the current map vector object.
     */
    iterator get_iterator() const {
      if(internal_ != NULL) {
        return iterator(
                 internal_->vector(),
                 &(internal_->position_to_id_map()), 0, internal_->size());
      }
      else {
        return iterator();
      }
    }

    /** @brief Returns the number of elements stored in the map
     *         vector.
     */
    int size() const {
      return internal_->size();
    }

    /** @brief The default constructor.
     */
    MapVector() {
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
      it.Reset();
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
        ar & (this->operator[](translated_position));
        ar & translated_position;
      }

      // Clear the indices to be saved after done.
      modifiable_self->indices_to_save_.resize(0);
    }

    /** @brief Load the object.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the number of elements.
      int num_elements;
      ar & num_elements;
      this->Init(num_elements);
      std::map<int, int> &id_to_position_map =
        internal_->id_to_position_map();
      std::map<int, int> &position_to_id_map =
        internal_->position_to_id_map();

      // Load each element and its mapping.
      for(int i = 0; i < num_elements; i++) {
        int original_index;
        T tmp_val;
        ar & tmp_val;
        ar & original_index;
        id_to_position_map[ original_index ] = i;
        position_to_id_map[ i ] = original_index;
        this->operator[](original_index) = tmp_val;
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /** @brief Initializes the map vector for a given number of
     *         elements.
     */
    void Init(int num_elements_in) {
      boost::intrusive_ptr <
      boost::MapVectorInternal<T> > tmp_internal(
        new boost::MapVectorInternal<T>());
      internal_.swap(tmp_internal);
      internal_->Init(num_elements_in);
    }

    /** @brief Returns a const reference to the object with the given
     *         ID.
     */
    const T &operator[](int i) const {
      return internal_->operator[](i);
    }

    /** @brief Returns a modifiable reference to the object with the
     *         given ID.
     */
    T &operator[](int i) {
      return internal_->operator[](i);
    }
};
}
}

namespace boost {
template<typename T>
inline void intrusive_ptr_add_ref(boost::MapVectorInternal<T> *ptr) {
  ptr->num_references_++;
}

template<typename T>
inline void intrusive_ptr_release(boost::MapVectorInternal<T> *ptr) {
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
