/** @file dense_point.h
 *
 *  A file containing utility functions for dense points.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DENSE_POINT_H
#define CORE_TABLE_DENSE_POINT_H

#include <armadillo>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/tracking_enum.hpp>
#include <vector>
#include "memory_mapped_file.h"

namespace core {
namespace table {

extern core::table::MemoryMappedFile *global_m_file_;

/** @brief The function that takes a raw pointer and creates an alias
 *         armadillo vector.
 */
template<typename T>
void PtrToArmaVec(
  const T *point_in, int length, arma::Col<T> *vec_out) {

  // This constructor uses the const_cast for a hack. For some reason,
  // Armadillo library does not allow creation of aliases for const
  // pointers, so I used const_cast here.
  const_cast<arma::u32 &>(vec_out->n_rows) = length;
  const_cast<arma::u32 &>(vec_out->n_cols) = 1;
  const_cast<arma::u32 &>(vec_out->n_elem) = length;
  const_cast<arma::u16 &>(vec_out->vec_state) = 1;
  const_cast<arma::u16 &>(vec_out->mem_state) = 2;
  const_cast<T *&>(vec_out->mem) = const_cast<T *>(point_in);
}

template<typename T>
void Alias(const arma::Col<T> &vec_in, arma::Col<T> *alias_out) {
  PtrToArmaVec(vec_in.memptr(), vec_in.n_elem, alias_out);
}
}
}

namespace boost {
namespace serialization {

template<class Archive, class T1, class T2>
void serialize(Archive &ar, std::pair<T1, T2> &pair, unsigned int version) {
  ar & pair.first;
  ar & pair.second;
}

template<class Archive, class T>
inline void serialize(
  Archive & ar,
  arma::Col<T> & t,
  const unsigned int file_version) {
  split_free(ar, t, file_version);
}

template<class Archive, class T>
void save(Archive & ar, const arma::Col<T> & t, unsigned int version) {

  // First save the dimensions.
  ar & t.n_elem;
  const T *ptr = t.memptr();
  for(unsigned int i = 0; i < t.n_elem; i++, ptr++) {
    ar & (*ptr);
  }
}

template<class Archive, class T>
void load(Archive & ar, arma::Col<T> & t, unsigned int version) {

  // Load the dimensions.
  int n_elem;
  ar & n_elem;

  // The new memory block.
  t.set_size(n_elem);
  for(int i = 0; i < n_elem; i++) {
    ar & t[i];
  }
}

template<>
template<typename T>
struct tracking_level < arma::Col<T> > {
  typedef mpl::integral_c_tag tag;
  typedef mpl::int_< boost::serialization::track_never > type;
  BOOST_STATIC_CONSTANT(
    int,
    value = tracking_level::type::value
  );
  BOOST_STATIC_ASSERT((
                        mpl::greater <
                        implementation_level< arma::Col<T> >,
                        mpl::int_<primitive_type>
                        >::value
                      ));
};
}
}

#endif
