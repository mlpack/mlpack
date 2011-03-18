/** @file dense_point.h
 *
 *  An implementation of dense points.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DENSE_POINT_H
#define CORE_TABLE_DENSE_POINT_H

#include <armadillo>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#include <vector>
#include "memory_mapped_file.h"

namespace core {
namespace table {

extern core::table::MemoryMappedFile *global_m_file_;

/** @brief The trait class for determining the length of a vector-like
 *         object. Currently supports the core::table::DensePoint and
 *         arma::vec class objects.
 */
template<typename PointType>
class LengthTrait {
  public:
    static int length(const PointType &p);
};

template<typename PointType>
class PointerTrait {
  public:
    static const double *ptr(const PointType &p);
};

/** @brief The dense point class.
 */
class DensePoint {

  private:

    // For BOOST serialization.
    friend class boost::serialization::access;

    /** @brief The pointer.
     */
    boost::interprocess::offset_ptr<double> ptr_;

    /** @brief The number of elements stored in the point object.
     */
    int n_rows_;

    /** @brief Whether the underlying memory is being aliased or not
     *         as a part of another memory.
     */
    bool is_alias_;

  private:

    /** @brief The private destructor.
     */
    void DestructPtr_() {
      if(ptr_ != NULL && is_alias_ == false) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(ptr_.get());
        }
        else {
          delete[] ptr_.get();
        }
      }
    }

  public:

    /** @brief Returns whether this point is an alias or not.
     */
    bool is_alias() const {
      return is_alias_;
    }

    /** @brief The assignment operator that steals the pointer and
     *         makes the other point an alias.
     */
    void operator=(const core::table::DensePoint &point_in) {

      ptr_ = const_cast<double *>(point_in.ptr());
      n_rows_ = point_in.length();
      is_alias_ = point_in.is_alias();

      // Steal the pointer and make the point in alias.
      const_cast<core::table::DensePoint &>(point_in).is_alias_ = true;
    }

    /** @brief The assignment operator that steals the pointer and
     *         makes the other point an alias.
     */
    DensePoint(const DensePoint &point_in) {
      this->operator=(point_in);
    }

    /** @brief Returns the length of the point.
     */
    int length() const {
      return n_rows_;
    }

    const double *ptr() const {
      return ptr_.get();
    }

    double *ptr() {
      return ptr_.get();
    }

    /** @brief Saves the point.
     */
    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // First the length of the point.
      ar & n_rows_;
      for(int i = 0; i < n_rows_; i++) {
        double element = ptr_[i];
        ar & element;
      }
    }

    /** @brief Loads the point.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the length of the point.
      ar & n_rows_;

      // Allocate the point.
      if(ptr_.get() == NULL && n_rows_ > 0) {
        ptr_ = (core::table::global_m_file_) ?
               core::table::global_m_file_->ConstructArray<double>(n_rows_) :
               new double[n_rows_];
      }
      for(int i = 0; i < n_rows_; i++) {
        ar & (ptr_[i]);
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    void Reset() {
      n_rows_ = 0;
      ptr_ = NULL;
      is_alias_ = false;
    }

    /** @brief The default constructor.
     */
    DensePoint() {
      Reset();
    }

    /** @brief The destructor.
     */
    ~DensePoint() {
      DestructPtr_();
      Reset();
    }

    const double operator[](int i) const {
      return ptr_.get()[i];
    }

    double &operator[](int i) {
      return ptr_.get()[i];
    }

    void Init(int length_in) {
      ptr_ =
        (core::table::global_m_file_) ?
        core::table::global_m_file_->ConstructArray<double>(length_in) :
        new double[length_in];
      n_rows_ = length_in;
      is_alias_ = false;
    }

    template<typename PointType>
    void CopyValues(const PointType &point_in) {
      const double *point_in_ptr =
        core::table::PointerTrait<PointType>::ptr(point_in);
      n_rows_ = core::table::LengthTrait<PointType>::length(point_in);
      memcpy(
        ptr_.get(), point_in_ptr, sizeof(double) * n_rows_);
      is_alias_ = false;
    }

    template<typename PointType>
    void Copy(const PointType &point_in) {
      if(core::table::LengthTrait<PointType>::length(point_in) > 0) {
        if(ptr_.get() == NULL) {
          int length = core::table::LengthTrait<PointType>::length(point_in);
          ptr_ =
            (core::table::global_m_file_) ?
            core::table::global_m_file_->ConstructArray<double>(
              length) : new double[ length ];
        }
        CopyValues(point_in);
      }
    }

    /** @brief Sets every entry of the matrix to the specified value
     *         val.
     */
    void SetAll(double val) {
      for(int i = 0; i < n_rows_; i++) {
        ptr_[i] = val;
      }
    }

    void SetZero() {
      memset(ptr_.get(), 0, sizeof(double) * n_rows_);
    }

    void Alias(double *ptr_in, int length_in) {
      ptr_ = ptr_in;
      n_rows_ = length_in;
      is_alias_ = true;
    }

    void Alias(const double *ptr_in, int length_in) {
      ptr_ = const_cast<double *>(ptr_in);
      n_rows_ = length_in;
      is_alias_ = true;
    }

    void Alias(const DensePoint &point_in) {
      ptr_ = const_cast<double *>(point_in.ptr());
      n_rows_ = point_in.length();
      is_alias_ = true;
    }

    /** @brief Prints the point.
     */
    void Print() const {
      printf("Vector of length: %d\n", n_rows_);
      for(int i = 0; i < n_rows_; i++) {
        printf("%g ", ptr_.get()[i]);
      }
      printf("\n");
    }
};

/** @brief The trait class instantiation for determining the length of
 *         an arma::vec object.
 */
template<>
class LengthTrait<arma::vec> {
  public:
    static int length(const arma::vec &p) {
      return p.n_elem;
    }
};

/** @brief The trait class instantiation for determining the length of
 *         core::table::DensePoint object.
 */
template<>
class LengthTrait<core::table::DensePoint> {
  public:
    static int length(const core::table::DensePoint &p) {
      return p.length();
    }
};

template<>
class PointerTrait<arma::vec> {
  public:
    static const double *ptr(const arma::vec &p) {
      return p.memptr();
    }
};

template<>
class PointerTrait<core::table::DensePoint> {
  public:
    static const double *ptr(const core::table::DensePoint &p) {
      return p.ptr();
    }
};

/** @brief The function that takes a raw double pointer and creates an
 *         alias armadillo vector.
 */
template<typename T>
static void DoublePtrToArmaVec(
  const double *point_in, T length, arma::vec *vec_out) {

  // This constructor uses the const_cast for a hack. For some reason,
  // Armadillo library does not allow creation of aliases for const
  // double pointers, so I used const_cast here.
  const_cast<arma::u32 &>(vec_out->n_rows) = length;
  const_cast<arma::u32 &>(vec_out->n_cols) = 1;
  const_cast<arma::u32 &>(vec_out->n_elem) = length;
  const_cast<arma::u16 &>(vec_out->vec_state) = 1;
  const_cast<arma::u16 &>(vec_out->mem_state) = 2;
  const_cast<double *&>(vec_out->mem) = const_cast<double *>(point_in);
}

/** @brief The function that takes a raw pointer out from a
 *         pre-existing core::table::DensePoint object and creates an
 *         alias armadillo vector.
 */
template<typename DensePointType>
static void DensePointToArmaVec(
  const DensePointType &point_in, arma::vec *vec_out) {

  core::table::DoublePtrToArmaVec(
    point_in.ptr(), point_in.length(), vec_out);
}
}
}

#endif
