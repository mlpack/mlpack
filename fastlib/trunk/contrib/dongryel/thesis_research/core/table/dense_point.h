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
#include "memory_mapped_file.h"

namespace core {
namespace table {

extern core::table::MemoryMappedFile *global_m_file_;

class DensePoint {

  private:

    friend class boost::serialization::access;

    boost::interprocess::offset_ptr<double> ptr_;

    int n_rows_;

    bool is_alias_;

  private:
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

    int length() const {
      return n_rows_;
    }

    const double *ptr() const {
      return ptr_.get();
    }

    double *ptr() {
      return ptr_.get();
    }

    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // First the length of the point.
      ar & n_rows_;
      for(int i = 0; i < n_rows_; i++) {
        double element = ptr_[i];
        ar & element;
      }
    }

    template<class Archive>
    void load(Archive &ar, const unsigned int version) {
      // Load the length of the point.
      ar & n_rows_;

      // Allocate the point.
      if(ptr_ == NULL) {
        ptr_ = (core::table::global_m_file_) ?
               (double *)
               core::table::global_m_file_->ConstructArray<double>(n_rows_) :
               new double[n_rows_];
      }
      for(int i = 0; i < n_rows_; i++) {
        ar & (ptr_[i]);
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    void Reset() {
      ptr_ = NULL;
      is_alias_ = false;
    }

    DensePoint() {
      Reset();
    }

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
        (double *)
        core::table::global_m_file_->ConstructArray<double>(length_in) :
        new double[length_in];
      n_rows_ = length_in;
      is_alias_ = false;
    }

    void Init(const std::vector<double> &vector_in) {
      ptr_ =
        (core::table::global_m_file_) ?
        (double *)
        core::table::global_m_file_->ConstructArray<double>(
          vector_in.size()) :
        new double[vector_in.size()];
      n_rows_ = vector_in.size();
      for(unsigned int i = 0; i < vector_in.size(); i++) {
        ptr_[i] = vector_in[i];
      }
      is_alias_ = false;
    }

    void CopyValues(const DensePoint &point_in) {
      memcpy(
        ptr_.get(), point_in.ptr(), sizeof(double) * point_in.length());
      n_rows_ = point_in.length();
      is_alias_ = false;
    }

    void Copy(const DensePoint &point_in) {
      ptr_ =
        (core::table::global_m_file_) ?
        (double *) core::table::global_m_file_->ConstructArray<double>(
          point_in.length()) :
        new double[point_in.length()];
      CopyValues(point_in);
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

    void Add(
      double scale_factor, const core::table::DensePoint &point_in) {
      for(int i = 0; i < point_in.length(); i++) {
        ptr_.get()[i] += scale_factor * point_in[i];
      }
    }

    void operator+=(const core::table::DensePoint &point_in) {
      for(int i = 0; i < point_in.length(); i++) {
        ptr_.get()[i] += point_in[i];
      }
    }

    void operator/=(double scale_factor) {
      for(int i = 0; i < n_rows_; i++) {
        ptr_.get()[i] /= scale_factor;
      }
    }

    void operator*=(double scale_factor) {
      for(int i = 0; i < n_rows_; i++) {
        ptr_.get()[i] *= scale_factor;
      }
    }

    void Print() const {
      printf("Vector of length: %d\n", n_rows_);
      for(int i = 0; i < n_rows_; i++) {
        printf("%g ", ptr_.get()[i]);
      }
      printf("\n");
    }
};
};
};

#endif
