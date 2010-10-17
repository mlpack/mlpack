/** @file dense_point.h
 *
 *  An implementation of dense points.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DENSE_POINT_H
#define CORE_TABLE_DENSE_POINT_H

#include <armadillo>
#include <boost/serialization/serialization.hpp>
#include "core/table/abstract_point.h"

namespace core {
namespace table {
class DenseConstPoint: public core::table::AbstractPoint {
  protected:

    double *ptr_;

    int n_rows_;

  public:

    const double *ptr() const {
      return ptr_;
    }

    virtual ~DenseConstPoint() {

      // A const point is always defined as an alias to a part of an
      // already-existing memory block, so you do not free it.
      Reset();
    }

    void Reset() {
      ptr_ = NULL;
      n_rows_ = 0;
    }

    DenseConstPoint() {
      Reset();
    }

    int length() const {
      return n_rows_;
    }

    double operator[](int i) const {
      return ptr_[i];
    }

    void Alias(double *ptr_in, int length_in) {
      ptr_ = ptr_in;
      n_rows_ = length_in;
    }

    void Alias(const DenseConstPoint &point_in) {
      ptr_ = const_cast<double *>(point_in.ptr());
      n_rows_ = point_in.length();
    }
};

class DensePoint: public DenseConstPoint {

  private:

    friend class boost::serialization::access;

    bool is_alias_;

  public:

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
      int length;
      ar & length;

      // Allocate the point.
      ptr_ = new double[length];
      for(int i = 0; i < length; i++) {
        ar & (ptr_[i]);
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    void Reset() {
      is_alias_ = false;
    }

    DensePoint() {
      Reset();
    }

    virtual ~DensePoint() {
      if(DenseConstPoint::ptr_ != NULL && is_alias_ == false) {
        delete DenseConstPoint::ptr_;
      }
      DenseConstPoint::Reset();
      Reset();
    }

    double &operator[](int i) {
      return DenseConstPoint::ptr_[i];
    }

    void Init(int length_in) {
      DenseConstPoint::ptr_ = new double[length_in];
      DenseConstPoint::n_rows_ = length_in;
    }

    void Init(const std::vector<double> &vector_in) {
      DenseConstPoint::ptr_ = new double[vector_in.size()];
      DenseConstPoint::n_rows_ = vector_in.size();
      for(unsigned int i = 0; i < vector_in.size(); i++) {
        ptr_[i] = vector_in[i];
      }
    }

    void Copy(const DenseConstPoint &point_in) {
      DenseConstPoint::ptr_ = new double[point_in.length()];
      memcpy(
        DenseConstPoint::ptr_, point_in.ptr(),
        sizeof(double) * point_in.length());
      DenseConstPoint::n_rows_ = point_in.length();
    }

    void SetZero() {
      memset(ptr_, 0, sizeof(double) * n_rows_);
    }

    void Alias(double *ptr_in, int length_in) {
      DenseConstPoint::ptr_ = ptr_in;
      DenseConstPoint::n_rows_ = length_in;
      is_alias_ = true;
    }

    void operator=(const core::table::DenseConstPoint &point_in) {
      memcpy(ptr_, point_in.ptr(), sizeof(double) * point_in.length());
    }

    void operator+=(const core::table::DenseConstPoint &point_in) {
      for(int i = 0; i < point_in.length(); i++) {
        ptr_[i] += point_in[i];
      }
    }

    void operator/=(double scale_factor) {
      for(int i = 0; i < DenseConstPoint::n_rows_; i++) {
        ptr_[i] /= scale_factor;
      }
    }

    void operator*=(double scale_factor) {
      for(int i = 0; i < DenseConstPoint::n_rows_; i++) {
        ptr_[i] *= scale_factor;
      }
    }
};
};
};

#endif
