/** @file dense_point.h
 *
 *  An implementation of dense points.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TABLE_DENSE_POINT_H
#define CORE_TABLE_DENSE_POINT_H

#include <armadillo>
#include "core/table/abstract_point.h"

namespace core {
namespace table {
class DenseConstPoint: public core::table::AbstractPoint {
  protected:

    arma::vec *ptr_;

  public:

    virtual ~DenseConstPoint() {
      delete ptr_;
      ptr_ = NULL;
    }

    const arma::vec &reference() const {
      return *ptr_;
    }

    DenseConstPoint() {
      ptr_ = NULL;
    }

    int length() const {
      return ptr_->n_elem;
    }

    double operator[](int i) const {
      return (*ptr_)[i];
    }

    void Alias(const double *ptr_in, int length_in) {
      if(ptr_ != NULL) {
        delete ptr_;
      }
      ptr_ = new arma::vec(ptr_in, length_in);
    }

    void Alias(const DenseConstPoint &point_in) {
      if(ptr_ != NULL) {
        delete ptr_;
      }
      ptr_ = new arma::vec(point_in.reference().memptr(), point_in.length());
    }
};

class DensePoint: public DenseConstPoint {
  public:

    virtual ~DensePoint() {
    }

    arma::vec &reference() {
      return *ptr_;
    }

    double &operator[](int i) {
      return (* DenseConstPoint::ptr_)[i];
    }

    void Init(int length_in) {
      if(ptr_ != NULL) {
        delete ptr_;
      }
      DenseConstPoint::ptr_ = new arma::vec();
      DenseConstPoint::ptr_->set_size(length_in);
    }

    void Copy(DensePoint &point_in) {
      if(ptr_ != NULL) {
        delete ptr_;
      }
      DenseConstPoint::ptr_ = new arma::vec(
        point_in.reference().memptr(), point_in.length(), true);
    }

    void SetZero() {
      ptr_->fill(0.0);
    }

    void Alias(double *ptr_in, int length_in) {
      if(ptr_ != NULL) {
        delete ptr_;
      }
      ptr_ = new arma::vec(ptr_in, length_in, false);
    }

    void operator+=(const arma::vec &point_in) {
      (*ptr_) += point_in;
    }

    void operator/=(double scale_factor) {
      (*ptr_) /= scale_factor;
    }
};
};
};

#endif
