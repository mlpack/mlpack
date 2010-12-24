/** @file mean_variance_pair_matrix.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_MONTE_CARLO_MEAN_VARIANCE_PAIR_MATRIX_H
#define CORE_MONTE_CARLO_MEAN_VARIANCE_PAIR_MATRIX_H

#include <armadillo>
#include "core/monte_carlo/mean_variance_pair.h"

namespace core {
namespace table {
extern core::table::MemoryMappedFile *global_m_file_;
};
};

namespace core {
namespace monte_carlo {
class MeanVariancePairVector {
  private:

    int n_elements_;

    core::monte_carlo::MeanVariancePair *ptr_;

  public:
    ~MeanVariancePairVector() {
      if(ptr_ != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(ptr_);
        }
        else {
          delete[] ptr_;
        }
      }
      ptr_ = NULL;
    }

    MeanVariancePairVector() {
      n_elements_ = 0;
      ptr_ = NULL;
    }

    void CopyValues(const core::monte_carlo::MeanVariancePairVector &v) {
      for(int i = 0; i < n_elements_; i++) {
        ptr_[i].CopyValues(v[i]);
      }
    }

    void push_back(const arma::vec &v) {
      for(unsigned int i = 0; i < v.n_elem; i++) {
        ptr_[i].push_back(v[i]);
      }
    }

    void Init(int n_elements_in) {
      n_elements_ = n_elements_in;
      ptr_ = (core::table::global_m_file_) ?
             core::table::global_m_file_->ConstructArray <
             core::monte_carlo::MeanVariancePair > (n_elements_in) :
             new core::monte_carlo::MeanVariancePair[n_elements_];
    }

    void sample_means(arma::vec *point_out) const {
      point_out->set_size(n_elements_);
      for(int i = 0; i < n_elements_; i++) {
        (*point_out)[i] = ptr_[i].sample_mean();
      }
    }

    const core::monte_carlo::MeanVariancePair & operator[](int i) const {
      return ptr_[i];
    }

    core::monte_carlo::MeanVariancePair & operator[](int i) {
      return ptr_[i];
    }
};

class MeanVariancePairMatrix {
  private:

    int n_rows_;

    int n_cols_;

    int n_elements_;

    core::monte_carlo::MeanVariancePair *ptr_;

  public:
    ~MeanVariancePairMatrix() {
      if(ptr_ != NULL) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(ptr_);
        }
        else {
          delete[] ptr_;
        }
      }
      ptr_ = NULL;
    }

    MeanVariancePairMatrix() {
      n_rows_ = 0;
      n_cols_ = 0;
      n_elements_ = 0;
      ptr_ = NULL;
    }

    void push_back(const arma::mat &v) {
      for(unsigned int j = 0; j < v.n_cols; j++) {
        for(unsigned int i = 0; i < v.n_rows; i++) {
          this->get(i, j).push_back(v.at(i, j));
        }
      }
    }

    void sample_means(arma::mat *point_out) const {
      point_out->set_size(n_rows_, n_cols_);
      for(int j = 0; j < n_cols_; j++) {
        for(int i = 0; i < n_rows_; i++) {
          point_out->at(i, j) = this->get(i, j).sample_mean();
        }
      }
    }

    void CopyValues(const core::monte_carlo::MeanVariancePairMatrix &v) {
      for(int j = 0; j < n_cols_; j++) {
        for(int i = 0; i < n_rows_; i++) {
          this->get(i, j).CopyValues(v.get(i, j));
        }
      }
    }

    void Init(int n_rows_in, int n_cols_in) {
      n_rows_ = n_rows_in;
      n_cols_ = n_cols_in;
      n_elements_ = n_rows_ * n_cols_;
      ptr_ = (core::table::global_m_file_) ?
             core::table::global_m_file_->ConstructArray <
             core::monte_carlo::MeanVariancePair > (n_rows_in * n_cols_in) :
             new core::monte_carlo::MeanVariancePair[n_elements_];
    }

    const core::monte_carlo::MeanVariancePair &get(int row, int col) const {
      return ptr_[col * n_rows_ + row];
    }

    core::monte_carlo::MeanVariancePair &get(int row, int col) {
      return ptr_[col * n_rows_ + row];
    }
};
};
};

#endif
