/** @file mean_variance_pair_matrix.h
 *
 *  A mean variance pair class for matrix and vector quanitites.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_MONTE_CARLO_MEAN_VARIANCE_PAIR_MATRIX_H
#define CORE_MONTE_CARLO_MEAN_VARIANCE_PAIR_MATRIX_H

#include <armadillo>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#include "core/monte_carlo/mean_variance_pair.h"
#include "core/table/dense_matrix.h"

namespace core {
namespace table {
extern core::table::MemoryMappedFile *global_m_file_;
}
}

namespace core {
namespace monte_carlo {
class MeanVariancePairVector {
  private:

    // For BOOST serialization.
    friend class boost::serialization::access;

    /** @brief The number of mean variance pair objects.
     */
    int n_elements_;

    /** @brief The list of mean variance pair objects.
     */
    core::monte_carlo::MeanVariancePair *ptr_;

  private:

    void DestructPtr_() {
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

  public:

    /** @brief Saves the mean variance pair vector.
     */
    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // First the length of the vector, then save each element.
      ar & n_elements_;
      for(int i = 0; i < n_elements_; i++) {
        ar & ptr_[i];
      }
    }

    /** @brief Loads the vector of mean variance pairs.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the length.
      ar & n_elements_;

      // Allocate the vector.
      if(ptr_ == NULL && n_elements_ > 0) {
        this->Init(n_elements_);
      }
      for(int i = 0; i < n_elements_; i++) {
        ar & (ptr_[i]);
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /** @brief Scales each component of the mean variance pair
     *         vector by a common scalar.
     */
    void scale(double scale_factor_in) {
      for(int j = 0; j < n_elements_; j++) {
        ptr_[j].scale(scale_factor_in);
      }
    }

    /** @brief Pushes another mean variance pair vector information
     *         with a scale factor, componentwise. Assumes that the
     *         addition is done in an asymptotic normal way.
     */
    void ScaledCombineWith(
      double scale_in, const core::monte_carlo::MeanVariancePairVector &v) {

      for(int j = 0; j < n_elements_; j++) {
        core::monte_carlo::MeanVariancePair scaled_v;
        scaled_v.CopyValues(v[j]);
        scaled_v.scale(scale_in);
        ptr_[j].CombineWith(scaled_v);
      }
    }

    /** @brief Returns the length of the vector.
     */
    int length() const {
      return n_elements_;
    }

    /** @brief The destructor.
     */
    ~MeanVariancePairVector() {
      this->DestructPtr_();
    }

    /** @brief The default constructor.
     */
    MeanVariancePairVector() {
      n_elements_ = 0;
      ptr_ = NULL;
    }

    /** @brief Resets everything in the vector to empty set of
     *         samples.
     */
    void SetZero() {
      for(int j = 0; j < n_elements_; j++) {
        ptr_[j].SetZero();
      }
    }

    /** @brief Sets everything to zero and sets the total number of
     *         terms represented by this object to a given number.
     */
    void SetZero(int total_num_terms_in) {
      for(int i = 0; i < n_elements_; i++) {
        ptr_[i].SetZero(total_num_terms_in);
      }
    }

    void CopyValues(const core::monte_carlo::MeanVariancePairVector &v) {
      for(int i = 0; i < n_elements_; i++) {
        ptr_[i].CopyValues(v[i]);
      }
    }

    void Copy(const core::monte_carlo::MeanVariancePairVector &v) {
      this->Init(v.length());
      this->CopyValues(v);
    }

    void operator=(const MeanVariancePairVector &v) {
      if(ptr_ == NULL || n_elements_ != v.length()) {
        if(ptr_ != NULL) {
          DestructPtr_();
        }
        this->Init(v.length());
      }
      n_elements_ = v.length();
      this->CopyValues(v);
    }

    MeanVariancePairVector(const MeanVariancePairVector &v) {
      this->operator=(v);
    }

    void push_back(const arma::vec &v) {
      for(unsigned int i = 0; i < v.n_elem; i++) {
        ptr_[i].push_back(v[i]);
      }
    }

    void CombineWith(const core::monte_carlo::MeanVariancePairVector &v) {
      for(int i = 0; i < n_elements_; i++) {
        ptr_[i].CombineWith(v[i]);
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

    double max_scaled_deviation(
      double scale_in, double num_standard_deviations) const {

      double max_scaled_dev = 0.0;
      for(int i = 0; i < n_elements_; i++) {
        max_scaled_dev =
          std::max(
            max_scaled_dev,
            ptr_[i].scaled_deviation(
              scale_in, num_standard_deviations));
      }
      return max_scaled_dev;
    }

    /** @brief Sets the total number of terms.
     */
    void set_total_num_terms(int total_num_terms_in) {
      for(int i = 0; i < n_elements_; i++) {
        ptr_[i].set_total_num_terms(total_num_terms_in);
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

    // For BOOST serialization.
    friend class boost::serialization::access;

    /** @brief The number of rows.
     */
    int n_rows_;

    /** @brief The number of columns.
     */
    int n_cols_;

    /** @brief The number of elements.
     */
    int n_elements_;

    /** @brief The underlying matrix of mean variance pair objects.
     */
    core::monte_carlo::MeanVariancePair *ptr_;

  private:

    /** @brief Destroys the matrix.
     */
    void DestructPtr_() {
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

  public:

    /** @brief Saves the point.
     */
    template<class Archive>
    void save(Archive &ar, const unsigned int version) const {

      // Save the number of rows, the number of columns, and the
      // number of elements.
      ar & n_rows_;
      ar & n_cols_;
      ar & n_elements_;
      for(int i = 0; i < n_elements_; i++) {
        ar & ptr_[i];
      }
    }

    /** @brief Loads the point.
     */
    template<class Archive>
    void load(Archive &ar, const unsigned int version) {

      // Load the number of rows, the number of columns and the number
      // of elements.
      ar & n_rows_;
      ar & n_cols_;
      ar & n_elements_;

      // Allocate the vector.
      if(ptr_ == NULL && n_elements_ > 0) {
        this->Init(n_rows_, n_cols_);
      }
      for(int i = 0; i < n_elements_; i++) {
        ar & (ptr_[i]);
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    /** @brief The destructor.
     */
    ~MeanVariancePairMatrix() {
      DestructPtr_();
    }

    /** @brief The constructor.
     */
    MeanVariancePairMatrix() {
      n_rows_ = 0;
      n_cols_ = 0;
      n_elements_ = 0;
      ptr_ = NULL;
    }

    /** @brief Resets every component of the mean variance pair matrix
     *         to empty state.
     */
    void SetZero() {
      for(int i = 0; i < n_elements_; i++) {
        ptr_[i].SetZero();
      }
    }

    /** @brief Sets everything to zero and sets the total number of
     *         terms represented by this object to a given number.
     */
    void SetZero(int total_num_terms_in) {
      for(int i = 0; i < n_elements_; i++) {
        ptr_[i].SetZero(total_num_terms_in);
      }
    }

    void operator=(const MeanVariancePairMatrix &v) {
      if(ptr_ == NULL || n_rows_ != v.n_rows() ||
          n_cols_ != v.n_cols()) {
        if(ptr_ != NULL) {
          DestructPtr_();
        }
        if(v.n_rows() > 0 && v.n_cols()) {
          this->Init(v.n_rows(), v.n_cols());
        }
      }
      n_rows_ = v.n_rows();
      n_cols_ = v.n_cols();
      n_elements_ = n_rows_ * n_cols_;
      this->CopyValues(v);
    }

    MeanVariancePairMatrix(const MeanVariancePairMatrix &v) {
      this->operator=(v);
    }

    void push_back(const arma::mat &v) {
      for(unsigned int j = 0; j < v.n_cols; j++) {
        for(unsigned int i = 0; i < v.n_rows; i++) {
          this->get(i, j).push_back(v.at(i, j));
        }
      }
    }

    /** @brief Pushes another mean variance pair matrix information
     *         with a scale factor, componentwise. Assumes that the
     *         addition is done in an asymptotic normal way.
     */
    void ScaledCombineWith(
      double scale_in, const core::monte_carlo::MeanVariancePairMatrix &v) {

      for(int j = 0; j < n_cols_; j++) {
        for(int i = 0; i < n_rows_; i++) {
          core::monte_carlo::MeanVariancePair scaled_v;
          scaled_v.CopyValues(v.get(i, j));
          scaled_v.scale(scale_in);
          this->get(i, j).CombineWith(scaled_v);
        }
      }
    }

    /** @brief Returns the sample mean variances in a matrix form.
     */
    void sample_mean_variances(arma::mat *point_out) const {
      point_out->set_size(n_rows_, n_cols_);
      for(int j = 0; j < n_cols_; j++) {
        for(int i = 0; i < n_rows_; i++) {
          point_out->at(i, j) = this->get(i, j).sample_mean_variance();
        }
      }
    }

    double max_scaled_deviation(
      double scale_in, double num_standard_deviations) const {

      double max_scaled_dev = 0.0;
      for(int j = 0; j < n_cols_; j++) {
        for(int i = 0; i < n_rows_; i++) {
          max_scaled_dev =
            std::max(
              max_scaled_dev,
              this->get(i, j).scaled_deviation(
                scale_in, num_standard_deviations));
        }
      }
      return max_scaled_dev;
    }

    /** @brief Returns the sample means in a matrix form.
     */
    void sample_means(arma::mat *point_out) const {
      point_out->set_size(n_rows_, n_cols_);
      for(int j = 0; j < n_cols_; j++) {
        for(int i = 0; i < n_rows_; i++) {
          point_out->at(i, j) = this->get(i, j).sample_mean();
        }
      }
    }

    /** @brief Copies mean variance pair objects from another mean
     *         variance pair matrix object.
     */
    void CopyValues(const core::monte_carlo::MeanVariancePairMatrix &v) {
      for(int j = 0; j < n_cols_; j++) {
        for(int i = 0; i < n_rows_; i++) {
          this->get(i, j).CopyValues(v.get(i, j));
        }
      }
    }

    void Copy(const core::monte_carlo::MeanVariancePairMatrix &v) {
      this->Init(v.n_rows(), v.n_cols());
      this->CopyValues(v);
    }

    /** @brief Initializes the mean variance pair matrix with the
     *         given dimensionality.
     */
    void Init(int n_rows_in, int n_cols_in) {
      n_rows_ = n_rows_in;
      n_cols_ = n_cols_in;
      n_elements_ = n_rows_ * n_cols_;
      ptr_ = (core::table::global_m_file_) ?
             core::table::global_m_file_->ConstructArray <
             core::monte_carlo::MeanVariancePair > (n_rows_in * n_cols_in) :
             new core::monte_carlo::MeanVariancePair[n_elements_];
    }

    /** @brief Sets the total number of terms.
     */
    void set_total_num_terms(int total_num_terms_in) {
      for(int i = 0; i < n_elements_; i++) {
        ptr_[i].set_total_num_terms(total_num_terms_in);
      }
    }

    /** @brief Returns the mean variance pair object at (row, col)
     *         position.
     */
    const core::monte_carlo::MeanVariancePair &get(int row, int col) const {
      return ptr_[col * n_rows_ + row];
    }

    /** @brief Returns the mean variance pair object at (row, col)
     *         position.
     */
    core::monte_carlo::MeanVariancePair &get(int row, int col) {
      return ptr_[col * n_rows_ + row];
    }

    void CombineWith(const core::monte_carlo::MeanVariancePairMatrix &v) {
      for(int j = 0; j < n_cols_; j++) {
        for(int i = 0; i < n_rows_; i++) {
          this->get(i, j).CombineWith(v.get(i, j));
        }
      }
    }

    /** @brief Scales the mean variance pair matrix by a common
     *         scalar.
     */
    void scale(double scale_factor_in) {
      for(int j = 0; j < n_cols_; j++) {
        for(int i = 0; i < n_rows_; i++) {
          this->get(i, j).scale(scale_factor_in);
        }
      }
    }

    /** @brief Returns the number of rows.
     */
    int n_rows() const {
      return n_rows_;
    }

    /** @brief Returns the number of columns.
     */
    int n_cols() const {
      return n_cols_;
    }
};
}
}

#endif
