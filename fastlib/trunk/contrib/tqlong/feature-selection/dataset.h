/*
 * dataset.h
 *
 *  Created on: Feb 14, 2011
 *      Author: tqlong
 */

#ifndef DATASET_H_
#define DATASET_H_

#include <armadillo>
#include <iostream>
#include <vector>

template <typename X, typename Y>
struct Sample
{
  typedef X      x_type;
  typedef Y      y_type;

  x_type x_;
  y_type y_;

  Sample(const x_type& x, const y_type& y) : x_(x), y_(y) {}
  Sample() {}
};

class DataSet {
public:
  typedef arma::vec              x_type;
  typedef double                 y_type;
  typedef Sample<x_type, y_type> s_type;
  DataSet(const arma::mat &X, const arma::vec &y);
  DataSet() {}
  virtual ~DataSet();

  void load(const char* fileName);
  void load(std::istream& stream);
  void save(const char* fileName, arma::file_type type = arma::arma_ascii);
  void save(std::ostream& stream, arma::file_type type = arma::arma_ascii);
  inline y_type y(int idx) const { return y_[idx]; }
  inline x_type x(int idx) const { return X_.col(idx); }
  inline s_type s(int idx) const { return s_type(x(idx), y(idx)); }
  inline int dim() const { return X_.n_rows; }
  inline int n() const { return X_.n_cols; }
protected:
  arma::mat X_;
  arma::vec y_;
};

#endif /* DATASET_H_ */
