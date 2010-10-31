#ifndef NAIVE_DISTANCE_MATRIX_H
#define NAIVE_DISTANCE_MATRIX_H

#include "anmf.h"

BEGIN_ANMF_NAMESPACE;

class NaiveDistanceMatrix
{
  const Matrix &reference_, &query_;
  std::vector<double> price_;
public:
  NaiveDistanceMatrix(const Matrix &reference, const Matrix &query)
    : reference_(reference), query_(query), price_(query.n_cols(), 0)
  {
    DEBUG_ASSERT(reference.n_rows() == query.n_rows());
  }
  int n_rows() const { return reference_.n_cols(); }
  int n_cols() const { return query_.n_cols(); }
  double get(int i, int j) const
  {
    Vector r_i, q_j;
    reference_.MakeColumnVector(i, &r_i);
    query_.MakeColumnVector(j, &q_j);
    return -sqrt(la::DistanceSqEuclidean(r_i, q_j));
  }
  void setPrice(int j, double price)
  {
    price_[j] = price;
  }
  int getBestAndSecondBest(int bidder, int &best_item, double &best_surplus, double &second_surplus)
  {
    best_surplus = second_surplus = -std::numeric_limits<double>::infinity();
    for (int item = 0; item < query_.n_cols(); item++)
    {
      double surplus = get(bidder, item) - price_[item];
      if (surplus > best_surplus)
      {
        best_item = item;
        second_surplus = best_surplus;
        best_surplus = surplus;
      }
      else if (surplus > second_surplus)
      {
        second_surplus = surplus;
      }
    }
    return 0;
  }
};

END_ANMF_NAMESPACE;

#endif // NAIVE_DISTANCE_MATRIX_H
