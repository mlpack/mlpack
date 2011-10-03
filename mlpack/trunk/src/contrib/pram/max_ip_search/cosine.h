/**
 * @file cosine.h
 * @author Parikshit Ram
 *
 * This file computes the cosine between two vectors 
 * according to the formula: cos <AB = A.B / |A||B|
 *
 */
#ifndef COSINE_H
#define COSINE_H

#include <armadillo>

//namespace mlpack {
// namespace kernel {

/**
 */
//template<int t_pow, bool t_take_root = false>
class Cosine {
 public:
  /***
   * Default constructor does nothing, but is required to satisfy the Kernel
   * policy.
   */
  Cosine() { }

  /**
   * Computes the distance between two points.
   */
  static double Evaluate(const arma::vec& a, 
			 const arma::vec& b) {
    double dot_prod = arma::dot(a, b);
    double cosine = dot_prod 
      / (arma::norm(a, 2) * arma::norm(b, 2));

    return cosine;

  }
};

#endif
