/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file tree/dballbound_impl.h
 *
 * Bounds that are useful for binary space partitioning trees.
 * Implementation of DBallBound ball bound metric policy class.
 *
 * @experimental
 */

#ifndef TREE_DBALLBOUND_IMPL_H
#define TREE_DBALLBOUND_IMPL_H

#include "lmetric.h"

#include <armadillo>
#include "../base/arma_compat.h"

/**
 * Determines if a point is within the bound.
 */
template<typename TMetric, typename TPoint>
bool DBallBound<TMetric, TPoint>::Contains(const Point& point) const {
  return MidDistance(point) <= radius_;
}

/**
 * Gets the center.
 *
 * Don't really use this directly.  This is only here for consistency
 * with DHrectBound, so it can plug in more directly if a "centroid"
 * is needed.
 */
template<typename TMetric, typename TPoint>
void DBallBound<TMetric, TPoint>::CalculateMidpoint(Point *centroid) const {
  ot::InitCopy(centroid, center_);
}

/**
 * Calculates minimum bound-to-point squared distance.
 */
template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinDistance(const Point& point) const {
  return math::ClampNonNegative(MidDistance(point) - radius_);
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinDistanceSq(const Point& point) const {
  return std::pow(MinDistance(point), 2);
}

/**
 * Calculates minimum bound-to-bound squared distance.
 */
template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinDistance(const DBallBound& other) const {
  double delta = MidDistance(other.center_) - radius_ - other.radius_;
  return math::ClampNonNegative(delta);
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinDistanceSq(const DBallBound& other) const {
  return std::pow(MinDistance(other), 2);
}

/**
 * Computes maximum distance.
 */
template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MaxDistance(const Point& point) const {
  return MidDistance(point) + radius_;
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MaxDistanceSq(const Point& point) const {
  return std::pow(MaxDistance(point), 2);
}

/**
 * Computes maximum distance.
 */
template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MaxDistance(const DBallBound& other) const {
  return MidDistance(other.center_) + radius_ + other.radius_;
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MaxDistanceSq(const DBallBound& other) const {
  return std::pow(MaxDistance(other), 2);
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 *
 * Example: bound1.MinDistanceSq(other) for minimum squared distance.
 */
template<typename TMetric, typename TPoint>
DRange DBallBound<TMetric, TPoint>::RangeDistance(const DBallBound& other) const {
  double delta = MidDistance(other.center_);
  double sumradius = radius_ + other.radius_;
  return DRange(
      math::ClampNonNegative(delta - sumradius),
      delta + sumradius);
}

template<typename TMetric, typename TPoint>
DRange DBallBound<TMetric, TPoint>::RangeDistanceSq(const DBallBound& other) const {
  double delta = MidDistance(other.center_);
  double sumradius = radius_ + other.radius_;
  return DRange(
      std::pow(math::ClampNonNegative(delta - sumradius), 2),
      std::pow(delta + sumradius, 2));
}

/**
 * Calculates closest-to-their-midpoint bounding box distance,
 * i.e. calculates their midpoint and finds the minimum box-to-point
 * distance.
 *
 * Equivalent to:
 * <code>
 * other.CalcMidpoint(&other_midpoint)
 * return MinDistanceSqToPoint(other_midpoint)
 * </code>
 */
template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinToMid(const DBallBound& other) const {
  double delta = MidDistance(other.center_) - radius_;
  return math::ClampNonNegative(delta);
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinToMidSq(const DBallBound& other) const {
  return std::pow(MinToMid(other), 2);
}

/**
 * Computes minimax distance, where the other node is trying to avoid me.
 */
template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinimaxDistance(const DBallBound& other) const {
  double delta = MidDistance(other.center_) + other.radius_ - radius_;
  return math::ClampNonNegative(delta);
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinimaxDistanceSq(const DBallBound& other) const {
  return std::pow(MinimaxDistance(other), 2);
}

/**
 * Calculates midpoint-to-midpoint bounding box distance.
 */
template< >
inline double DBallBound<LMetric<2>, GenVector<double> >::MidDistance(const GenVector<double>& point) const {
  arma::vec tmp1;
  arma::vec tmp2;
  arma_compat::vectorToVec(center_, tmp1);
  arma_compat::vectorToVec(point, tmp2);
  return Metric::Distance(tmp1, tmp2);
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MidDistance(const DBallBound& other) const {
  return MidDistance(other.center_);
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MidDistanceSq(const DBallBound& other) const {
  return std::pow(MidDistance(other), 2);
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MidDistance(const Point& point) const {
  return Metric::Distance(center_, point);
}

#endif

