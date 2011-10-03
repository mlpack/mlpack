/**
 * @file tree/dhrectperiodicbound.h
 *
 * Bounds that are useful for binary space partitioning trees.
 *
 * This file describes the interface for the DHrectPeriodicBound policy, which
 * implements a hyperrectangle bound.
 *
 * @experimental
 */

#ifndef TREE_DHRECTPERIODICBOUND_H
#define TREE_DHRECTPERIODICBOUND_H

#include <armadillo>

/**
 * Hyper-rectangle bound for an L-metric.
 *
 * Template parameter t_pow is the metric to use; use 2 for Euclidean (L2).
 */
template<int t_pow = 2>
class DHrectPeriodicBound {

  public:
    static const int PREFERRED_POWER = t_pow;

  private:
    Range *bounds_;
    size_t dim_;
    arma::vec box_size_;

  public:
    /**
     * Empty constructor.
     */
    DHrectPeriodicBound();

    /**
     *Specifies the box size, but not dimensionality.
     */
    DHrectPeriodicBound(arma::vec box);

    /**
     * Initializes to specified dimensionality with each dimension the empty
     * set.
     */
    DHrectPeriodicBound(size_t dimension, arma::vec box);

    /**
     * Destructor: clean up memory.
     */
    ~DHrectPeriodicBound();

    /**
     * Modifies the box_size_ to the desired dimenstions.
     */
    void SetBoxSize(arma::vec box);

    /**
     * Returns the box_size_ vector.
     */
    arma::vec GetBoxSize();

    /**
     * Makes this (uninitialized) box the average of the two arguments,
     * i.e. the max and min of each range is the average of the maxes and mins
     * of the arguments.
     *
     * Added by: Bill March, 5/7
     */
    void AverageBoxesInit(const DHrectPeriodicBound& box1, const DHrectPeriodicBound& box2);

    /**
     * Resets all dimensions to the empty set.
     */
    void Reset();

    /**
     * Sets the dimensionality of the bound.
     */
    void SetSize(size_t dim);

    /**
     * Determines if a point is within this bound.
     */
    bool Contains(const arma::vec& point) const;

    /** Gets the dimensionality */
    size_t dim() const { return dim_; }

     /**
     * Sets and gets the range for a particular dimension.
     */
    Range& operator[](size_t i);
    const Range operator[](size_t i) const;

    /**
     * Calculates the maximum distance within the rectangle
     */
    double CalculateMaxDistanceSq() const;

    /** Calculates the midpoint of the range */
    void CalculateMidpoint(arma::vec& centroid) const;

    /**
     * Calculates minimum bound-to-point squared distance.
     */
    double MinDistanceSq(const arma::vec& point) const;

    /**
     * Calculates minimum bound-to-bound squared distance.
     *
     * Example: bound1.MinDistanceSq(other) for minimum squared distance.
     */
    double MinDistanceSq(const DHrectPeriodicBound& other) const;

    /**
     * Calculates maximum bound-to-point squared distance.
     */
    double MaxDistanceSq(const arma::vec& point) const;

    /**
     * Calculates maximum bound-to-point squared distance.
     */
    //double MaxDistanceSq(const double *point) const;

    /**
     * Computes maximum distance.
     */
    double MaxDistanceSq(const DHrectPeriodicBound& other) const;

    /**
     * Computes maximum distance with offset
     */
    double MaxDistanceSq(const DHrectPeriodicBound& other, const arma::vec& offset) const;

    double MaxDelta(const DHrectPeriodicBound& other, double box_width, int dim) const;
    double MinDelta(const DHrectPeriodicBound& other, double box_width, int dim) const;

    /**
     * Calculates minimum and maximum bound-to-bound squared distance.
     */
    Range RangeDistanceSq(const DHrectPeriodicBound& other) const;

    /**
     * Calculates minimum and maximum bound-to-point squared distance.
     */
    Range RangeDistanceSq(const arma::vec& point) const;

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
    double MinToMidSq(const DHrectPeriodicBound& other) const;

    /**
     * Computes minimax distance, where the other node is trying to avoid me.
     */
    double MinimaxDistanceSq(const DHrectPeriodicBound& other) const;

    /**
     * Calculates midpoint-to-midpoint bounding box distance.
     */
    double MidDistanceSq(const DHrectPeriodicBound& other) const;

    /**
     * Expands this region to include a new point.
     */
    DHrectPeriodicBound& operator|=(const arma::vec& vector);

    /**
     * Expands this region to encompass another bound.
     */
    DHrectPeriodicBound& operator|=(const DHrectPeriodicBound& other);

    /**
     * Expand this bounding box to encompass another point. Done to
     * minimize added volume in periodic coordinates.
     */
    DHrectPeriodicBound& Add(const arma::vec& other, const arma::vec& size);

    /**
     * Expand this bounding box in periodic coordinates, minimizing added volume
     */
    DHrectPeriodicBound& Add(const DHrectPeriodicBound& other, const arma::vec& size);
};

#include "dhrectperiodicbound_impl.h"

#endif
