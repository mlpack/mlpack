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
 * @file rangeset.h
 *
 * Sets of contiguous ranges with no overlap.
 */

#ifndef COL_RANGESET_H
#define COL_RANGESET_H

#include "../base/base.h"

#include <algorithm>
#include <iostream>
#include <vector>

/**
 * A set containing a union of  [start,end) ranges that are automatically
 * sorted and merged when possible.
 *
 * The copy constructor and inequality operators must be defined of the
 * template class.
 *
 * This is O(N) insertion, so not efficient for large sets.
 */
template<typename TBoundary>
class RangeSet {
 public:
  typedef TBoundary Boundary;

 public: 
  struct Range {
    Boundary begin;
    Boundary end;
  };
  
 private:
  std::vector<Range> ranges_;
  
 public:
  /**
   * Constructor that allows you to reserve enough space for your elements.
   */
  RangeSet(const unsigned int size=0) {
    ranges_.reserve(size);
  }

  /**
   * Reinitializes to an empty set of ranges.
   */
  void Reset() {
    ranges_.resize(0);
  }

  /**
   * Unions this with a particular range.
   *
   * This will automatically merge with any surrounding ranges.
   */
  void Union(const Boundary& begin, const Boundary& end);

  const std::vector<Range>& ranges() const {
    return ranges_;
  }

  /**
   * Gets a constant range element.
   */
  const Range& operator[] (index_t i) const {
    return ranges_[i];
  }

  /**
   * Gets the number of discrete ranges.
   */  
  index_t size() const {
    return ranges_.size();
  }

  /**
   * Print the RangeSet to stdout.
   *
   * Will not work with any object lacking operator<<.
   */
  void PrintDebug() {
    std::cout << "Range set: " << std::endl;
    for( typename std::vector<Range>::iterator it = ranges_.begin();
	it < ranges_.end();
	++it ) {
      std::cout << it->begin << " " << it->end << "\t";
    }
    std::cout << std::endl;
  }

};

template<typename TBoundary>
void RangeSet<TBoundary>::Union(
    const Boundary& begin, const Boundary& end) {

  // Incorrect range
  if( begin > end )
    return;

  // If its empty, there's no reason to do more than this
  if( ranges_.empty() ) {
    Range new_range;
    new_range.begin = begin;
    new_range.end = end;
    ranges_.push_back(new_range);
    return;
  }

  std::vector<Range> new_list;
  typename std::vector<Range>::iterator i;

  // add everything that strictly precedes the new one to add
  for( i = ranges_.begin();
      i < ranges_.end() && !(begin <= i->end); ++i) {
    new_list.push_back(*i);
  }

  // merge everything that overlaps
  const Boundary *selected_end = &end;
  const Boundary *selected_begin = &begin;

  for( ; i < ranges_.end() && end >= i->begin; ++i ) {
    if( i->begin < *selected_begin )
      selected_begin = &i->begin;
    if( i->end > *selected_end )
      selected_end = &i->end;
  }

  // Make a new Range and add it.
  Range new_range; 
  new_range.begin = *selected_begin;
  new_range.end = *selected_end;
  new_list.push_back(new_range);

  // add everything that comes after
  for (; i < ranges_.end(); ++i) {
    new_list.push_back(*i);
  }

  // replace the list
  ranges_.assign(new_list.begin(),new_list.end());
}

#endif
