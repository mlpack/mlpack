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
 * Sets of contiguous ranges.
 */

#ifndef COL_RANGESET_H
#define COL_RANGESET_H

#include "fastlib/col/arraylist.h"
//#include "arraylist.h"

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
    
    OT_DEF_BASIC(Range) {
      OT_MY_OBJECT(begin);
      OT_MY_OBJECT(end);
    }
  };
  
 private:
  ArrayList<Range> ranges_;
  
  OT_DEF(RangeSet) {
    OT_MY_OBJECT(ranges_);
  }

 public:
  /**
   * Creates an empty set of ranges.
   */
  void Init() {
    ranges_.Init();
  }

  /**
   * Reinitializes to an empty set of ranges.
   */
  void Reset() {
    ranges_.Clear();
  }

  /**
   * Unions this with a particular range.
   *
   * This will automatically merge with any surrounding ranges.
   */
  void Union(const Boundary& begin, const Boundary& end);

  const ArrayList<Range>& ranges() const {
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

};

template<typename TBoundary>
void RangeSet<TBoundary>::Union(
    const Boundary& begin, const Boundary& end) {
  if (unlikely(!(begin < end))) {
    // Merging with empty range?
    return;
  }
  
  // Not really efficient, but easy to follow.
  ArrayList<Range> new_list;
  index_t i;

  new_list.Init();

  i = 0;

  // add everything that strictly precedes the new one to add
  while (i < ranges_.size() && !(begin <= ranges_[i].end)) {
    new_list.PushBackCopy(ranges_[i]);
    i++;
  }

  // merge everything that overlaps
  const Boundary *selected_end = &end;
  const Boundary *selected_begin = &begin;

  while (i < ranges_.size() && end >= ranges_[i].begin) {
    if (ranges_[i].begin < *selected_begin) {
      selected_begin = &ranges_[i].begin;
    }
    if (ranges_[i].end > *selected_end) {
      selected_end = &ranges_[i].end;
    }
    i++;
  }

  Range *new_range = new_list.PushBackRaw();
  new(&new_range->begin)Boundary(*selected_begin);
  new(&new_range->end)Boundary(*selected_end);

  // add everything that comes after
  for (; i < ranges_.size(); i++) {
    new_list.PushBackCopy(ranges_[i]);
  }

  // replace the list
  ranges_.Swap(&new_list);
}

#endif
