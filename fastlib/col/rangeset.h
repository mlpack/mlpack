#ifndef COL_RANGESET_H
#define COL_RANGESET_H

#include "arraylist.h"

/**
 * A set of [begin,end) ranges.
 *
 * The copy constructor and inequality operators must be defined.
 */
template<typename TBoundary>
class RangeSet {
 public:
  typedef TBoundary Boundary;

 public: 
  struct Range {
    Boundary begin;
    Boundary end;
    
    OT_DEF(Range) {
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
  void Init() {
    ranges_.Init();
  }
  
  void Reset() {
    ranges_.Clear();
  }

  void Union(const Boundary& begin, const Boundary& end) {
    // Not really efficient, but easy to follow.
    ArrayList<Range> new_list;
    index_t i;

    new_list.Init();

    i = 0;

    // add everything that strictly precedes the new one to add
    while (i < ranges_.size() && !(begin <= ranges_[i].end)) {
      new_list.AddBack(ranges_[i]);
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

    Range *new_range = new_list.AddBackUnconstructed();
    new(&new_range->begin)Boundary(*selected_begin);
    new(&new_range->end)Boundary(*selected_end);

    // add everything that comes after
    for (; i < ranges_.size(); i++) {
      new_list.AddBack(ranges_[i]);
    }

    // replace the list
    ranges_.Swap(&new_list);
  }
  
  const ArrayList<Range>& ranges() const {
    return ranges_;
  }
};

#endif
