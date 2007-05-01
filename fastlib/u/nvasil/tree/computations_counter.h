/*
 * =====================================================================================
 * 
 *       Filename:  computations_counter.h
 * 
 *      Description:  This class keeps track of the computations 
 *                    distances computed, distances etc
 *
 * 
 *        Version:  1.0
 *        Created:  01/31/2007 04:23:16 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos  Vasiloglou
 *        Company:  Georgia Tech Fastlab ESP-LAB
 * 
 * =====================================================================================
 */
#ifndef COMPUTATIONS_COUNTER_
#define COMPUTATIONS_COUNTER_
#include "fastlib/fastlib.h"

template<bool diagnostic>
class ComputationsCounter { 
 public:
	ComputationsCounter() {
	  comparisons_=0;
		distances_=0;
	}
	void Reset() {
	  comparisons_=0;
		distances_=0;
	}
  void UpdadeComparisons();
	void UpdateDistances();
	uint64 get_comparisons() {
	  return comparisons_;
	}
	uint64 get_distances() {
	  return distances_;
	}
 private:
	uint64 comparisons_;
	uint64 distances_;
};

template<>
class ComputationsCounter<true> {
 public:
	void UpdateComparisons() {
    comparisons_++;
  }
  void UpdateDistances() {
    distances_++;
  }
	void Reset() {
		comparisons_=0;
		distances_=0;
	}
	uint64 get_comparisons() {
	  return comparisons_;
	}
	uint64 get_distances() {
	  return distances_;
	}
	private:
	uint64 comparisons_;
	uint64 distances_;

};

template<>
class ComputationsCounter<false> {
 public:
	void UpdateComparisons() {

  }
  void UpdateDistances() {

  }
	void Reset(){
	}

};

#endif // COMPUTATIONS_COUNTER_



