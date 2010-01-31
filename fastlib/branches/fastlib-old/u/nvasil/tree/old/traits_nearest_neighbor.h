#ifndef TRAITS_NEAREST_NEIGHBOR_H_
#define TRAITS_NEAREST_NEIGHBOR_H_

#include <vector>
#include <utility>
#include <limits>
#include "point.h"

using namespace std;

template<typename T, typename P> class ComparePairs {
 public: 
  bool operator()(const pair<T,P> &p1, const pair<T,P> &p2) const {
    return p1.first < p2.first;
  }
};


template<typename PRECISION, typename IDPRECISION,  typename ALLOCATOR> 
struct TraitsNearestNeighbor {
  static void Prepare(Point<PRECISION, IDPRECISION, ALLOCATOR> &nearest, 
			                int32 range) {
  }	
  static void Prepare(vector<pair<PRECISION, 
			                Point<PRECISION, IDPRECISION, ALLOCATOR> > > &nearest, 
                      int32 range) {
  	Point<PRECISION, IDPRECISION, ALLOCATOR> dummy;
  	dummy.SetNULL();
  	nearest.clear();
  	for(int32 i=nearest.size(); i<range; i++) {
  	  nearest.push_back(make_pair(numeric_limits<PRECISION>::max(), dummy));
  	}  
  }
  static void Prepare(vector<pair<PRECISION, 
			                Point<PRECISION, IDPRECISION, ALLOCATOR> > > &nearest,
                      PRECISION range) {
  
  }
  
  static void Push(Point<PRECISION, IDPRECISION, ALLOCATOR> &nearest, 
                   Point<PRECISION, IDPRECISION, ALLOCATOR> point, PRECISION dist, 
                   PRECISION &distance, int32 range) {
    if (dist < distance) {
      nearest = point;
      distance = dist;
    }
  }
  
  static void Push(vector<pair<PRECISION, Point<PRECISION, IDPRECISION, ALLOCATOR> > > &nearest, 
                   Point<PRECISION, IDPRECISION, ALLOCATOR> point, PRECISION dist, 
                   PRECISION &distance, int32 range) {
    if (dist <= distance ) {                 	
      nearest.push_back(make_pair(dist, point));
    }
  } 
  static void Push(vector<pair<PRECISION, Point<PRECISION, IDPRECISION, ALLOCATOR> > > &nearest, 
                   Point<PRECISION, IDPRECISION, ALLOCATOR> point, PRECISION dist, 
                   PRECISION &distance, PRECISION range) {
    if (dist<=range) {
     nearest.push_back(make_pair(dist, point));
    }
  }
  
  static PRECISION GetTheCurrentMaximum(
      vector<pair<PRECISION, Point<PRECISION, IDPRECISION, ALLOCATOR> > > &nearest,
      int32 range) {
    return nearest.back().first;
  }  
  static PRECISION GetTheCurrentMaximum(vector<pair<PRECISION, 
	    Point<PRECISION, IDPRECISION, ALLOCATOR> > > &nearest,
      PRECISION range) {
    return range;
  }
  
  static void Adjust(Point<PRECISION, IDPRECISION, ALLOCATOR> &nearest, 
			               PRECISION &distance, int32 range) {
  } 
  static void Adjust(vector<pair<PRECISION, 
			               Point<PRECISION, IDPRECISION, ALLOCATOR> > > &nearest,
                     PRECISION &distance, 
                     int32 range) {                   	
  	if (nearest.empty()) { 
  	  return; 
  	}
  	partial_sort(&nearest[0], &nearest[0]+range,  &nearest[nearest.size()], 
  	     ComparePairs<PRECISION, Point<PRECISION, IDPRECISION, ALLOCATOR> >());
  	uint32 size = nearest.size();
  	for(int32 i=0; i<(int32)(size - range); i++) {
  	  nearest.pop_back();
  	}
  	distance = nearest.back().first;
  	
  }
  static void Adjust(vector<pair<PRECISION, Point<PRECISION, IDPRECISION, ALLOCATOR> > > &nearest,
                     PRECISION &distance,
                     PRECISION range) {
    if (nearest.empty()) { 
  	  return; 
  	}
    sort(nearest.begin(), nearest.end(), 
         ComparePairs<PRECISION, Point<PRECISION, IDPRECISION, ALLOCATOR> >());                     	
    distance = range;
  }  
};    
  	                
#endif /*TRAITS_NEAREST_NEIGHBOR_H_*/
