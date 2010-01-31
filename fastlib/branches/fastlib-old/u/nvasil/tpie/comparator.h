// Copyright (c) 2005 Andrew Danner
//
// File: comparator.h
// Author: Andrew Danner
// Created: 28 Jun 2005
//
// Mappings/Wrappers for converting between different comparison types
//
// $Id: comparator.h,v 1.4 2005/07/07 20:39:22 adanner Exp $
//
#ifndef _COMPARATOR_H
#define _COMPARATOR_H

// Get definitions for working with Unix and Windows
#include "u/nvasil/tpie/portability.h"


// In is unlikely that users will need to directly need these classes
// except to maybe upgrade old code with minimal changes. In the future it
// may be better to make TPIE's compare() method match the syntax of the
// STL operator ()

//Convert STL comparison object with operator() to a TPIE comparison
//object with a compare() function. 
template<class T, class STLCMP>
class STL2TPIE_cmp{
  private:
    STLCMP *isLess; //Class with STL comparison operator()

  public:
    STL2TPIE_cmp(STLCMP* _cmp) {isLess=_cmp; }
    //Do not use with applications that test if compare returns +1
    //Because it never does so.
    inline int compare(const T& left, const T& right){
      if( (*isLess)(left, right) ){ return -1; }
      else { return 0; }
    }
};
  
 
//Convert a TPIE comparison object with a compare() function. 
//to STL comparison object with operator() 
template<class T, class TPCMP>
class TPIE2STL_cmp{
  private:
    TPCMP* cmpobj; //Class with TPIE comparison compare()

  public:
    TPIE2STL_cmp(TPCMP* cmp) {cmpobj=cmp;}
    inline bool operator()(const T& left, const T& right) const{
      return (cmpobj->compare(left, right) < 0);
    }
};

//Convert a class with a comparison operator <
//to a TPIE comparison object with a compare() function. 
template<class T>
class op2TPIE_cmp{
  public:
    op2TPIE_cmp(){};
    //Do not use with applications that test if compare returns +1
    //Because it never does so.
    inline int compare(const T& left, const T& right){
      if( left < right ){ return -1; }
      else { return 0; }
    }
};

//Convert a class with a comparison operator <
//to an STL comparison object with a comparison operator ().
//Not implemented here. It is called less in <functional>, part of STL

#endif // _COMPARATOR_H 
