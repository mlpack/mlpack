#ifndef METRIC_H
#define METRIC_H

#include "fastlib/fastlib.h"

namespace mtrc{


  inline double MinSphereDistSq(const DHrectBound<2>& query,
			     const DHrectBound<2>& ref){
     
    DRange a = query.get(1);
    DRange b = ref.get(1);
    
    double t1 = b.lo - a.hi;
    double t2 = a.lo - b.hi;    
    double t = (t1 + fabs(t1)) + (t2 + fabs(t2));
    double theta = max(fabs(a.hi), fabs(a.lo));
    theta = max(max(fabs(b.hi), fabs(b.lo)), theta);

    a = query.get(0);
    b = ref.get(0);
    double p1 = b.lo - a.hi;
    double p2 = a.lo - b.hi;    
    double p = (p1 + fabs(p1)) + (p2 + fabs(p2));   
   
    p = p*sin(theta);
    return (t*t + p*p) / 4;
  }

 inline double MinSphereDistSq(const DHrectBound<2>& ref, const Vector& query){
     
    DRange b = ref.get(1);
    
    double t1 = b.lo - query[1];
    double t2 = query[1] - b.hi;    
    double t = (t1 + fabs(t1)) + (t2 + fabs(t2));
    double theta = fabs(query[1]);
    theta = max(max(fabs(b.hi), fabs(b.lo)), theta);

    b = ref.get(0);
    double p1 = b.lo - query[0];
    double p2 = query[0] - b.hi;    
    double p = (p1 + fabs(p1)) + (p2 + fabs(p2));   

    p = p*sin(theta);
    return (t*t + p*p) / 4;
  }

  inline double MinRedShiftDistSq(const DHrectBound<2>& query,
			     const DHrectBound<2>& ref){
     
    DRange a = query.get(2);
    DRange b = ref.get(2);
    
    double z1 = b.lo - a.hi;
    double z2 = a.lo - b.hi;
    if ((z1+ fabs(z1)) + (z2 +fabs(z2)) > 0.4 ){
      return BIG_BAD_NUMBER;
    } else {
      return MinSphereDistSq(query, ref);
    }
  }

 inline double MinRedShiftDistSq(const DHrectBound<2>& ref, 
				 const Vector& query){
     
   DRange b = ref.get(2);
   
   double z1 = b.lo - query[2];
   double z2 = query[2] - b.hi;
   if ((z1+ fabs(z1)) + (z2 +fabs(z2)) > 0.4 ){
     return BIG_BAD_NUMBER;
   } else {
     return MinSphereDistSq(ref, query);
   }
 }

  inline double SphereDistSq(const Vector& query, const Vector& ref){
    double dist =  acos(sin(query[1])*sin(ref[1]) + cos(ref[1])*cos(query[1])*
			cos(ref[0]-query[0]));
    return dist*dist;
  }

  inline double RedShiftDistSq(const Vector& query, const Vector& ref){
    if (fabs(query[2] - ref[2]) > 0.2){
      return BIG_BAD_NUMBER;
    } else {
      return SphereDistSq(query, ref);
    }
  }

}

#endif
