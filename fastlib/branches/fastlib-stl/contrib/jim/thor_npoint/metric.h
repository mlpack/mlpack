#ifndef METRIC_H
#define METRIC_H

#include "fastlib/fastlib.h"

namespace mtrc{

  inline double GetRad(const DHrectBound<2>& ref, Vector& center){
    center.Init(2);
    DRange a = ref.get(0);
    DRange b = ref.get(1);
    center[0] = 0.5*(a.hi + a.lo);
    center[1] = 0.5*(b.hi + b.lo);
    double theta = max(fabs(b.hi), fabs(b.lo));
    double rad = acos(sin(theta)*sin(center[1]) + cos(theta)*cos(center[1])*
		      cos(center[0]-a.lo));
    return rad;

  }

  inline double MinSphereDistSq(const DHrectBound<2>& query,
			     const DHrectBound<2>& ref){
    Vector c1, c2;
    double r1 = GetRad(query, c1);
    double r2 = GetRad(ref, c2);
    double dist = acos(sin(c1[1])*sin(c2[1]) + cos(c1[1])*cos(c2[1])*
		       cos(c1[0] - c2[0]));

    dist = max(dist - r1 - r2, 0.0);
    return dist*dist;
  }

  

 inline double MinSphereDistSq(const DHrectBound<2>& ref, const Vector& query){
     
   Vector cr;
   double r = GetRad(ref, cr);   
   double dist = acos(sin(cr[1])*sin(query[1]) + cos(cr[1])*cos(query[1])*
		       cos(cr[0] - query[0]));
    dist = max(dist - r, 0.0);
    return dist*dist;
  }

  inline double MinRedShiftDistSq(const DHrectBound<2>& query,
			     const DHrectBound<2>& ref, double dz){
     
    DRange a = query.get(2);
    DRange b = ref.get(2);
    
    double z1 = b.lo - a.hi;
    double z2 = a.lo - b.hi;
    if ((z1+ fabs(z1)) + (z2 +fabs(z2)) > 2*dz){
      return BIG_BAD_NUMBER;
    } else {
      return MinSphereDistSq(query, ref);
    }
  }

 inline double MinRedShiftDistSq(const DHrectBound<2>& ref, 
				 const Vector& query, double dz){
     
   DRange b = ref.get(2);
   
   double z1 = b.lo - query[2];
   double z2 = query[2] - b.hi;
   if ((z1+ fabs(z1)) + (z2 +fabs(z2)) >  2*dz){
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

  inline double RedShiftDistSq(const Vector& query, const Vector& ref,
	double dz){
    if (fabs(query[2] - ref[2]) > dz){
      return BIG_BAD_NUMBER;
    } else {
      return SphereDistSq(query, ref);
    }
  }

}

#endif
