#ifndef PERIODIC_TREE_H
#define PERIODIC_TREE_H

#include <fastlib/fastlib.h>
#include "fastlib/tree/bounds.h"

namespace prdc {

  inline double MinDistanceSq(const DHrectBound<2>& query, 
			      const DHrectBound<2>& ref,
			      const Vector& box_size){
   double sum = 0;
   for (index_t d = 0; d < ref.dim(); d++){   
     DRange a = query.get(d);
     DRange b = ref.get(d);
     double v = 0, L = box_size[d];
     if (a.lo > b.hi){
       v = min(a.lo - b.hi, b.lo - a.hi + L);
     } else if (b.lo > a.hi ) {
       v = min(a.lo - b.hi + L, b.lo - a.hi);
     }   
     sum += v*v;
   }   
   return sum;
 }


  inline double MinDistanceSq(const DHrectBound<2>& ref,
			      const Vector& query, const Vector& box_size){
   double sum = 0;  
        
   for (index_t d = 0; d < ref.dim(); d++){
     double v = 0, a = query[d], L = box_size[d];
     double d1, d2;
     const DRange b = ref.get(d);  
     d1 = a - b.hi + L*(a < b.lo);
     d2 = b.lo - a + L*(a > b.hi);
     v =  max(min(d1, d2), 0.0);     
     sum += v*v;
   }   
   return sum;
 }

  inline double MinDistanceSqWrap(const DHrectBound<2>& ref, 
				  const DHrectBound<2>& query, 
				  const Vector& box_size){
    double sum = 0;
    for (index_t d = 0; d < ref.dim(); d++){       
      const DRange a = query.get(d);
      const DRange b = ref.get(d);
      
      double v = 0, d1, d2, d3, L = box_size[d];
      d1 = (a.hi >= a.lo | b.hi >= b.lo)*min(b.lo - a.hi, a.lo-b.hi);
      d2 = (a.hi >= a.lo & b.hi >= b.lo)*min(b.lo - a.hi, a.lo-b.hi + L);
      d3 = (a.hi >= a.lo & b.hi >= b.lo)*min(b.lo - a.hi + L, a.lo-b.hi);
      v = (d1 + fabs(d1)) + (d2+  fabs(d2)) + (d3 + fabs(d3));
      sum += v*v;
    }   
    return sum / 4;
  } 

  inline double MinDistanceSqWrap(const DHrectBound<2>& ref,
				  const Vector& query, const Vector& box_size){
   double sum = 0;  
         
   for (index_t d = 0; d < ref.dim(); d++){
     const DRange b = ref.get(d);
     double a = query[d];
     double v = 0, bh;
     bh = b.hi-b.lo;
     bh = bh - floor(bh / box_size[d])*box_size[d];
     a = a - b.lo;
     a = a - floor(a / box_size[d])*box_size[d];
     if (bh < a){
       v = min(a - bh, box_size[d]-a);
     }
     sum += v*v;
   }
   
   return sum;
 }


 /**
  * Expand this bounding box to encompass another point. Done to 
  * minimize added volume in periodic coordinates.
  */
  /*
  void Add(DHrectBound<2>* this, const Vector&  other, const Vector& size){
   // Catch case of uninitialized bounds 
   if ((this->get(0)).hi < -10*size[0]){
     (*this) |= other;    
   }

   for (index_t i= 0; i < this->dim(); i++){
     double ah, al;
     ah = (this->get(i)).hi - other[i];
     al = (this->get(i)).lo - other[i];
     // Project onto range 0 to L
     ah = ah - floor(ah / size[i])*size[i];
     al = al - floor(al / size[i])*size[i];
     if (ah >= al){
       if (size[i] - al < ah){
	 bounds_[i].hi = other[i] - floor(other[i] / size[i] + 0.5)*size[i];
       } else {
	 bounds_[i].lo = other[i] - floor(other[i] / size[i] + 0.5)*size[i];;
       }      
     }    
   }
   return *this;
   }*/




  /*
    These are some functions for max distances in periodic coordinates that
    I'm no longer using. I'm keeping them here in case they turn out
    to be useful later on.
   */


  /*
 double PeriodicMaxDistance1Norm(const Vector& point, const Vector& box_size)
 const {
   double sum = 0;
   const DRange *a = this->bounds_;  
   for (index_t d = 0; d < dim_; d++){
     double b = point[d];
     double v = box_size[d] / 2.0;
     double ah, al;
     ah = a[d].hi - b;
     ah = ah - floor(ah / box_size[d])*box_size[d];
     if (ah < v){
       v = ah;
     } else {
       al = a[d].lo - b;
       al = al - floor(al / box_size[d])*box_size[d];
       if (al > v){
	 v = 2*v-al;
       }
     }
     sum += fabs(v);
   }      
   return sum;
 }

 double PeriodicMaxDistanceSq(const Vector& point, const Vector& box_size)
 const {
   double sum = 0;
   const DRange *a = this->bounds_;  
   for (index_t d = 0; d < dim_; d++){
     double b = point[d];
     double v = box_size[d] / 2.0;
     double ah, al;
     ah = a[d].hi - b;
     ah = ah - floor(ah / box_size[d])*box_size[d];
     if (ah < v){
       v = ah;
     } else {
       al = a[d].lo - b;
       al = al - floor(al / box_size[d])*box_size[d];
       if (al > v){
	 v = 2*v-al;
       }
     }
     sum += math::PowAbs<t_pow, 1>(v);
   }   
   return math::Pow<2, t_pow>(sum);
 }


 double PeriodicMaxDistanceSq(const DHrectBound& other, const Vector& box_size)
 const {
   double sum = 0;
   const DRange *a = this->bounds_;
   const DRange *b = other.bounds_;
   
   DEBUG_SAME_SIZE(dim_, other.dim_);
   
   for (index_t d = 0; d < dim_; d++){
     double v = box_size[d] / 2.0;
     double dh, dl;
     dh = a[d].hi - b[d].lo;
     dh = dh - floor(dh / box_size[d])*box_size[d];
     dl = b[d].hi - a[d].lo;
     dl = dl - floor(dl / box_size[d])*box_size[d];
     v = max(min(dh,v), min(dl, v));
       
     sum += math::PowAbs<t_pow, 1>(v);
   }   
   return math::Pow<2, t_pow>(sum);
 }

  */

}

#endif
