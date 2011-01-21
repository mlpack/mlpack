// Sparse Linear Algebra

#ifndef SPARSELA_H
#define SPARSELA_H

//#include <cmath>
#include "example.h"

//using namespace std;

/**
 * Sparse vector scaling: v <= scale * v
 */
void SparseScaleOverwrite(SVEC *v, double scale) {
  size_t i;
  if (scale == 0.0) {
    EmptyFeatures(v);
    return;
  }
  else if (scale == 1.0) {
    return;
  }
  else if (scale == -1.0) {
    for (i=0; i<v->num_nz_feats; i++) {
      v->feats[i].wval = -(v->feats[i].wval);
    }
    return;
  }
  else {
    for (i=0; i<v->num_nz_feats; i++) {
      v->feats[i].wval = v->feats[i].wval * scale;
    }
  }
}

/**
 * Sparse vector scaling: dest <= scale * x
 */
void SparseScale(SVEC *dest, double scale, EXAMPLE *x) {
  if (scale != 0.0) {
    size_t i, nz_x = x->num_nz_feats;
    dest->feats = (FEATURE *)realloc(dest->feats, nz_x*sizeof(FEATURE));
    if (scale == 1.0) {
      for (i=0; i<nz_x; i++) {
	dest->feats[i].widx = x->feats[i].widx;
	dest->feats[i].wval = x->feats[i].wval;
      }
      return;
    }
    else if (scale == -1.0) {
      for (i=0; i<nz_x; i++) {
	dest->feats[i].widx = x->feats[i].widx;
	dest->feats[i].wval = -(x->feats[i].wval);
      }
      return;
    }
    else {
      for (i=0; i<nz_x; i++) {
	dest->feats[i].widx = x->feats[i].widx;
	dest->feats[i].wval = x->feats[i].wval * scale;
      }
    }
    dest->num_nz_feats = nz_x;
  }
  else {
    EmptyFeatures(dest);
  }
}


/**
 * Sparse squared L2 norm: return w^T w
 */
double SparseSqL2Norm(SVEC *w) {
  double v = 0.0;
  size_t i, nz_w = w->num_nz_feats;
  for (i=0; i<nz_w; i++) {
    v += pow(w->feats[i].wval, 2);
  }
  return v;
}

/**
 * Sparse dot product: return w^T x
 */
double SparseDot(SVEC *w, EXAMPLE *x) {
  double dv = 0.0;
  size_t nz_w = w->num_nz_feats;
  size_t nz_x = x->num_nz_feats;
  size_t ct_w = 0, ct_x = 0;
  if (nz_w == 0 || nz_x == 0) {
    return 0.0;
  }
  else {
    while (ct_w<nz_w && ct_x<nz_x) {
      if (w->feats[ct_w].widx == x->feats[ct_x].widx) {
	dv += w->feats[ct_w].wval * x->feats[ct_x].wval;
	++ct_w;
	++ct_x;
      }
      else {
	if (w->feats[ct_w].widx > x->feats[ct_x].widx) {
	  ++ct_x;
	}
	else {
	  ++ct_w;
	}
      }
    }
    return dv;
  }
}

/**
 * Sparse vector scaled add: w<= w+ scale * x
 */
void SparseAddExpertOverwrite(SVEC *w, double scale, SVEC *x) {
  size_t nz_w = w->num_nz_feats;
  size_t nz_x = x->num_nz_feats;
  size_t ct_w = 0, ct_x = 0;
  FEATURE f_tmp;

  if (nz_x == 0 || scale == 0.0) { // x: all zeros or scale ==0
    // w remains unchanged
    return;
  }
  else if (nz_w == 0) { // w: all zeros
    nz_w = nz_x;
    w->feats = (FEATURE *)my_malloc(sizeof(FEATURE)*(nz_w));
    for (ct_w=0; ct_w<nz_w; ct_w++) {
      w->feats[ct_w].widx = x->feats[ct_w].widx;
      w->feats[ct_w].wval = scale * x->feats[ct_w].wval;
    }
    w->num_nz_feats = nz_w;
    return;
  }
  else { // neither w nor x is of all zeros
    while (ct_w<nz_w || ct_x<nz_x) {
      if (ct_w == nz_w) { // w reaches end, while x still not
	w->feats = (FEATURE *)realloc(w->feats, (nz_w+1)*sizeof(FEATURE));
	w->feats[nz_w].widx = x->feats[ct_x].widx;
	w->feats[nz_w].wval = scale * x->feats[ct_x].wval;
	nz_w ++;
	w->num_nz_feats = nz_w;
	++ct_w;
	++ct_x;
      }
      else if (ct_x == nz_x) { // x reaches end, while w still not
	// the succeeding w remain unchanged
	break;
      }
      else { // neither w nor x reaches end
	if (w->feats[ct_w].widx == x->feats[ct_x].widx) {
	  w->feats[ct_w].wval = w->feats[ct_w].wval + scale * x->feats[ct_x].wval;
	  ++ct_w;
	  ++ct_x;
	}
	else if (w->feats[ct_w].widx > x->feats[ct_x].widx) {
	  f_tmp.widx = x->feats[ct_x].widx;
	  f_tmp.wval = scale * x->feats[ct_x].wval;
	  InsertOne(w, &f_tmp, ct_w);
	  ++ct_w;
	  ++ct_x;
	}
	else { // w->feats[ct_w].widx < x->feats[ct_x].widx
	  // w->feats[ct_w] remains unchanged
	  ++ct_w;
	}
      }
    }
    /*
    // shrink w
    for (ct_w=0; ct_w<nz_w; ct_w++) {
      if (fabs(w->feats[ct_w].wval) < 1.0e-5) { // TODO: thresholding
	RemoveOne(w, ct_w);
	ct_w--;
      }
    }
    */
  }
}

void SparseAddExpertOverwrite(SVEC *w, double scale, EXAMPLE *x) {
  size_t nz_w = w->num_nz_feats;
  size_t nz_x = x->num_nz_feats;
  size_t ct_w = 0, ct_x = 0;
  FEATURE f_tmp;

  if (nz_x == 0 || scale == 0.0) { // x: all zeros or scale ==0
    // w remains unchanged
    return;
  }
  else if (nz_w == 0) { // w: all zeros
    nz_w = nz_x;
    w->feats = (FEATURE *)my_malloc(sizeof(FEATURE)*(nz_w));
    for (ct_w=0; ct_w<nz_w; ct_w++) {
      w->feats[ct_w].widx = x->feats[ct_w].widx;
      w->feats[ct_w].wval = scale * x->feats[ct_w].wval;
    }
    w->num_nz_feats = nz_w;
    return;
  }
  else { // neither w nor x is of all zeros
    while (ct_w<nz_w || ct_x<nz_x) {
      if (ct_w == nz_w) { // w reaches end, while x still not
	w->feats = (FEATURE *)realloc(w->feats, (nz_w+1)*sizeof(FEATURE));
	w->feats[nz_w].widx = x->feats[ct_x].widx;
	w->feats[nz_w].wval = scale * x->feats[ct_x].wval;
	nz_w ++;
	w->num_nz_feats = nz_w;
	++ct_w;
	++ct_x;
      }
      else if (ct_x == nz_x) { // x reaches end, while w still not
	// the succeeding w remain unchanged
	break;
      }
      else { // neither w nor x reaches end
	if (w->feats[ct_w].widx == x->feats[ct_x].widx) {
	  w->feats[ct_w].wval = w->feats[ct_w].wval + scale * x->feats[ct_x].wval;
	  ++ct_w;
	  ++ct_x;
	}
	else if (w->feats[ct_w].widx > x->feats[ct_x].widx) {
	  f_tmp.widx = x->feats[ct_x].widx;
	  f_tmp.wval = scale * x->feats[ct_x].wval;
	  InsertOne(w, &f_tmp, ct_w);
	  ++ct_w;
	  ++ct_x;
	}
	else { // w->feats[ct_w].widx < x->feats[ct_x].widx
	  // w->feats[ct_w] remains unchanged
	  ++ct_w;
	}
      }
    }
    /*
    // shrink w
    for (ct_w=0; ct_w<nz_w; ct_w++) {
      if (fabs(w->feats[ct_w].wval) < 1.0e-5) { // TODO: thresholding
	RemoveOne(w, ct_w);
	ct_w--;
      }
    }
    */
  }
}

/**
 * Sparse vector add: w<= w + x
 */
void SparseAddOverwrite(SVEC *w, SVEC *x) {
  size_t nz_w = w->num_nz_feats;
  size_t nz_x = x->num_nz_feats;
  size_t ct_w = 0, ct_x = 0;
  FEATURE f_tmp;

  if (nz_x == 0) { // x: all zeros
    // w remains unchanged
    return;
  }
  else if (nz_w == 0) { // w: all zeros
    nz_w = nz_x;
    w->feats = (FEATURE *)my_malloc(sizeof(FEATURE)*(nz_w));
    for (ct_w=0; ct_w<nz_w; ct_w++) {
      w->feats[ct_w].widx = x->feats[ct_w].widx;
      w->feats[ct_w].wval = x->feats[ct_w].wval;
    }
    w->num_nz_feats = nz_w;
    return;
  }
  else { // neither w nor x is of all zeros
    while (ct_w<nz_w || ct_x<nz_x) {
      if (ct_w == nz_w) { // w reaches end, while x still not
	w->feats = (FEATURE *)realloc(w->feats, (nz_w+1)*sizeof(FEATURE));
	w->feats[nz_w].widx = x->feats[ct_x].widx;
	w->feats[nz_w].wval = x->feats[ct_x].wval;
	nz_w ++;
	w->num_nz_feats = nz_w;
	++ct_w;
	++ct_x;
      }
      else if (ct_x == nz_x) { // x reaches end, while w still not
	// the succeeding w remain unchanged
	break;
      }
      else { // neither w nor x reaches end
	if (w->feats[ct_w].widx == x->feats[ct_x].widx) {
	  w->feats[ct_w].wval = w->feats[ct_w].wval + x->feats[ct_x].wval;
	  ++ct_w;
	  ++ct_x;
	}
	else if (w->feats[ct_w].widx > x->feats[ct_x].widx) {
	  f_tmp.widx = x->feats[ct_x].widx;
	  f_tmp.wval = x->feats[ct_x].wval;
	  InsertOne(w, &f_tmp, ct_w);
	  ++ct_w;
	  ++ct_x;
	}
	else { // w->feats[ct_w].widx < x->feats[ct_x].widx
	  // w->feats[ct_w] remains unchanged
	  ++ct_w;
	}
      }
    }
    /*
    // shrink w
    for (ct_w=0; ct_w<nz_w; ct_w++) {
      if (fabs(w->feats[ct_w].wval) < 1.0e-5) { // TODO: thresholding
	RemoveOne(w, ct_w);
	ct_w--;
      }
    }
    */
  }
}

/**
 * Sparse vector add: w<= p - n
 */
void SparseMinus(SVEC *w, SVEC *p, SVEC *n) {
  size_t nz_w = 0;
  size_t nz_p = p->num_nz_feats;
  size_t nz_n = n->num_nz_feats;
  size_t ct_p = 0, ct_n = 0, ct_w = 0;

  while (ct_p<nz_p || ct_n<nz_n) {
    if (ct_p == nz_p) {
      ++ct_n;
      ++nz_w;
    }
    else if (ct_n == nz_n) {
      ++ct_p;
      ++nz_w;
    }
    else {
      if (p->feats[ct_p].widx == n->feats[ct_n].widx) {
	++ct_p;
	++ct_n;
	++nz_w;
      }
      else if (p->feats[ct_p].widx > n->feats[ct_n].widx) {
	++ct_n;
	++nz_w;
      }
      else {
	++ct_p;
	++nz_w;
      }
    }
  }
  w->feats = (FEATURE *)realloc(w->feats, nz_w*sizeof(FEATURE));
  w->num_nz_feats = nz_w;

  ct_p = 0; ct_n = 0; ct_w = 0;
  while (ct_p<nz_p || ct_n<nz_n) {
    if (ct_p == nz_p) {
      w->feats[ct_w].widx = n->feats[ct_n].widx;
      w->feats[ct_w].wval = - (n->feats[ct_n].wval);
      ++ct_n;
      ++ct_w;
    }
    else if (ct_n == nz_n) {
      w->feats[ct_w].widx = p->feats[ct_p].widx;
      w->feats[ct_w].wval = p->feats[ct_p].wval;
      ++ct_p;
      ++ct_w;
    }
    else {
      if (p->feats[ct_p].widx == n->feats[ct_n].widx) {
	w->feats[ct_w].widx = p->feats[ct_p].widx;
	w->feats[ct_w].wval = p->feats[ct_p].wval - (n->feats[ct_n].wval);
	++ct_p;
	++ct_n;
	++ct_w;
      }
      else if (p->feats[ct_p].widx > n->feats[ct_n].widx) {
	w->feats[ct_w].widx = n->feats[ct_n].widx;
	w->feats[ct_w].wval = - (n->feats[ct_n].wval);
	++ct_n;
	++ct_w;
      }
      else {
	w->feats[ct_w].widx = p->feats[ct_p].widx;
	w->feats[ct_w].wval = p->feats[ct_p].wval;
	++ct_p;
	++ct_w;
      }
    }
  }
  /*
  // shrink w
  if (nz_w > 0) {
    for (ct_w=0; ct_w<w->num_nz_feats; ct_w++) {
      if (fabs(w->feats[ct_w].wval) < 1.0e-5) { // TODO: thresholding
	RemoveOne(w, ct_w);
      }
    }
  }
  */
}

/**
 * Sparse vector exponential dot multiply: w<= w .* exp(x)
 */
void SparseExpMultiplyOverwrite(SVEC *w, SVEC *x) {
  size_t nz_w = w->num_nz_feats;
  size_t nz_x = x->num_nz_feats;
  size_t ct_w = 0, ct_x = 0;

  if (nz_x == 0 || nz_w == 0) { // x/w: all zeros
    // w remains unchanged
    return;
  }
  else { // neither w nor x is of all zeros
    while (ct_w<nz_w || ct_x<nz_x) {
      if (ct_w == nz_w || ct_x == nz_x) { // w/x reaches end
	break;
      }
      else { // neither w nor x reaches end
	if (w->feats[ct_w].widx == x->feats[ct_x].widx) {
	  if (x->feats[ct_x].wval != 0)
	    w->feats[ct_w].wval = w->feats[ct_w].wval * exp(x->feats[ct_x].wval);
	  ++ct_w;
	  ++ct_x;
	}
	else if (w->feats[ct_w].widx < x->feats[ct_x].widx) {
	  ++ct_w;
	}
	else { // w->feats[ct_w].widx > x->feats[ct_x].widx
	  ++ct_x;
	}
      }
    }
    /*
    // shrink w
    for (ct_w=0; ct_w<nz_w; ct_w++) {
      if (fabs(w->feats[ct_w].wval) < 1.0e-5) { // TODO: thresholding
	RemoveOne(w, ct_w);
	ct_w--;
      }
    }
    */
  }
}

/**
 * Sparse vector negative exponential dot multiply: w<= w .* exp(-x)
 */
void SparseNegExpMultiplyOverwrite(SVEC *w, SVEC *x) {
  size_t nz_w = w->num_nz_feats;
  size_t nz_x = x->num_nz_feats;
  size_t ct_w = 0, ct_x = 0;

  if (nz_x == 0 || nz_w == 0) { // x/w: all zeros
    // w remains unchanged
    return;
  }
  else { // neither w nor x is of all zeros
    while (ct_w<nz_w || ct_x<nz_x) {
      if (ct_w == nz_w || ct_x == nz_x) { // w/x reaches end
	break;
      }
      else { // neither w nor x reaches end
	if (w->feats[ct_w].widx == x->feats[ct_x].widx) {
	  if (x->feats[ct_x].wval != 0)
	    w->feats[ct_w].wval = w->feats[ct_w].wval / exp(x->feats[ct_x].wval);
	  ++ct_w;
	  ++ct_x;
	}
	else if (w->feats[ct_w].widx < x->feats[ct_x].widx) {
	  ++ct_w;
	}
	else { // w->feats[ct_w].widx > x->feats[ct_x].widx
	  ++ct_x;
	}
      }
    }
    /*
    // shrink w
    for (ct_w=0; ct_w<nz_w; ct_w++) {
      if (fabs(w->feats[ct_w].wval) < 1.0e-5) { // TODO: thresholding
	RemoveOne(w, ct_w);
	ct_w--;
      }
    }
    */
  }
}

#endif
