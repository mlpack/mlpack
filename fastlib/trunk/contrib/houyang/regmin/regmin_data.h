/**
 * @author Hua Ouyang
 *
 * @file regmin_data.h
 *
 * This head file contains sparse data related functions
 *
 */

#ifndef U_REGMIN_DATA_H
#define U_REGMIN_DATA_H

#include "fastlib/fastlib.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


#define ID_LINEAR 0
#define ID_GAUSSIAN 1

/**
* An nonzero entry(dimension) of a data point
*/
struct NZ_entry {
  index_t index;
  double value;
};

/**
* Sparse labeled dataset
*/
struct Dataset_sl {
  index_t n_features;
  index_t n_points;
  index_t n_classes;
  double *y; // labels of data points
  struct NZ_entry **x; // data points
  //ArrayList<NZ_entry> *x; // TODO
};


/**
* Class for Linear Kernel
*/
class SVMLinearKernel {
 public:
  // Init of kernel parameters
  ArrayList<double> kpara_; // kernel parameters
  void Init(datanode *node) { //TODO: NULL->node
    kpara_.Init(); 
  }
  // Kernel name
  void GetName(String* kname) {
    kname->Copy("linear");
  }
  // Get an type ID for kernel
  int GetTypeId() {
    return ID_LINEAR;
  }
  // Kernel value evaluation
  double Eval(NZ_entry *pa, NZ_entry *pb) {
    // sparse dot product
    double kv = 0.0;
    if (pa->index == -1 || pb->index == -1) {
      return 0.0;
    }
    else {
      while (pa->index !=-1 && pb->index !=-1) {
	if (pa->index == pb->index) {
	  kv += pa->value * pb->value;
	  ++pa;
	  ++pb;
	}
	else {
	  if (pa->index > pb->index) {
	    ++pb;
	  }
	  else {
	    ++pa;
	  }
	}
      }
    }
    return kv; 
  }
  // Save kernel parameters to file
  void SaveParam(FILE* fp) {
  }
};

/**
* Class for Gaussian RBF Kernel
*/
class SVMRBFKernel {
 public:
  // Init of kernel parameters
  ArrayList<double> kpara_; // kernel parameters
  void Init(datanode *node) { //TODO: NULL->node
    kpara_.Init(2);
    kpara_[0] = fx_param_double_req(NULL, "sigma"); // sigma
    kpara_[1] = -1.0 / (2 * kpara_[0] * kpara_[0]); // -gamma = -1/(2 sigma^2)
  }
  // Kernel name
  void GetName(String* kname) {
    kname->Copy("gaussian");
  }
  // Get an type ID for kernel
  int GetTypeId() {
    return ID_GAUSSIAN;
  }
  // Kernel value evaluation
  double Eval(NZ_entry *pa, NZ_entry *pb) {
    double kv = 0;
    while (pa->index !=-1 && pb->index !=-1) {
      if (pa->index == pb->index) {
	double tmp = pa->value - pb->value;
	kv += tmp * tmp;
	++pa;
	++pb;
      }
      else {
	if (pa->index > pb->index) {
	  kv += pb->value * pb->value;
	  ++pb;
	}
	else {
	  kv += pa->value * pa->value;
	  ++pa;
	}
      }
    }
    while (pa->index != -1) {
      kv += pa->value * pa->value;
      ++pa;
    }
    while (pb->index != -1) {
      kv += pb->value * pb->value;
      ++ pb;
    }
    kv = exp( kpara_[1] * kv );
    return kv;
  }
  // Save kernel parameters to file
  void SaveParam(FILE* fp) {
    fprintf(fp, "sigma %g\n", kpara_[0]);
    fprintf(fp, "gamma %g\n", kpara_[1]);
  }
};


/**
 * Sparse vector scaling: w<= scale *w
 */
void SparseScale(double scale, ArrayList<NZ_entry> &w) {
  if (scale == 0) {
    w.ShrinkTo(0);
  }
  else {
    for (index_t i=0; i<w.size(); i++) {
      w[i].value = w[i].value * scale;
    }
  }
}

/**
 * Sparse dot product: return w^T x
 */
double SparseDot(ArrayList<NZ_entry> &w, NZ_entry *x) {
  double dv = 0.0;
  index_t w_nz = w.size();
  index_t ct = 0;
  if (w_nz == 0 || x->index == -1) {
    return 0.0;
  }
  else {
    while (ct<w_nz && x->index !=-1) {
      if (w[ct].index == x->index) {
	dv += w[ct].value * x->value;
	++ct;
	++x;
      }
      else {
	if (w[ct].index > x->index) {
	  ++x;
	}
	else {
	  ++ct;
	}
      }
    }
    return dv;
  }
}

/**
 * Sparse vector scaled add: w<= w+ scale * x
 */
void SparseAddExpert(double scale, NZ_entry *x, ArrayList<NZ_entry> &w) {
  index_t ct = 0;
  NZ_entry nz_tmp;

  if (x->index == -1 || scale == 0) { // x: all zeros or scale ==0
    // w remains unchanged
  }
  else if (w.size() == 0) { // w: all zeros
    while (x->index != -1) {
      nz_tmp.index  = x->index;
      nz_tmp.value = scale * x->value;
      w.PushBackCopy(nz_tmp);
      ++x;
    }
  }
  else { // neither w nor x is of all zeros
    while (ct<w.size() || x->index !=-1) {
      if (ct == w.size()) { // w reaches end, while x still not
	nz_tmp.index = x->index;
	nz_tmp.value = scale * x->value;
	w.PushBackCopy(nz_tmp);
	++ct;
	++x;
      }
      else if (x->index == -1) { // x reaches end, while w still not
	// the succeeding w remain unchanged
	break;
      }
      else { // neither w nor x reaches end
	if (w[ct].index == x->index) {
	  w[ct].value = w[ct].value + scale * x->value;
	  ++ct;
	  ++x;
	}
	else if (w[ct].index > x->index) {
	  nz_tmp.index = x->index;
	  nz_tmp.value = scale * x->value;
	  w.InsertCopy(ct, nz_tmp); // w's dimension increases by 1
	  ++ct;
	  ++x;
	}
	else { // w[ct].index < x->index
	  // w[ct] remains unchanged
	  ++ct;
	}
      }
    }
    // shrink w
    for (ct=0; ct<w.size(); ct++) {
      if (fabs(w[ct].value) < 1.0e-5) { // TODO: thresholding
	w.Remove(ct);
	ct--;
      }
    }
  }

}

/**
 * Sparse vector subtraction: z <= y-x
 */
void SparseSub(ArrayList<NZ_entry> &x,ArrayList<NZ_entry> &y, ArrayList<NZ_entry> &z) {
  index_t x_size, y_size;
  index_t i, ct_x, ct_y;
  x_size = x.size();
  y_size = y.size();
  ct_x = 0; ct_y = 0;
  NZ_entry nz_tmp;
 
  /*
  for (index_t i=0; i<y.size(); i++) {
    printf("y[%d].index=%d,.value=%f\n", i, y[i].index, y[i].value);
  }
  for (index_t i=0; i<x.size(); i++) {
    printf("x[%d].index=%d,.value=%f\n", i, x[i].index, x[i].value);
  }
  */

  z.ShrinkTo(0);
  
  if (y_size == 0) {  // y: all zeros
    z.GrowTo(x_size);
    for (i=0; i<x_size; i++) {
      z[i].index = x[i].index;
      z[i].value = - x[i].value;
    }
  }
  else if (x_size == 0) { // x: all zeros
    z.GrowTo(y_size);
    for (i=0; i<y_size; i++) {
      z[i].index = y[i].index;
      z[i].value = y[i].value;
    }
  }
  else { // neither x nor y is of all zeros
    while ( ct_x < x_size || ct_y < y_size ) {
      if (ct_x == x_size) { // x reaches end, while y still not
	nz_tmp.index = y[ct_y].index;
	nz_tmp.value = y[ct_y].value;
	z.PushBackCopy(nz_tmp);
	++ct_y;
      }
      else if (ct_y == y_size) { // y reaches end, while x still not
	nz_tmp.index = x[ct_x].index;
	nz_tmp.value = - x[ct_x].value;
	z.PushBackCopy(nz_tmp);
	++ct_x;
      }
      else { // neither x nor y reaches end
	if (x[ct_x].index == y[ct_y].index) {
	  nz_tmp.index = x[ct_x].index;
	  nz_tmp.value = y[ct_y].value - x[ct_x].value;
	  z.PushBackCopy(nz_tmp);
	  ++ct_x;
	  ++ct_y;
	}
	else if (y[ct_y].index > x[ct_x].index) {
	  nz_tmp.index = x[ct_x].index;
	  nz_tmp.value = - x[ct_x].value;
	  z.InsertCopy(ct_y, nz_tmp); // w's dimension increases by 1
	  ++ct_x;
	}
	else { // y[ct].index < x[ct_x].index
	  ++ct_y;
	}
      }
    }
    
  }
  
  /*
  for (index_t i=0; i<z.size(); i++) {
    printf("z[%d].index=%d,.value=%f\n", i, z[i].index, z[i].value);
  }
  */

}




/**
 * Sparse vector subtraction: y <= y-x
 */
void SparseSubOverwrite(ArrayList<NZ_entry> &x,ArrayList<NZ_entry> &y) {
  index_t x_size;
  index_t i, ct_x, ct_y;
  x_size = x.size();
  ct_x = 0; ct_y = 0;
  NZ_entry nz_tmp;

  if (y.size() == 0) {  // y: all zeros
    y.GrowTo(x_size);
    for (i=0; i<x_size; i++) {
      y[i].index = x[i].index;
      y[i].value = - x[i].value;
    }
  }
  else if (x_size == 0) { // x: all zeros
    // y remains unchanged
  }
  else { // neither x nor y is of all zeros
    while (ct_x < x_size || ct_y < y.size()  ) {
      if (ct_x == x_size) { // x reaches end, while y still not
	break;
      }
      else if (ct_y == y.size()) { // y reaches end, while x still not
	nz_tmp.index = x[ct_x].index;
	nz_tmp.value = - x[ct_x].value;
	y.PushBackCopy(nz_tmp);
	++ct_x;
	++ct_y;
      }
      else { // neither x nor y reaches end
	if (y[ct_y].index == x[ct_x].index) {
	  y[ct_y].value = y[ct_y].value - x[ct_x].value;
	  ++ct_x;
	  ++ct_y;
	}
	else if (y[ct_y].index > x[ct_x].index) {
	  nz_tmp.index = x[ct_x].index;
	  nz_tmp.value = - x[ct_x].value;
	  y.InsertCopy(ct_y, nz_tmp); // w's dimension increases by 1
	  ++ct_x;
	  ++ct_y;
	}
	else { // y[ct].index < x[ct_x].index
	  ++ct_y;
	}
      }
    }
    
  }

}





/**
 * Sparse element-wise multiplication of vectors: y <= y .* x
 */
void SparseElementMulOverwrite(ArrayList<NZ_entry> &y, ArrayList<NZ_entry> &x) {
  index_t ct_x = 0;
  index_t ct_y = 0;

  if (y.size() == 0) { // y: all zeros
    // y remains unchanged
  }
  else if (x.size() == 0) { // x: all zeros
    y.ShrinkTo(0);
  }
  else { // neither x nor y is of all zeros
    while (ct_x < x.size() || ct_y < y.size()) {
      if (ct_x == x.size()) { // x reaches end, while y still not
	y.Remove(ct_y);
	//printf("ct_y=%d\n", ct_y);
      }
      else if (ct_y == y.size()) { // y reaches end, while x still not
	break;
      }
      else { // neither x nor y reaches end
	if (x[ct_x].index == y[ct_y].index) {
	  y[ct_y].value = y[ct_y].value * x[ct_x].value;
	  ++ct_x;
	  ++ct_y;
	}
	else if (y[ct_y].index > x[ct_x].index) {
	  ++ct_x;
	  //printf("y[%d].index=%d;  x[%d].index=%d\n",ct_y, y[ct_y].index, ct_x, x[ct_x].index);
	}
	else { // y[ct].index < x[ct_x].index
	  y.Remove(ct_y);
	}
      }
    }
  }

}


void print_svec(ArrayList<NZ_entry> &v) {
  printf("v== ");
  for (index_t i=0; i<v.size(); i++) {
    printf("%d:%g, ", v[i].index, v[i].value);
  }
  printf("\n");
}

void print_ex(NZ_entry *x) {
  printf("x== ");
  while (x->index !=-1) {
    printf("%d:%g, ", x->index, x->value);
    ++x;
  }
  printf("\n");
}


#endif
