#ifndef SVM_LIB_H
#define SVM_LIB_H

#include <fastlib/fastlib.h>

namespace SVMLib {
  /** Kernel function definition */
  typedef double (*KernelFunc)(const Vector& x, const Vector& y);
  /** SMO options */
  struct SMOOptions {
    enum VERBOSE {NONE = 0, ITER, FINAL} verbose;
    double tolKKT; // default 1e-6
    double KKTViolationLevel; // default 1e-3
  };

  class IndexSet {
    index_t n_total, n_set; // number of points
    ArrayList<index_t> index_set; // index subset, size n_set
    ArrayList<index_t> set_index; // index of points in the index_set, size n_total
  public:
    IndexSet(index_t n) {
      n_total = n;
      n_set = 0;
      index_set.Init();
      set_index.Init();
      InitEmpty();
    }
    void InitEmpty() {
      for (index_t i = 0; i < n_total; i++) set_index[i] = -1;
      index_set.Clear();
    }
    void addremove(index_t i, bool b = true) {
      DEBUG_ASSERT(i < n_total);
      // adding
      if (b && set_index[i] == -1) {
	index_set.PushBackCopy(i);
	set_index[i] = n_set;
	n_set++;
      }
      // removing
      if (!b && set_index[i] != -1) {
	index_t index = set_index[i];
	set_index[i] = -1;
	if (index < n_set-1) {
	  index_set[index] = index_set[n_set-1]; // move the last point to index position
	  set_index[index_set[n_set-1]] = index; // set the new index
	}
	n_set--;
      }
    }
    index_t get_n() const { return n_set; }
    const ArrayList<index_t>& get_index() const { return index_set; }
    bool is_set(index_t i) const { 
      DEBUG_ASSERT(i < n_total);
      return set_index[i] != -1; 
    }
    int operator[] (index_t i) const {
      DEBUG_ASSERT(i < n_set);
      return index_set[i];
    }
  };

  /** SMO main function
   *  Solve l1-SVM with box constraints
   *  min 1/2 alpha^T K alpha - alpha^T 1
   *  s.t 0 \leq alpha \leq box
   *      alpha^T y = 0
   */
  int seqminopt(const Matrix& X, const Vector& y, const Vector& box,
		KernelFunc kfunc, SMOOptions options,
		Vector& alpha, IndexSet& SVindex, double& offset);

  class Kernel {
    int n_points;
    bool kernel_stored;
    const Matrix& X_;
    KernelFunc kfunc_;
    Matrix full_kernel;
    Vector kernel_diag;
  public:
    Kernel(KernelFunc kfunc, const Matrix& X) : X_(X) {
      n_points = X_.n_cols();
      kernel_stored = (n_points <= 1000);
      kfunc_ = kfunc;
      kernel_diag.Init(n_points);
      if (kernel_stored) {
	full_kernel.Init(n_points, n_points);
	for (index_t i = 0; i < n_points; i++)
	  for (index_t j = 0; j < n_points; j++)
	    full_kernel.ref(i, j) = kernel_call(i, j);
	for (index_t i = 0; i < n_points; i++)
	  kernel_diag[i] = full_kernel.get(i, i);
      }
      else {
	// TODO
	full_kernel.Init(0, 0);
      }
    }
    void get_element(index_t i, index_t j, 
		     double& Kii, double& Kjj, double Kij) {
      if (kernel_stored) {
	Kii = kernel_diag[i];
	Kjj = kernel_diag[j];
	Kij = full_kernel.get(i, j);
      }
      else {
	// TODO
      }
    }

    /** Get a kernel matrix column, col_i must be uninitialized */
    void get_column(index_t i, Vector* col_i) {
      if (kernel_stored) {
	full_kernel.MakeColumnVector(i, col_i);
      }
      else {
	// TODO
      }
    }
    
    /** Get the diagonal of the kernel matrix, diag must be uninitialized*/
    void get_diag(Vector* diag) {
      diag->Alias(kernel_diag);
    }
  private:
    double kernel_call(index_t i, index_t j) {
      Vector col_i;
      Vector col_j;
      X_.MakeColumnVector(i, &col_i);
      X_.MakeColumnVector(j, &col_j);
      return kfunc_(col_i, col_j);
    }
  };
};

#endif
