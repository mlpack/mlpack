#ifndef AXILROD_TELLER_FORCE_H
#define AXILROD_TELLER_FORCE_H

#include "fastlib/fastlib.h"

#include "mlpack/series_expansion/kernel_aux.h"

#include "multibody_kernel.h"

class AxilrodTellerForceProblem {

 public:

  class MultiTreeDelta {

   public:
    Vector negative_force_vector_e;

    double l1_norm_negative_force_vector_u;

    double l1_norm_positive_force_vector_l;

    Vector positive_force_vector_e;

    double n_pruned;
    
    double used_error;
    
   public:
    void SetZero() {
      negative_force_vector_e.SetZero();
      l1_norm_negative_force_vector_u = 0;
      l1_norm_positive_force_vector_l = 0;
      positive_force_vector_e.SetZero();
      n_pruned = 0;
      used_error = 0;
    }

    void Init() {

      // Hard-codes to use 3-dimensional vectors.
      negative_force_vector_e.Init(3);
      positive_force_vector_e.Init(3);

      // Initializes to zeros...
      SetZero();
    }    
  };

  class MultiTreeQueryPostponed {
    
   public:

    Vector negative_force_vector_e;

    double l1_norm_negative_force_vector_u;

    double l1_norm_positive_force_vector_l;

    Vector positive_force_vector_e;

    double n_pruned;
    
    double used_error;

    void ApplyDelta(const MultiTreeDelta &delta_in) {
      la::AddTo(3, delta_in.negative_force_vector_e.ptr(),
		negative_force_vector_e.ptr());
      l1_norm_negative_force_vector_u += 
	delta_in.l1_norm_negative_force_vector_u;
      l1_norm_positive_force_vector_l +=
	delta_in.l1_norm_positive_force_vector_l;
      la::AddTo(3, delta_in.positive_force_vector_e.ptr(),
		positive_force_vector_e.ptr());
      n_pruned += delta_in.n_pruned;
      used_error += delta_in.used_error;
    }

    void ApplyPostponed(const MultiTreeQueryPostponed &postponed_in) {
      la::AddTo(3, postponed_in.negative_force_vector_e.ptr(),
		negative_force_vector_e.ptr());
      l1_norm_negative_force_vector_u +=
	postponed_in.l1_norm_negative_force_vector_u;
      l1_norm_positive_force_vector_l +=
	postponed_in.l1_norm_positive_force_vector_l;
      la::AddTo(3, postponed_in.positive_force_vector_e.ptr(),
		positive_force_vector_e.ptr());
      n_pruned += postponed_in.n_pruned;
      used_error += postponed_in.used_error;     
    }

    void SetZero() {
      negative_force_vector_e.SetZero();
      l1_norm_negative_force_vector_u = 0;
      l1_norm_positive_force_vector_l = 0;
      positive_force_vector_e.SetZero();
      n_pruned = 0;
      used_error = 0;
    }

    void Init() {

      // Hard-codes to use 3-dimensional vectors.
      negative_force_vector_e.Init(3);
      positive_force_vector_e.Init(3);

      // Initializes to zeros...
      SetZero();
    }
  };

  class MultiTreeQuerySummary {
   public:

    double l1_norm_negative_force_vector_u;
    
    double l1_norm_positive_force_vector_l;
    
    double n_pruned_l;
    
    double used_error_u;

   public:

    void SetZero() {
      l1_norm_negative_force_vector_u = 0;
      l1_norm_positive_force_vector_l = 0;
      n_pruned_l = 0;
      used_error_u = 0;
    }
    
    void ApplyPostponed(const MultiTreeQueryPostponed &postponed_in) {
      l1_norm_negative_force_vector_u += 
	postponed_in.l1_norm_negative_force_vector_u;
      l1_norm_positive_force_vector_l +=
	postponed_in.l1_norm_positive_force_vector_l;
      n_pruned_l += postponed_in.n_pruned;
      used_error_u += postponed_in.used_error;
    }

    void Accumulate(const MultiTreeQuerySummary &summary_in) {
      l1_norm_negative_force_vector_u = 
	std::min(l1_norm_negative_force_vector_u,
		 summary_in.l1_norm_negative_force_vector_u);
      l1_norm_positive_force_vector_l = 
	std::min(l1_norm_positive_force_vector_l,
		 summary_in.l1_norm_positive_force_vector_l);
      n_pruned_l = std::min(n_pruned_l, summary_in.n_pruned_l);
      used_error_u = std::max(used_error_u, summary_in.used_error_u);
    }

    void StartReaccumulate() {
      l1_norm_negative_force_vector_u = DBL_MAX;
      l1_norm_positive_force_vector_l = DBL_MAX;
      n_pruned_l = DBL_MAX;
      used_error_u = 0;
    }

  };

  class MultiTreeStat {

   public:
    MultiTreeQueryPostponed postponed;
    
    MultiTreeQuerySummary summary;

    //GaussianKernelAux::TFarFieldExpansion farfield_expansion;

    //GaussianKernelAux::TLocalExpansion local_expansion;
    
    OT_DEF_BASIC(MultiTreeStat) {
      OT_MY_OBJECT(postponed);
      OT_MY_OBJECT(summary);
      //OT_MY_OBJECT(farfield_expansion);
      //OT_MY_OBJECT(local_expansion);
    }
    
  public:
    
    void FinalPush(MultiTreeStat &child_stat) {
      //child_stat.postponed.ApplyPostponed(postponed);
      //local_expansion.TranslateToLocal(child_stat.local_expansion);
    }
    
    void SetZero() {
      postponed.SetZero();
      summary.SetZero();
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count) {
      postponed.Init();
      SetZero();
    }
    
    void Init(const Matrix& dataset, index_t &start, index_t &count,
	      const MultiTreeStat& left_stat, 
	      const MultiTreeStat& right_stat) {
      postponed.Init();
      SetZero();
    }
    
    template<typename TKernelAux>
    void Init(const TKernelAux &kernel_aux_in) {
      //farfield_expansion.Init(kernel_aux_in);
      //local_expansion.Init(kernel_aux_in);
    }
    
    template<typename TBound, typename TKernelAux>
    void Init(const TBound &bounding_primitive,
	      const TKernelAux &kernel_aux_in) {
      
      // Initialize the center of expansions and bandwidth for series
      // expansion.
      //Vector bounding_box_center;
      //Init(kernel_aux_in);
      //bounding_primitive.CalculateMidpoint(&bounding_box_center);
      //(farfield_expansion.get_center())->CopyValues(bounding_box_center);
      //(local_expansion.get_center())->CopyValues(bounding_box_center);
      
      // Reset the postponed quantities to zero.
      SetZero();
    }
  };

  class MultiTreeQueryResult {
   public:

    Vector l1_norm_positive_force_vector_l;

    /** @brief Each column is a force vector for each particle.
     */    
    Matrix positive_force_vector_e;
    
    Matrix negative_force_vector_e;

    Vector l1_norm_negative_force_vector_u;

    Matrix force_vector_e;

    OT_DEF_BASIC(MultiTreeQueryResult) {
      OT_MY_OBJECT(l1_norm_positive_force_vector_l);
      OT_MY_OBJECT(positive_force_vector_e);
      OT_MY_OBJECT(negative_force_vector_e);
      OT_MY_OBJECT(l1_norm_negative_force_vector_u);
      OT_MY_OBJECT(force_vector_e);
    }

   public:
    
    void ApplyPostponed(const MultiTreeQueryPostponed &postponed_in, 
			index_t q_index) {
      
      l1_norm_positive_force_vector_l[q_index] += 
	postponed_in.l1_norm_positive_force_vector_l;
      la::AddTo(3, postponed_in.positive_force_vector_e.ptr(),
		positive_force_vector_e.GetColumnPtr(q_index));
      la::AddTo(3, postponed_in.negative_force_vector_e.ptr(),
		negative_force_vector_e.GetColumnPtr(q_index));
      l1_norm_negative_force_vector_u[q_index] +=
	postponed_in.l1_norm_negative_force_vector_u;
    }

    void Init(int num_queries) {
      l1_norm_positive_force_vector_l.Init(num_queries);
      positive_force_vector_e.Init(3, num_queries);
      negative_force_vector_e.Init(3, num_queries);
      l1_norm_negative_force_vector_u.Init(num_queries);
      force_vector_e.Init(3, num_queries);

      SetZero();
    }

    void PostProcess(index_t q_index) {
      la::AddOverwrite(3, positive_force_vector_e.GetColumnPtr(q_index),
		       negative_force_vector_e.GetColumnPtr(q_index),
		       force_vector_e.GetColumnPtr(q_index));
    }

    void PrintDebug(const char *output_file_name) const {
      FILE *stream = fopen(output_file_name, "w+");
      
      for(index_t q = 0; q < force_vector_e.n_cols(); q++) {

	const double *force_vector_e_column = 
	  force_vector_e.GetColumnPtr(q);

	for(index_t d = 0; d < 3; d++) {
	  fprintf(stream, "%g ", force_vector_e_column[d]);
	}

	fprintf(stream, "%g %g", l1_norm_positive_force_vector_l[q],
		l1_norm_negative_force_vector_u[q]);
	fprintf(stream, "\n");
      }
      
      fclose(stream);
    }

    void SetZero() {
      l1_norm_positive_force_vector_l.SetZero();
      positive_force_vector_e.SetZero();
      negative_force_vector_e.SetZero();
      l1_norm_negative_force_vector_u.SetZero();
      force_vector_e.SetZero();
    }
  };

  /** @brief Defines the global variable for the Axilrod-Teller force
   *         computation.
   */
  class MultiTreeGlobal {

   public:
    
    /** @brief The kernel object.
     */
    AxilrodTellerForceKernelAux kernel_aux;

    /** @brief The chosen indices.
     */
    ArrayList<index_t> chosen_indices;

   public:

    void Init() {

      kernel_aux.Init();
      chosen_indices.Init(AxilrodTellerForceProblem::order);
    }

  };

  /** @brief The order of interaction is 3-tuple problem.
   */
  static const int order = 3;

  template<typename MultiTreeGlobal, typename Tree>
  static bool ConsiderTupleExact(const MultiTreeGlobal &globals,
				 ArrayList<Tree *> &nodes) {
    
    // Need to fill this out...
    return false;
  }

};

#endif
