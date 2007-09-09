/**
 * @file series_expansion.h
 *
 * The header file for the series expansion.
 */

#ifndef SERIES_EXPANSION
#define SERIES_EXPANSION

#include "fastlib/fastlib.h"

#include "series_expansion_aux.h"

/**
 * Series expansion class.
 */
class SeriesExpansion {
  FORBID_COPY(SeriesExpansion);
  
 public:
  enum KernelType { GAUSSIAN, EPANECHNIKOV };

  enum ExpansionType { FARFIELD, LOCAL };

 private:

  /** The type of kernel */
  KernelType kernel_type_;

  /** The type of coefficients: far-field or local */
  ExpansionType expansion_type_;
  
  /** The center of the expansion */
  Vector center_;

  /** bandwidth squared */
  double bwsqd_;

  /** The coefficients */
  Vector coeffs_;

  /** order */
  int order_;

 public:

  SeriesExpansion() {}
  
  ~SeriesExpansion() {}

  // getters and setters
  
  /** Get the coefficients */
  double get_bwsqd() const { return bwsqd_; }

  const Vector& get_center() const { return center_; }

  const Vector& get_coeffs() const { return coeffs_; }

  ExpansionType get_expansion_type() const { return expansion_type_; }

  /** Get the kernel type */
  KernelType get_kernel_type() const { return kernel_type_; }

  /** Get the approximation order */
  int get_order() const { return order_; }

  // interesting functions...

  /**
   * Computes the far-field coefficients for the given data
   */
  void ComputeFarFieldCoeffs(const Matrix& data, const Vector& weights,
			     const ArrayList<int>& rows, int order,
			     const SeriesExpansionAux& sea);

  /**
   * Computes the local coefficients for the given data
   */
  void ComputeLocalCoeffs(const Matrix& data, const Vector& weights,
			  const ArrayList<int>& rows, int order,
			  const SeriesExpansionAux& sea);

  /**
   * Evaluates the far-field coefficients at the given point
   */
  double EvaluateFarField(Matrix* data=NULL, int row_num=-1,
			  Vector* point=NULL, 
			  SeriesExpansionAux* sea=NULL);

  /**
   * Initializes the current SeriesExpansion object with the given
   * center.
   */
  void Init(KernelType kernel_type, ExpansionType expansion_type,
	    const Vector& center, int max_total_num_coeffs, double bwsqd);

  /**
   * Prints out the series expansion represented by this object.
   */
  void PrintDebug(const char *name="", FILE *stream=stderr) const;

  /**
   * Far-field to Far-field translation operator: translates the given
   * far-field expansion to the new center
   */
  void TransFarToFar(const SeriesExpansion &se, const SeriesExpansionAux &sea);

  /**
   * Far-field to local translation operator: translates the given far
   * expansion to the local expansion at the new center.
   */
  void TransFarToLocal(const SeriesExpansion &se,
		       const SeriesExpansionAux &sea);

  /**
   * Local to local translation operator: translates the given local
   * expansion to the local expansion at the new center.
   */
  void TransLocalToLocal(const SeriesExpansion &se, 
			 const SeriesExpansionAux &sea);

};

#endif
