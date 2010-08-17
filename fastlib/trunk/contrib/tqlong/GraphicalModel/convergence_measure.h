#ifndef CONVERGENCE_MEASURE_H
#define CONVERGENCE_MEASURE_H

#include "gm.h"

BEGIN_GRAPHICAL_MODEL_NAMESPACE;

/** Convergence measure based on number of iterations, changes, etc.
  */
struct ConvergenceMeasure
{
  int criteria_;
  int iter_;
  double changeTolerance_;

  // Criteria mask
  static const int Iter;
  static const int Change;

  ConvergenceMeasure(int criteria = Iter, int maxIter = 10, double changeTolerance = 1e-5)
    : criteria_(criteria), iter_(maxIter), changeTolerance_(changeTolerance)
  {
  }

  bool operator > (ConvergenceMeasure cvm) const
  {
    if ((criteria_ & Iter) && iter_ >= cvm.iter_)
      cout << "Maximum iteration reached" << endl;
    if ((criteria_ & Change) && changeTolerance_ <= cvm.changeTolerance_)
      cout << "Total change smaller than tolerance (" << cvm.changeTolerance_ << ")" << endl;
    return ((criteria_ & Iter) && iter_ >= cvm.iter_) ||
           ((criteria_ & Change) && changeTolerance_ <= cvm.changeTolerance_);
  }
};

END_GRAPHICAL_MODEL_NAMESPACE;

#endif // CONVERGENCE_MEASURE_H
