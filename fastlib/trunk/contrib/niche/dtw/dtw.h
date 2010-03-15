#ifndef DTW_H
#define DTW_H

#define INSIDE_DTW_IMPL_H

#include "fastlib/fastlib.h"

void LoadTimeSeries(const char* filename, Vector* p_time_series);

double ComputeDTWAlignmentScore(int b,
				const Matrix &x, const Matrix &y);

double ComputeDTWAlignmentScore(int b,
				const Vector &x, const Vector &y, 
  				ArrayList< GenVector<int> >* p_best_path);


#include "dtw_impl.h"
#undef INSIDE_DTW_IMPL_H

#endif /* DTW_H */
