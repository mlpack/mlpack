/** @author Dongryeol Lee
 *
 *  @file log_determinant.h
 */

#ifndef FASTLIB_CONTRIB_DONGRYEL_GP_REGRESSION_LOG_DETERMINANT_H
#define FASTLIB_CONTRIB_DONGRYEL_GP_REGRESSION_LOG_DETERMINANT_H

#include "bilinear_form_estimator.h"
#include "fastlib/la/matrix.h"

namespace fl {
namespace ml {
class LogDeterminant {

  private:

    fl::ml::BilinearFormEstimator<fl::ml::LogTransformation> bilinear_log_form_;

  private:

    void RandomVector_(GenVector<double> &v);

  public:

    LogDeterminant();

    void Init(Anasazi::LinearOperator *linear_operator_in);

    double MonteCarloCompute();

    double Compute();

    double NaiveCompute();
};
};
};

#endif
