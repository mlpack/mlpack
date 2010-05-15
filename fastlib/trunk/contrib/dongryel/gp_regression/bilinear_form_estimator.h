/** @author Dongryeol Lee
 *
 *  @file bilinear_form_estimator.h
 */

#ifndef FASTLIB_CONTRIB_DONGRYEL_GP_REGRESSION_BILINEAR_FORM_ESTIMATOR_H
#define FASTLIB_CONTRIB_DONGRYEL_GP_REGRESSION_BILINEAR_FORM_ESTIMATOR_H

#include <vector>
#include "fastlib/la/matrix.h"
#include "fastlib/la/uselapack.h"
#include "linear_operator.h"

namespace fl {
namespace ml {

class SquareRootTransformation {
  public:
    static double Transform(double val_in) {
      return sqrt(val_in);
    }
};

class IdentityTransformation {
  public:
    static double Transform(double val_in) {
      return val_in;
    }
};

class InverseTransformation {
  public:
    static double Transform(double val_in) {
      return 1.0 / val_in;
    }
};

class LogTransformation {
  public:
    static double Transform(double val_in) {
      return log(val_in);
    }
};

template<typename TransformationType>
class BilinearFormEstimator {

  private:

    class TridiagonalLinearOperator {
      private:
        const std::vector<double> *diagonal_entries_;
        const std::vector<double> *offdiagonal_entries_;

      public:

        int n_rows() const;

        int n_cols() const;

        double get(int row, int col) const;

        const std::vector<double> *diagonal_entries() const;

        TridiagonalLinearOperator(
          const std::vector<double> &diagonal_entries_in,
          const std::vector<double> &offdiagonal_entries_in);

        int Apply(const Epetra_MultiVector &vecs,
                  Epetra_MultiVector &prods) const;

        void PrintDebug(const char *name = "", FILE *stream = stderr) const;
    };

  private:

#ifdef EPETRA_MPI
    Epetra_MpiComm comm_;
#else
    Epetra_SerialComm comm_;
#endif

    Anasazi::LinearOperator *linear_operator_;

    const Epetra_Map *map_;

  private:
    void AddExpert_(double scalar,
                    const Epetra_MultiVector &source,
                    Epetra_MultiVector *destination) const;

    void Scale_(double scalar,
                const Epetra_MultiVector &source,
                Epetra_MultiVector *destination) const;

    double Dot_(const Epetra_MultiVector &first_vec,
                const Epetra_MultiVector &second_vec) const;

    template<typename LinearOperatorType>
    double ComputeQuadraticForm_(
      int num_iterations,
      const GenVector<double> &starting_vector,
      const LinearOperatorType &linear_operator_in,
      const Epetra_Map &map_in,
      int level,
      bool *broke_down);

  public:

    BilinearFormEstimator();

    Anasazi::LinearOperator *linear_operator();

    void Init(Anasazi::LinearOperator *linear_operator_in);

    double Compute(
      const GenVector<double> &left_argument,
      const GenVector<double> &right_argument,
      bool naive_compute);

    double Compute(const GenVector<double> &argument);

    double NaiveCompute(const GenVector<double> &argument);
};
};
};

#endif
