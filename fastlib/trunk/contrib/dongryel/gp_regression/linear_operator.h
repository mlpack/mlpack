
#ifndef FASTLIB_CONTRIB_DONGRYEL_FASTLIB_TRILINOS_WRAPPERS_LINEAR_OPERATOR_H
#define FASTLIB_CONTRIB_DONGRYEL_FASTLIB_TRILINOS_WRAPPERS_LINEAR_OPERATOR_H

#undef F77_FUNC
#undef LI
#include "trilinos/AnasaziEpetraAdapter.hpp"
#include "trilinos/AnasaziBasicEigenproblem.hpp"
#include "trilinos/AnasaziBlockKrylovSchurSolMgr.hpp"
#include "trilinos/AnasaziBasicSort.hpp"
#include "trilinos/AztecOO.h"
#include "trilinos/Epetra_BlockMap.h"
#include "trilinos/Epetra_CrsMatrix.h"
#include "trilinos/Epetra_DataAccess.h"
#include "trilinos/Epetra_LinearProblem.h"
#include "trilinos/Epetra_Map.h"
#include "trilinos/Epetra_MultiVector.h"
#include "trilinos/Epetra_Operator.h"

#ifdef EPETRA_MPI
#include "trilinos/Epetra_MpiComm.h"
#else
#include "trilinos/Epetra_SerialComm.h"
#endif

#include "trilinos/Epetra_Vector.h"
#undef F77_FUNC
#include "fastlib/la/matrix.h"
#include <vector>

namespace Anasazi {

class LinearOperator: public virtual Epetra_Operator {

  protected:

#ifdef EPETRA_MPI
    const Epetra_MpiComm *comm_;
#else
    const Epetra_SerialComm *comm_;
#endif

    const Epetra_Map *map_;

  public:

    virtual ~LinearOperator() {
    }

    LinearOperator() {
      comm_ = NULL;
      map_ = NULL;
    }

#ifdef EPETRA_MPI
    LinearOperator(const Epetra_MpiComm &comm_in,
                   const Epetra_Map &map_in) {
      comm_ = &comm_in;
      map_ = &map_in;
    }
#else
    LinearOperator(const Epetra_SerialComm &comm_in,
                   const Epetra_Map &map_in) {
      comm_ = &comm_in;
      map_ = &map_in;
    }
#endif

    virtual int Apply(const Epetra_MultiVector &vec,
                      Epetra_MultiVector &prod) const = 0;

    int SetUseTranspose(bool use_transpose) {
      return -1;
    }

    int ApplyInverse(const Epetra_MultiVector &X,
                     Epetra_MultiVector &Y) const {
      return -1;
    }

    double NormInf() const {
      return -1;
    }

    const char *Label() const {
      return "Generic linear operator";
    }

    bool UseTranspose() const {
      return false;
    }

    bool HasNormInf() const {
      return false;
    }

    const Epetra_Comm &Comm() const {
      return *comm_;
    }

    const Epetra_Map &OperatorDomainMap() const {
      const Epetra_Map &map_reference = *map_;
      return map_reference;
    }

    const Epetra_Map &OperatorRangeMap() const {
      const Epetra_Map &map_reference = *map_;
      return map_reference;
    }

    void PrintDebug(const char *name = "", FILE *stream = stderr) const {
      fprintf(stream, "----- MATRIX ------: %s\n", name);
      for (int r = 0; r < this->n_rows(); r++) {
        for (int c = 0; c < this->n_cols(); c++) {
          fprintf(stream, "%+3.3f ", this->get(r, c));
        }
        fprintf(stream, "\n");
      }
    }

    virtual int n_rows() const = 0;

    virtual int n_cols() const = 0;

    virtual double get(int row, int col) const = 0;
};
};

#endif
