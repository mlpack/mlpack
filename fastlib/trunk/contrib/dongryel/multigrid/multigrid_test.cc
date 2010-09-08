/** @file multigrid_test.cc
 *
 *  @brief The test driver for the multigrid solver.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "ml_include.h"

#include "Epetra_ConfigDefs.h"

#include "Epetra_CrsMatrix.h"
#include "Epetra_Map.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Time.h"

//#ifdef ML_MPI
#include "Epetra_MpiComm.h"
#include "mpi.h"
//#else
//#include "Epetra_SerialComm.h"
//#endif

#include "ml_epetra_operator.h"
#include "ml_epetra_utils.h"

int main(int argc, char *argv[]) {

  int myPid = 0;
  int numProc = 1;

#ifdef ML_MPI
  // Initialize MPI
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProc);
  MPI_Comm_rank(MPI_COMM_WORLD, &myPid);
  Epetra_MpiComm Comm( MPI_COMM_WORLD );
#else
  Epetra_SerialComm Comm;
#endif

  int numElePerDirection = 20*numProc;
  int numNodes = (numElePerDirection - 1)*(numElePerDirection - 1);

  Epetra_Map Map(numNodes, 0, Comm);

  // Define the matrix A from the discretization of Laplace equation 
  // with homogeneous Dirichlet condition (using square Q1 elements)

  Epetra_CrsMatrix A(Copy, Map, 0);
  for (int iY = 0; iY < numElePerDirection - 1; ++iY) {
    for (int iX = 0; iX < numElePerDirection - 1; ++iX) {

      int node = iX + iY*(numElePerDirection-1);
      if (Map.LID(node) == -1)
        continue;

      double val = 8.0;
      int pos = node;
      A.InsertGlobalValues(node, 1, &val, &pos);
      if (iY > 0) {
        val = -1.0;
        pos = iX + (iY-1)*(numElePerDirection-1);
        A.InsertGlobalValues(node, 1, &val, &pos);
      }
      if (iY < numElePerDirection-2) {
        val = -1.0;
        pos = iX + (iY+1)*(numElePerDirection-1);
        A.InsertGlobalValues(node, 1, &val, &pos);
      }

      // Insert values on the left side of stencil
      if (iX > 0) {
        val = -1.0;
        pos = iX-1 + iY*(numElePerDirection-1);
        A.InsertGlobalValues(node, 1, &val, &pos);
        if (iY > 0) {
          val = -1.0;
          pos = iX-1 + (iY-1)*(numElePerDirection-1);
          A.InsertGlobalValues(node, 1, &val, &pos);
        }
        if (iY < numElePerDirection-2) {
          val = -1.0;
          pos = iX-1 + (iY+1)*(numElePerDirection-1);
          A.InsertGlobalValues(node, 1, &val, &pos);
        }
      }

      // Insert values on the right side of stencil
      if (iX < numElePerDirection - 2) {
        val = -1.0;
        pos = iX+1 + iY*(numElePerDirection-1);
        A.InsertGlobalValues(node, 1, &val, &pos);
        if (iY > 0) {
          val = -1.0;
          pos = iX+1 + (iY-1)*(numElePerDirection-1);
          A.InsertGlobalValues(node, 1, &val, &pos);
        }
        if (iY < numElePerDirection-2) {
          val = -1.0;
          pos = iX+1 + (iY+1)*(numElePerDirection-1);
          A.InsertGlobalValues(node, 1, &val, &pos);
        }
      }

    } // for (int iX = 0; iX < numElePerDirection - 1; ++iX)
  } // for (int iY = 0; iY < numElePerDirection - 1; ++iY)
  A.FillComplete();
  A.OptimizeStorage();

  // Define the ML preconditioner
  int AMG_NLevels = 10;
  ML *ml_handle;
  ML_Create(&ml_handle, AMG_NLevels);

  ML_Set_PrintLevel(10);

  EpetraMatrix2MLMatrix(ml_handle, 0, &A);

  ML_Aggregate *agg_object;
  ML_Aggregate_Create(&agg_object);
  ML_Aggregate_Set_MaxCoarseSize(agg_object, 10);
  ML_Aggregate_Set_Threshold(agg_object, 0.0);
  ML_Aggregate_Set_Threshold(agg_object, 0.0);
  AMG_NLevels = ML_Gen_MGHierarchy_UsingAggregation(ml_handle, 0, ML_INCREASING, agg_object);

  // Set a smoother for the AMG method
  int degree = 2;
  for (int j = 0; j < AMG_NLevels; j++)
    ML_Gen_Smoother_Cheby(ml_handle, j, ML_BOTH, 30., degree);

  ML_Gen_Solver(ml_handle, ML_MGV, 0, AMG_NLevels-1);

  ML_Epetra::MultiLevelOperator Prec(ml_handle, Comm, Map, Map);

#ifdef ML_MULTIPLE_RHS_BLOCK_FACTOR
  // Check the multiple right hand side capabilities
  Epetra_Time MyWatch(Comm);
  for (int iBlock = 1; iBlock < 21; ++iBlock) {

    // Define the block vectors
    Epetra_MultiVector X(Map, iBlock);
    X.Random();
    Epetra_MultiVector Y0(Map, iBlock);
    Y0.PutScalar(0.0);
    Epetra_MultiVector Y1(Map, iBlock);
    Y1.PutScalar(0.0);
    Epetra_MultiVector Y2(Map, iBlock);
    Y2.PutScalar(0.0);

    // Apply the preconditioner one vector at a time 
    double t0 = - MyWatch.WallTime();
    Prec.ApplyInverse_WKC(X, Y0);
    t0 += MyWatch.WallTime();

    // Apply the preconditioner per block (Blocking factor defined at compilation)
    double t1 = - MyWatch.WallTime();
    Prec.ApplyInverse(X, Y1);
    t1 += MyWatch.WallTime();

    // Apply the preconditioner per block (Maximum blocking factor)
    double t2 = - MyWatch.WallTime();
    Prec.ApplyInverse(X, Y2);
    Prec.ApplyInverse(X, Y2);
    t2 += MyWatch.WallTime();

    // Check the norms
    double *y0Norm = new double[iBlock];
    Y0.Norm2(y0Norm);

    Y1.Update(1.0, Y0, -1.0);
    double *y1Norm = new double[iBlock];
    Y1.Norm2(y1Norm);
    double maxError1 = 0.0;
    for (int j = 0; j < iBlock; ++j) {
      if (y1Norm[j] > maxError1*y0Norm[j])
        maxError1 = y1Norm[j]/y0Norm[j];
    } 

    Y2.Update(1.0, Y0, -1.0);
    double *y2Norm = new double[iBlock];
    Y2.Norm2(y2Norm);
    double maxError2 = 0.0;
    for (int j = 0; j < iBlock; ++j) {
      if (y2Norm[j] > maxError2*y0Norm[j])
        maxError2 = y2Norm[j]/y0Norm[j];
    } 


    delete[] y0Norm; 
    delete[] y1Norm; 
    delete[] y2Norm; 

    if (myPid == 0) {
      cout << " --- Block size " << iBlock << " --- " << endl;
      cout << " >> Blocking factor " << ML_MULTIPLE_RHS_BLOCK_FACTOR;
      cout << " ... Error " << maxError1 << " Speedup " << t0/t1 << endl;
      cout << " >> Blocking factor " << iBlock;
      cout << " ... Error " << maxError2 << " Speedup " << t0/t2 << endl;
      cout << endl;
    }

  }
#else
  if (Comm.MyPID() == 0)
    cout << "Please configure ML with the option --with-ml_multiple_rhs=NN,\nwhere NN is the blocking factor.";
#endif //ifdef ML_MULTIPLE_RHS_BLOCK_FACTOR
  ML_Aggregate_Destroy(&agg_object);
  ML_Destroy(&ml_handle);

#ifdef ML_MPI
  MPI_Finalize();
#endif

  return 0;
}
