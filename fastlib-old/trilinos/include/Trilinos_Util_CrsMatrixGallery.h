// @HEADER
// ***********************************************************************
// 
//                 TriUtils: Trilinos Utilities Package
//                 Copyright (2001) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ***********************************************************************
// @HEADER

#ifndef __TRILINOS_UTILS_GALLERY_H
#define __TRILINOS_UTILS_GALLERY_H

class Epetra_Comm;
class Epetra_Map;
class Epetra_BlockMap;
class Vector;
#include "Epetra_CrsMatrix.h"
#include "Epetra_VbrMatrix.h"
class Epetra_Export;
class Epetra_LinearProblem;
#include <string>
#include <vector>
#include "Trilinos_Util_CommandLineParser.h"

namespace Trilinos_Util {

class CrsMatrixGallery 
{
public:

  //@{ \name Constructors/Destructor.
  //! Triutils_Gallery Constructor.
  /*! Creates a Triutils_Gallery instance.

  The first parameter is the name of the matrix. We refer to the Trilinos
  Tutorial for a detailed description of available matrices.

  \note The matrix name can be empty (""), and set later using, for example,
  Set("matrix_name","laplace_2d");
  
  An example of program using this class is reported below.

  \code
int main(int argc, char *argv[])
{

#ifdef HAVE_MPI
  MPI_Init(&argc,&argv);
  Epetra_MpiComm Comm (MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif

  // create an Epetra matrix reading an H/B matrix
  Trilinos_Util_CrsMatrixGallery Gallery("hb", Comm);

  // set the name of the matrix
  Gallery.Set("matrix name", "bcsstk14.rsa");
  
  Epetra_CrsMatrix * A;
  Epetra_Vector * ExactSolution;
  Epetra_Vector * RHS;
  Epetra_Vector * StartingSolution;

  // at this point the matrix is read from file
  A = Gallery.GetMatrix();
  ExactSolution = Gallery.GetExactSolution();

  // at this point the RHS is allocated and filled
  RHS = Gallery.GetRHS();
  StartingSolution = Gallery.GetStartingSolution();
  
  // create linear problem
  Epetra_LinearProblem Problem(A,StartingSolution,RHS);
  // create AztecOO instance
  AztecOO Solver(Problem);

  Solver.SetAztecOption( AZ_precond, AZ_dom_decomp );  
  Solver.Iterate(1000,1E-9);

  // compute residual
  double residual;
  
  Gallery.ComputeResidual(&residual);
  if( Comm.MyPID()==0 ) cout << "||b-Ax||_2 = " << residual << endl;
  
  Gallery.ComputeDiffBetweenStartingAndExactSolutions(&residual);
  if( Comm.MyPID()==0 ) cout << "||x_exact - x||_2 = " << residual << endl;

 #ifdef HAVE_MPI
  MPI_Finalize() ;
#endif

return 0 ;
  } 
  \endcode

  Class CommandLineParser can be used as well. In this case, one may
  decide to use the following:
  \code
  Trilinos_Util::CommandLineParser CLP(argc,argv);
  // set a problem with no matrix name
  Trilinos_Util::CrsMatrixGallery Gallery("", Comm);
  // read parameters and settings from the shell line
  G.Set(CLP);
  // continue with your code...
  \endcode
  
  \param In
  comm - Epetra communicator
  */
  CrsMatrixGallery( const string name, const Epetra_Comm & comm );

  //! Creates an Triutils_Gallery object using a given map.
  /*! Create a Triutils_Gallery object using an Epetra_Map.
    Problem size must match the elements in map.

    \param In
    name - definition of the problem to be created.

    \param In
    map - Epetra_Map
  */
  CrsMatrixGallery( const string name, const Epetra_Map & map );
  
  //! Triutils_Gallery destructor
  ~CrsMatrixGallery();

  //@}

  //@{ \name Setting methods
 
  //! Sets a gallery options using an interger value.
  int Set(const string parameter, const int value);

  //!  Sets a gallery options using a C++ string .
  int Set(const string parameter, const string value );

  //! Sets a gallery options using an double value.
  int Set(const string parameter, const double value);

  //! Sets a gallery options using an Epetra_Vector.
  /*! Sets a gallery options using an Epetra_Vector. The Epetra_Vector
  is copied into internal structures, and freed by the destructor.
  */
  int Set(const string parameter, const Epetra_Vector & value);

  //! Sets gallery options using values passed from the shell
  int Set(Trilinos_Util::CommandLineParser & CLP);

  //@}

  //@{ \name Extraction methods.

  //! Returns a pointer to the CrsMatrix.
  Epetra_CrsMatrix * GetMatrix();

  Epetra_CrsMatrix & GetMatrixRef();

  //! Returns a pointer to the exact solution.
  /*! Returns a pointer to the exact solution.
    
    Some choices are available to define the exact solution, using
    Set("exact solution", value). value can be:
    - constant: the exact solution vector is made up of 1's.
    - random: a random solution vector
    - linear: value at node i is defined as alpha*i. The double value
    alpha can be set via Set("alpha",DoubleVal).
  */
  Epetra_MultiVector * GetExactSolution();

  //! Returns a pointer to the starting solution (typically, for HB problems).
  /*! Returns a pointer to the starting solution. This is typically used
    while reading a HB problem. However, the user can set a starting
    solution using Set("starting solution", "value"). Value can be
    - zero
    - random 
  */
  Epetra_MultiVector * GetStartingSolution();
  
  //! Returns a pointer to the rhs corresponding to the selected exact solution.
  Epetra_MultiVector * GetRHS();

  //! Returns a pointer the internally stored Map.
  const Epetra_Map * GetMap();

  const Epetra_Map & GetMapRef();  

  // ==================== //
  // LINEAR PROBLEM STUFF //
  // ==================== //

  //! Returns a pointer to Epetra_LinearProblem
  Epetra_LinearProblem * GetLinearProblem();

  //! Computes the 2-norm of the residual
  void ComputeResidual(double* residual);

  //! Computes the 2-norm of the difference between the starting solution and the exact solution
  void ComputeDiffBetweenStartingAndExactSolutions(double* residual);

  //! Print out matrix and vectors
  void PrintMatrixAndVectors(ostream & os);

  void PrintMatrixAndVectors();

  //! Get pointers to double vectors containing coordinates of points.
  void GetCartesianCoordinates(double * & x, double * & y, double * & z);
  
  //! Print out detailed information about the problem at hand
  friend ostream & operator << (ostream& os,
				const Trilinos_Util::CrsMatrixGallery & G );
				
  //! Print matrix on file in MATLAB format
  int WriteMatrix( const string & FileName, const bool UseSparse=true );
  
  //@}

protected:

  //@{ \name Creation methods.
  
  //! Creates a map.
  /*! Creates an Epetra_Map. Before calling this function, the problem
  size must have been specified.

  CreateMap() allows some different maps. The type of map is set using
  Set("map",value). Value is a string, defined as:
  - linear: Creates a linear map. Elements are divided into continuous
  chunks among the processors.

  - box: used for problems defined on cartesian grids over a square. The
  nodes is subdivided into mx x my subdomains. mx and my are specified
  via Set("mx",IntValue) and Set("my",IntValue).

  - interlaces: elements are subdivided so that element i is assigned to
  process i%NumProcs.

  - random: assign each node to a random process
  
  - greedy: (only for HB matrices) implements a greedy algorithm to
    decompose the graph of the HB matrix among the processes
    
  */
  void CreateMap();
  
  //! Creates the CrdMatrix.
  void CreateMatrix();

  //! Creates the exact solution.
  void CreateExactSolution();

  //! Creates the starting solution.
  void CreateStartingSolution();

  //! Create the RHS corresponding to the desired exact solution.  
  void CreateRHS();
  
  // Create an identity matrix.
  void CreateEye();

  // Creates a diagonal matrix. Elements on the diagonal are called `a'.
  void CreateMatrixDiag();
    
  // Creates a tridiagonal matrix. Elements on the diagonal are called `a',
  // elements on the sub-diagonal 'b', and on the super-diagonal 'c'.
  void CreateMatrixTriDiag();
  
  // Create a matrix for a Laplacian in 1D
  void CreateMatrixLaplace1d();
  
  void CreateMatrixLaplace1dNeumann();
  
  void CreateMatrixCrossStencil2d();

  void CreateMatrixCrossStencil2dVector();

  void CreateMatrixLaplace2d();

  void CreateMatrixLaplace2d_BC();

  void CreateMatrixLaplace2d_9pt();

  void CreateMatrixStretched2d();

  void CreateMatrixRecirc2d();

  void CreateMatrixRecirc2dDivFree();
  
  void CreateMatrixLaplace2dNeumann();
  
  void CreateMatrixUniFlow2d();
  
  void CreateMatrixLaplace3d();

  void CreateMatrixCrossStencil3d();

  void CreateMatrixCrossStencil3dVector();

  void CreateMatrixLehmer();

  void CreateMatrixMinij();

  void CreateMatrixRis();

  void CreateMatrixHilbert();

  void CreateMatrixJordblock();

  void CreateMatrixCauchy();

  void CreateMatrixFiedler();

  void CreateMatrixHanowa();

  void CreateMatrixKMS();
  
  void CreateMatrixParter();

  void CreateMatrixPei();

  void CreateMatrixOnes();

  void CreateMatrixVander();
  
  // read an HB matrix. This function requires other Trilinos util files
  void ReadMatrix();

  // returns the neighbors of a given node. The node is supposed to be on
  // a 2D Cartesian grid 
  void  GetNeighboursCartesian2d( const int i, const int nx, const int ny,
				  int & left, int & right, 
				  int & lower, int & upper);
  // returns the neighbors of a given node. The node is supposed to be on
  // a 3D Cartesian grid   
  void  GetNeighboursCartesian3d( const int i, const int nx, const int ny, const int nz,
				  int & left, int & right, int & lower, int & upper,
				  int & below, int & above );

  // put to NULL or default values all internal data
  void ZeroOutData();

  void SetupCartesianGrid2D();

  void SetupCartesianGrid3D();

  void ExactSolQuadXY(double x, double y, double & u);
  
  void ExactSolQuadXY(double x, double y, double & u,
		      double & ux, double & uy,
		      double & uxx, double & uyy);
  
  
  //@}
  
  // ======================== //
  // I N T E R N A L  D A T A //
  // ======================== //
  
  const Epetra_Comm * comm_;

  // matrix and vectors (scalar)
  Epetra_CrsMatrix * matrix_;
  Epetra_MultiVector * ExactSolution_;
  Epetra_MultiVector * StartingSolution_;
  Epetra_MultiVector * rhs_;
  Epetra_Map * map_;

  // linear problem
  Epetra_LinearProblem * LinearProblem_;

  // information about the problem to generate
  string name_;
  int NumGlobalElements_;
  int NumMyElements_;
  int * MyGlobalElements_;
  string MapType_;
  bool ContiguousMap_;
  std::vector<int> MapMap_;
  string ExactSolutionType_;
  string StartingSolutionType_;
  string ExpandType_;
  string RhsType_;
  
  // parameters
  int nx_, ny_, nz_;
  int mx_, my_, mz_;

  double lx_, ly_, lz_;
  
  int NumPDEEqns_;
  int NumVectors_;
  
  Epetra_Vector * VectorA_, * VectorB_, * VectorC_, * VectorD_, * VectorE_, *VectorF_, * VectorG_;
  
  double a_, b_, c_, d_, e_, f_, g_;
  double alpha_, beta_, gamma_, delta_;
  double conv_, diff_, source_;
  double epsilon_;
  
  string FileName_;

  // others
  string ErrorMsg;
  string OutputMsg;
  bool verbose_;
  
};

// ========================= //
// extension to VBR matrices //
// ==========================//

class VbrMatrixGallery : public CrsMatrixGallery
{

public:

  VbrMatrixGallery(const string name, const Epetra_Map & map) :
    CrsMatrixGallery(name,map),
    VbrMatrix_(0),
    VbrExactSolution_(0),
    VbrStartingSolution_(0),
    VbrRhs_(0),
    BlockMap_(0),
    MaxBlkSize_(1),
    VbrLinearProblem_(0)
   {} ;

  VbrMatrixGallery(const string name, const Epetra_Comm & Comm) :
    CrsMatrixGallery(name,Comm),
    VbrMatrix_(0),
    VbrExactSolution_(0),
    VbrStartingSolution_(0),
    VbrRhs_(0),
    BlockMap_(0),
    MaxBlkSize_(1),
    VbrLinearProblem_(0)
  {} ;

  ~VbrMatrixGallery(); 
  
  // ========= //
  // VBR STUFF //
  // ========= //
  
  //! Returns a pointer the internally stored BlockMap.
  const Epetra_BlockMap * GetBlockMap();

  const Epetra_BlockMap & GetBlockMapRef();
  
  //! Returns a VbrMatrix, starting from the CsrMatrix.
  /*! Returns a VbrMatrix, starting from the CsrMatrix. This vbr matrix
    is formally equivalent to the CrsMatrix returned by
    GetMatrix(). However, each node of the CrsMatrix is replicated
    num_PDE_eqns times (this value is passed in input, or set via Set("num pde
    eqns",IntValue)).
  */
  Epetra_VbrMatrix * GetVbrMatrix(const int NumPDEEqns);

  //! Returns a VbrMatrix, starting from the CsrMatrix.
  Epetra_VbrMatrix * GetVbrMatrix();

  Epetra_VbrMatrix & GetVbrMatrixRef();

  //! Returns a pointer to the RHS for the selected Vbr exact solution
  /*!  Returns a pointer to the RHS  corresponding to the selected exact solution to the linear systems defined by the Epetra_VbrMatrix.
   */
  Epetra_MultiVector * GetVbrRHS();

  //! Returns a pointer to the selected Vbr exact solution
  Epetra_MultiVector * GetVbrExactSolution();

  //! Returns a pointer to the starting solution for Vbr problems
  Epetra_MultiVector * GetVbrStartingSolution();


  // create the Vbr matrix. 
  void CreateVbrMatrix(void);  

  //! Returns a pointer to Epetra_LinearProblem for VBR
  Epetra_LinearProblem * GetVbrLinearProblem();

  //! Computes the 2-norm of the residual for the VBR problem
  void ComputeResidualVbr(double* residual);

  //! Computes the 2-norm of the difference between the starting solution and the exact solution for the VBR problem  
  void ComputeDiffBetweenStartingAndExactSolutionsVbr(double* residual);

  //! Print out Vbr matrix and vectors
  void PrintVbrMatrixAndVectors(ostream & os);

  void PrintVbrMatrixAndVectors();

protected:

  // Creates a block map, based on map, wich NumPDEEqns equations on each node.
  void CreateBlockMap(void);
  
  //! Creates the exact solution for a Epetra_VbrMatrix.
  void CreateVbrExactSolution(void);

  //! Creates the starting solution for Vbr.
  void CreateVbrStartingSolution();
  
  //!  Create the RHS corresponding to the desired exact solution for the Vbr problem.
  void CreateVbrRHS();

  // matrix and vectors (vbr)
  Epetra_VbrMatrix * VbrMatrix_;
  Epetra_MultiVector * VbrExactSolution_;
  Epetra_MultiVector * VbrStartingSolution_;
  Epetra_MultiVector * VbrRhs_;
  Epetra_BlockMap * BlockMap_;
  int MaxBlkSize_;

  // linear problem  
  Epetra_LinearProblem * VbrLinearProblem_;

};

} // namespace Trilinos_Util
#endif
