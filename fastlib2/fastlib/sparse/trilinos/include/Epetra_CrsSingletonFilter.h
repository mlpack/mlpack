
//@HEADER
/*
************************************************************************

              Epetra: Linear Algebra Services Package 
                Copyright (2001) Sandia Corporation

Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
license for use of this work by or on behalf of the U.S. Government.

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation; either version 2.1 of the
License, or (at your option) any later version.
 
This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA
Questions? Contact Michael A. Heroux (maherou@sandia.gov) 

************************************************************************
*/
//@HEADER

#ifndef EPETRA_CRSSINGLETONFILTER_H
#define EPETRA_CRSSINGLETONFILTER_H

#include "Epetra_Object.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_MapColoring.h"
#include "Epetra_SerialDenseVector.h"
class Epetra_LinearProblem;
class Epetra_Map;
class Epetra_MultiVector;
class Epetra_Import;
class Epetra_Export;
class Epetra_IntVector;

//! Epetra_CrsSingletonFilter: A class for explicitly eliminating matrix rows and columns.

/*! The Epetra_CrsSingletonFilter class takes an existing Epetra_LinearProblem object, analyzes
    it structure and explicitly eliminates singleton rows and columns from the matrix and appropriately
    modifies the RHS and LHS of the linear problem.  The result of this process is a reduced system of equations
    that is itself an Epetra_LinearProblem object.  The reduced system can then be solved using any solver that
    is understands an Epetra_LinearProblem.  The solution for the full system is obtained by calling ComputeFullSolution().
    
    Singleton rows are defined to be rows that have a single nonzero entry in the matrix.  The equation associated with
    this row can be explicitly eliminated because it involved only one variable.  For example if row i has a single nonzero
    value in column j, call it A(i,j), we can explicitly solve for x(j) = b(i)/A(i,j), where b(i) is the ith entry of the RHS
    and x(j) is the jth entry of the LHS.

    Singleton columns are defined to be columns that have a single nonzero entry in the matrix.  The variable associated
    with this column is fully dependent, meaning that the solution for all other variables does not depend on it.  If this
    entry is A(i,j) then the ith row and jth column can be removed from the system and x(j) can be solved after the solution
    for all other variables is determined.

    By removing singleton rows and columns, we can often produce a reduced system that is smaller and far less dense, and in
    general having better numerical properties.

    The basic procedure for using this class is as follows:
<ol>
<li> Construct full problem: Construct and Epetra_LinearProblem containing the "full" matrix, RHS and LHS.  This is
     done outside of Epetra_CrsSingletonFilter class.
     Presumably, you have some reason to believe that this system may contain singletons.
<li> Construct an Epetra_CrsSingletonFilter instance:  Constructor needs no arguments.
<li> Analyze matrix: Invoke the Analyze() method, passing in the Epetra_RowMatrix object from your full linear
     problem mentioned in the first step above.
<li> Go/No Go decision to construct reduced problem:
     Query the results of the Analyze method using the SingletonsDetected() method.  This method 
     returns "true" if there were singletons found in the matrix.  You can also query any of the other methods
     in the Filter Statistics section to determine if you want to proceed with the construction of the reduced system.
<li> Construct reduced problem: 
     If, in the previous step, you determine that you want to proceed with the construction of the reduced problem,
     you should next call the ConstructReducedProblem() method, passing in the full linear problem object from the first
     step.  This method will use the information from the Analyze() method to construct a reduce problem that has
     explicitly eliminated the singleton rows, solved for the corresponding LHS values and updated the RHS.  This 
     step will also remove singleton columns from the reduced system.  Once the solution of the reduced problem is
     is computed (via any solver that understands an Epetra_LinearProblem), you should call the ComputeFullSolution()
     method to compute the LHS values assocaited with the singleton columns.
<li> Solve reduced problem: Obtain a pointer to the reduced problem using the ReducedProblem() method.
     Using the solver of your choice, solve the reduced system.
<li> Compute solution to full problem:  Once the solution the reduced problem is determined, the ComputeFullSolution()
     method will place the reduced solution values into the appropriate locations of the full solution LHS and then
     compute the values associated with column singletons.  At this point, you have a complete solution to the original
     full problem.
<li> Solve a subsequent full problem that differs from the original problem only in values: It is often the case that the
     structure of a problem will be the same for a sequence of linear problems.  In this case, the UpdateReducedProblem()
     method can be useful.  After going through the above process one time, if you have a linear problem that is structural
     \e identical to the previous problem, you can minimize memory and time costs by using the UpdateReducedProblem() 
     method, passing in the subsequent problem.  Once you have called the UpdateReducedProblem() method, you can then
     solve the reduce problem problem as you wish, and then compute the full solution as before.  The pointer generated
     by ReducedProblem() will not change when UpdateReducedProblem() is called.
</ol>
*/    

class Epetra_CrsSingletonFilter {
      
 public:

   //! @name Constructors/Destructor
  //@{ 
  //! Epetra_CrsSingletonFilter default constructor.
  Epetra_CrsSingletonFilter();

  //! Epetra_CrsSingletonFilter Destructor
  virtual ~Epetra_CrsSingletonFilter();
  //@}
  //! @name Analyze methods
  //@{ 
  //! Analyze the input matrix, removing row/column pairs that have singletons.
  /*! Analyzes the user's input matrix to determine rows and columns that should be explicitly
      eliminated to create the reduced system.  Look for rows and columns that have single entries.  
      These rows/columns
      can easily be removed from the problem.  
      The results of calling this method are two MapColoring objects accessible via RowMapColors() and 
      ColMapColors() accessor methods.  All rows/columns that would be eliminated in the reduced system
      have a color of 1 in the corresponding RowMapColors/ColMapColors object.  All kept rows/cols have a 
      color of 0.
  */
  int Analyze(Epetra_RowMatrix * FullMatrix);

  //! Returns true if singletons were detected in this matrix (must be called after Analyze() to be effective).
  bool SingletonsDetected() const {if (!AnalysisDone_) return(false); else return(RowMapColors_->MaxNumColors()>1);};
  //@}

  //! @name Reduce methods
  //@{ 
  //! Return a reduced linear problem based on results of Analyze().
  /*! Creates a new Epetra_LinearProblem object based on the results of the Analyze phase.  A pointer
      to the reduced problem is obtained via a call to ReducedProblem().  
    	   
    \return Error code, set to 0 if no error.
  */
  int ConstructReducedProblem(Epetra_LinearProblem * Problem);

  //! Update a reduced linear problem using new values.
  /*! Updates an existing Epetra_LinearProblem object using new matrix, LHS and RHS values.  The matrix
      structure must be \e identical to the matrix that was used to construct the original reduced problem.  
    	   
    \return Error code, set to 0 if no error.
  */
  int UpdateReducedProblem(Epetra_LinearProblem * Problem);

  //@}
  //! @name Methods to construct Full System Solution
  //@{ 
  //! Compute a solution for the full problem using the solution of the reduced problem, put in LHS of FullProblem().
  /*! After solving the reduced linear system, this method can be called to compute the
      solution to the original problem, assuming the solution for the reduced system is valid. The solution of the 
      unreduced, original problem will be in the LHS of the original Epetra_LinearProblem.
    
  */
  int ComputeFullSolution();
  //@}
  //! @name Filter Statistics
  //@{ 
  //! Return number of rows that contain a single entry, returns -1 if Analysis not performed yet.
  int NumRowSingletons() const {return(NumGlobalRowSingletons_);};

  //! Return number of columns that contain a single entry that are \e not associated with singleton row, returns -1 if Analysis not performed yet.
  int NumColSingletons() const {return(NumGlobalColSingletons_);};

  //! Return total number of singletons detected, returns -1 if Analysis not performed yet.
  /*! Return total number of singletons detected across all processors.  This method will not return a
      valid result until after the Analyze() method is called.  The dimension of the reduced system can 
      be computed by subtracting this number from dimension of full system.
      \warning This method returns -1 if Analyze() method has not been called.
  */
  int NumSingletons() const {return(NumColSingletons()+NumRowSingletons());};

  //! Returns ratio of reduced system to full system dimensions, returns -1.0 if reduced problem not constructed.
  double RatioOfDimensions() const {return(RatioOfDimensions_);};

  //! Returns ratio of reduced system to full system nonzero count, returns -1.0 if reduced problem not constructed.
  double RatioOfNonzeros() const {return(RatioOfNonzeros_);};

  //@}
  //! @name Attribute Access Methods
  //@{ 

  //! Returns pointer to the original unreduced Epetra_LinearProblem.
  Epetra_LinearProblem * FullProblem() const {return(FullProblem_);};

  //! Returns pointer to the derived reduced Epetra_LinearProblem.
  Epetra_LinearProblem * ReducedProblem() const {return(ReducedProblem_);};

  //! Returns pointer to Epetra_CrsMatrix from full problem.
  Epetra_RowMatrix * FullMatrix() const {return(FullMatrix_);};

  //! Returns pointer to Epetra_CrsMatrix from full problem.
  Epetra_CrsMatrix * ReducedMatrix() const {return(ReducedMatrix_);};

  //! Returns pointer to Epetra_MapColoring object: color 0 rows are part of reduced system.
  Epetra_MapColoring * RowMapColors() const {return(RowMapColors_);};

  //! Returns pointer to Epetra_MapColoring object: color 0 columns are part of reduced system.
  Epetra_MapColoring * ColMapColors() const {return(ColMapColors_);};

  //! Returns pointer to Epetra_Map describing the reduced system row distribution.
  Epetra_Map * ReducedMatrixRowMap() const {return(ReducedMatrixRowMap_);};

  //! Returns pointer to Epetra_Map describing the reduced system column distribution.
  Epetra_Map * ReducedMatrixColMap() const {return(ReducedMatrixColMap_);};

  //! Returns pointer to Epetra_Map describing the domain map for the reduced system.
  Epetra_Map * ReducedMatrixDomainMap() const {return(ReducedMatrixDomainMap_);};

  //! Returns pointer to Epetra_Map describing the range map for the reduced system.
  Epetra_Map * ReducedMatrixRangeMap() const {return(ReducedMatrixRangeMap_);};
  //@}

 protected:

 

  //  This pointer will be zero if full matrix is not a CrsMatrix.
  Epetra_CrsMatrix * FullCrsMatrix() const {return(FullCrsMatrix_);};

  const Epetra_Map & FullMatrixRowMap() const {return(FullMatrix()->RowMatrixRowMap());};
  const Epetra_Map & FullMatrixColMap() const {return(FullMatrix()->RowMatrixColMap());};
  const Epetra_Map & FullMatrixDomainMap() const {return((FullMatrix()->OperatorDomainMap()));};
  const Epetra_Map & FullMatrixRangeMap() const {return((FullMatrix()->OperatorRangeMap()));};
  void InitializeDefaults();
  int ComputeEliminateMaps();
  int Setup(Epetra_LinearProblem * Problem);
  int InitFullMatrixAccess();
  int GetRow(int Row, int & NumIndices, int * & Indices);
  int GetRowGCIDs(int Row, int & NumIndices, double * & Values, int * & GlobalIndices);
  int GetRow(int Row, int & NumIndices, double * & Values, int * & Indices);
  int CreatePostSolveArrays(const Epetra_IntVector & RowIDs,
			    const Epetra_MapColoring & RowMapColors,
			    const Epetra_IntVector & ColProfiles,
			    const Epetra_IntVector & NewColProfiles,
			    const Epetra_IntVector & ColHasRowWithSingleton);
  
  int ConstructRedistributeExporter(Epetra_Map * SourceMap, Epetra_Map * TargetMap,
				    Epetra_Export * & RedistributeExporter,
				    Epetra_Map * & RedistributeMap);
  
  Epetra_LinearProblem * FullProblem_;
  Epetra_LinearProblem * ReducedProblem_;
  Epetra_RowMatrix * FullMatrix_;
  Epetra_CrsMatrix * FullCrsMatrix_;
  Epetra_CrsMatrix * ReducedMatrix_;
  Epetra_MultiVector * ReducedRHS_;
  Epetra_MultiVector * ReducedLHS_;
  
  Epetra_Map * ReducedMatrixRowMap_;
  Epetra_Map * ReducedMatrixColMap_;
  Epetra_Map * ReducedMatrixDomainMap_;
  Epetra_Map * ReducedMatrixRangeMap_;
  Epetra_Map * OrigReducedMatrixDomainMap_;
  Epetra_Import * Full2ReducedRHSImporter_;
  Epetra_Import * Full2ReducedLHSImporter_;
  Epetra_Export * RedistributeDomainExporter_;
  
  int * ColSingletonRowLIDs_;
  int * ColSingletonColLIDs_;
  int * ColSingletonPivotLIDs_;
  double * ColSingletonPivots_;
  
  
  int AbsoluteThreshold_;
  double RelativeThreshold_;

  int NumMyRowSingletons_;
  int NumMyColSingletons_;
  int NumGlobalRowSingletons_;
  int NumGlobalColSingletons_;
  double RatioOfDimensions_;
  double RatioOfNonzeros_;
  
  bool HaveReducedProblem_;
  bool UserDefinedEliminateMaps_;
  bool AnalysisDone_;
  bool SymmetricElimination_;
  
  Epetra_MultiVector * tempExportX_;
  Epetra_MultiVector * tempX_;
  Epetra_MultiVector * tempB_;
  Epetra_MultiVector * RedistributeReducedLHS_;
  int * Indices_;
  Epetra_SerialDenseVector Values_;
  
  Epetra_MapColoring * RowMapColors_;
  Epetra_MapColoring * ColMapColors_;
  bool FullMatrixIsCrsMatrix_;
  int MaxNumMyEntries_;
  
  
 private:
  //! Copy constructor (defined as private so it is unavailable to user).
  Epetra_CrsSingletonFilter(const Epetra_CrsSingletonFilter & Problem);
  Epetra_CrsSingletonFilter & operator=(const Epetra_CrsSingletonFilter & Problem);
};
#endif /* EPETRA_CRSSINGLETONFILTER_H */
