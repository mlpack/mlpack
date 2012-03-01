/**
 * @file sparse_coding.cpp
 * @author Nishant Mehta
 *
 * Implementation of Sparse Coding with Dictionary Learning using l1 (LASSO) or 
 * l1+l2 (Elastic Net) regularization
 */

#include "sparse_coding.hpp"

using namespace std;
using namespace arma;
using namespace mlpack::regression;

#define OBJ_TOL 1e-2 // 1E-9
#define NEWTON_TOL 1e-6 // 1E-9

namespace mlpack {
namespace sparse_coding {

/*
void SparseCoding::Init(double* memX, u32 nDims, u32 nPoints,
			u32 nAtoms, double lambda1) {
  matX = mat(memX, nDims, nPoints, false, true);

  this->nDims = nDims;
  this->nPoints = nPoints;

  this->nAtoms = nAtoms;
  //matD = mat(nDims, nAtoms);
  matZ = mat(nAtoms, nPoints);
  
  this->lambda1 = lambda1;
  lambda2 = 0;
}
*/

/*
void SparseCoding::SetDictionary(double* memD) {
  matD = mat(memD, nDims, nAtoms, false, true);
}
*/





SparseCoding::SparseCoding(const mat& matX, u32 nAtoms, double lambda1, double lambda2) :
  nDims(matX.n_rows),  
  nAtoms(nAtoms),
  nPoints(matX.n_cols),
  matX(matX),
  matZ(mat(nAtoms, nPoints)),
  lambda1(lambda1),
  lambda2(lambda2)
{ /* nothing left to do */ }
  
  
void SparseCoding::SetData(const mat& matX) {
  this -> matX = matX;
}


void SparseCoding::SetDictionary(const mat& matD) {
  this -> matD = matD;
}


void SparseCoding::InitDictionary() {  
  DataDependentRandomInitDictionary();
}


void SparseCoding::LoadDictionary(const char* dictionaryFilename) {  
  matD.load(dictionaryFilename);
}

// always a not good decision!
void SparseCoding::RandomInitDictionary() {
  matD = randn(nDims, nAtoms);
  for(u32 j = 0; j < nAtoms; j++) {
    matD.col(j) /= norm(matD.col(j), 2);
  }
}

// the sensible heuristic
void SparseCoding::DataDependentRandomInitDictionary() {
  matD = mat(nDims, nAtoms);
  for(u32 j = 0; j < nAtoms; j++) {
    vec vecD_j = matD.unsafe_col(j);
    RandomAtom(vecD_j);
  }
}


void SparseCoding::RandomAtom(vec& atom) {
  atom.zeros();
  const u32 nSeedAtoms = 3;
  for(u32 i = 0; i < nSeedAtoms; i++) {
    atom +=  matX.col(rand() % nPoints);
  }
  atom /= norm(atom, 2);
}


void SparseCoding::DoSparseCoding(u32 nIterations) {

  bool converged = false;
  double lastObjVal = 1e99;  

  Log::Info << "Initial Coding Step" << endl;
  OptimizeCode();
  uvec adjacencies = find(matZ);
  Log::Info << "\tSparsity level: " 
	    << 100.0 * ((double)(adjacencies.n_elem)) 
                     / ((double)(nAtoms * nPoints))
	    << "%\n";
  Log::Info << "\tObjective value: " << Objective() << endl;
  
  for(u32 t = 1; t <= nIterations && !converged; t++) {
    Log::Info << "Iteration " << t << " of " << nIterations << endl;

    Log::Info << "Dictionary Step\n";
    OptimizeDictionary(adjacencies);
    //ProjectDictionary(); // is this necessary? solutions to OptimizeDictionary should be feasible
    Log::Info << "\tObjective value: " << Objective() << endl;

    Log::Info << "Coding Step" << endl;
    OptimizeCode();
    adjacencies = find(matZ);
    Log::Info << "\tSparsity level: " 
	      << 100.0 * ((double)(adjacencies.n_elem)) 
                       / ((double)(nAtoms * nPoints))
	      << "%\n";
    double curObjVal = Objective();
    Log::Info << "\tObjective value: " << curObjVal << endl;
    
    double objValImprov = lastObjVal - curObjVal;
    Log::Info << "\t\t\t\t\tImprovement: " << std::scientific
	      <<  objValImprov << endl;
    if(objValImprov < OBJ_TOL) {
      converged = true;
      Log::Info << "Converged within tolerance\n";
    }
    
    lastObjVal = curObjVal;
  }
}


void SparseCoding::OptimizeCode() {
  mat matGram;
  if(lambda2 > 0) {
    matGram = trans(matD) * matD + lambda2 * eye(nAtoms, nAtoms);
  }
  else {
    matGram = trans(matD) * matD;
  }
  
  for(u32 i = 0; i < nPoints; i++) {
    // report progress
    if((i % 100) == 0) {
      Log::Debug << "\t" << i << endl;
    }
    
    //Lars lars;
    // do we still need 0.5 * lambda? no, because we're using the standard objective now, which includes 0.5 scaling for quadratic terms
    //lars.Init(D.memptr(), matX.colptr(i), nDims, nAtoms, true, lambda1); // apparently not as fast as using the below duo
                                                                                       // this may change, depending on the dimensionality and sparsity

    // the duo
    //lars.Init(matD.memptr(), matX.colptr(i), nDims, nAtoms, false, lambda1);
    //lars.SetGram(matGram.memptr(), nAtoms);
    //lars.DoLARS();
 

    bool useCholesky = false;
    LARS* lars;
    if(lambda2 > 0) {
      lars = new LARS(useCholesky, lambda1, lambda2);
    }
    else {
      lars = new LARS(useCholesky, lambda1);
    }
    lars -> SetGram(matGram);
    lars -> DoLARS(matD, matX.unsafe_col(i));
    
    vec beta;
    lars -> Solution(beta);
    matZ.col(i) = beta;
    delete lars;
  }
}


void SparseCoding::OptimizeDictionary(uvec adjacencies) {
  // count number of atomic neighbors for each point x^i
  uvec neighborCounts = zeros<uvec>(nPoints, 1);
  if(adjacencies.n_elem > 0) {
    // this gets the column index
    u32 curPointInd = (u32)(adjacencies(0) / nAtoms);
    u32 curCount = 1;
    for(u32 l = 1; l < adjacencies.n_elem; l++) {
      if((u32)(adjacencies(l) / nAtoms) == curPointInd) {
	curCount++;
      }
      else {
	neighborCounts(curPointInd) = curCount;
	curPointInd = (u32)(adjacencies(l) / nAtoms);
	curCount = 1;
      }
    }
    neighborCounts(curPointInd) = curCount;
  }
  
  // handle the case of inactive atoms (atoms not used in the given coding)
  std::vector<u32> inactiveAtoms;
  std::vector<u32> activeAtoms;
  activeAtoms.reserve(nAtoms);
  for(u32 j = 0; j < nAtoms; j++) {
    if(accu(matZ.row(j) != 0) == 0) {
      inactiveAtoms.push_back(j);
    }
    else {
      activeAtoms.push_back(j);
    }
  }
  u32 nActiveAtoms = activeAtoms.size();
  u32 nInactiveAtoms = inactiveAtoms.size();

  // efficient construction of Z restricted to active atoms
  mat matActiveZ;
  if(inactiveAtoms.empty()) {
    matActiveZ = matZ;
  }
  else {
    uvec inactiveAtomsVec = conv_to< uvec >::from(inactiveAtoms);
    RemoveRows(matZ, inactiveAtomsVec, matActiveZ);
  }
  
  uvec atomReverseLookup = uvec(nAtoms);
  for(u32 i = 0; i < nActiveAtoms; i++) {
    atomReverseLookup(activeAtoms[i]) = i;
  }

  if(nInactiveAtoms > 0) {
    Log::Info << "There are " << nInactiveAtoms << " inactive atoms. They will be re-initialized randomly.\n";
  }
  
  
  Log::Debug << "Solving Dual via Newton's Method\n";
  
  mat matDEstimate;
  // solve using Newton's method in the dual - note that the final dot multiplication with inv(A) seems to be unavoidable. Although more expensive, the code written this way (we use solve()) should be more numerically stable than just using inv(A) for everything.
  vec dualVars = zeros<vec>(nActiveAtoms);
  //vec dualVars = 1e-14 * ones<vec>(nActiveAtoms);
  //vec dualVars = 10.0 * randu(nActiveAtoms, 1); // method used by feature sign code - fails miserably here. perhaps the MATLAB optimizer fmincon does something clever?
  /*vec dualVars = diagvec(solve(matD, matX * trans(matZ)) - matZ * trans(matZ));
  for(u32 i = 0; i < dualVars.n_elem; i++) {
    if(dualVars(i) < 0) {
      dualVars(i) = 0;
    }
  }
  */
  //dualVars.print("dual vars");

  bool converged = false;
  mat matZXT = matActiveZ * trans(matX);
  mat matZZT = matActiveZ * trans(matActiveZ);
  for(u32 t = 1; !converged; t++) {
    mat A = matZZT + diagmat(dualVars);
    
    mat matAInvZXT = solve(A, matZXT);
    
    vec gradient = -( sum(square(matAInvZXT), 1) - ones<vec>(nActiveAtoms) );
    
    mat hessian = 
      -( -2 * (matAInvZXT * trans(matAInvZXT)) % inv(A) );
    
    //printf("solving for dual variable update...");
    vec searchDirection = -solve(hessian, gradient);
    //vec searchDirection = -gradient;

 
    
    // BEGIN ARMIJO LINE SEARCH
    const double c = 1e-4;
    double alpha = 1.0;
    const double rho = 0.9;
    double sufficientDecrease = c * dot(gradient, searchDirection);

    /*
    {
      double sumDualVars = sum(dualVars);    
      double fOld = 
	-( -trace(trans(matZXT) * matAInvZXT) - sumDualVars );
      printf("fOld = %f\t", fOld);
      double fNew = 
	-( -trace(trans(matZXT) * solve(matZZT + diagmat(dualVars + alpha * searchDirection), matZXT))
	  - (sumDualVars + alpha * sum(searchDirection)) );
      printf("fNew = %f\n", fNew);
    }
    */
    
    double improvement;
    while(true) {
      // objective
      double sumDualVars = sum(dualVars);
      double fOld = 
	-( -trace(trans(matZXT) * matAInvZXT) - sumDualVars );
      double fNew = 
	-( -trace(trans(matZXT) * solve(matZZT + diagmat(dualVars + alpha * searchDirection), matZXT))
	   - (sumDualVars + alpha * sum(searchDirection)) );

      // printf("alpha = %e\n", alpha);
      // printf("norm of gradient = %e\n", norm(gradient, 2));
      // printf("sufficientDecrease = %e\n", sufficientDecrease);
      // printf("fNew - fOld - sufficientDecrease = %e\n", 
      // 	     fNew - fOld - alpha * sufficientDecrease);
      if(fNew <= fOld + alpha * sufficientDecrease) {
	searchDirection = alpha * searchDirection;
	improvement = fOld - fNew;
	break;
      }
      alpha *= rho;
    }
    // END ARMIJO LINE SEARCH
    
    dualVars += searchDirection;
    //printf("\n");
    double normGradient = norm(gradient, 2);
    Log::Debug << "Newton Iteration " << t << ":" << endl;
    Log::Debug << "\tnorm of gradient = " << std::scientific << normGradient << endl;
    Log::Debug << "\timprovement = " << std::scientific << improvement << endl;

    // if(normGradient < NEWTON_TOL) {
    //   converged = true;
    // }
    if(improvement < NEWTON_TOL) {
      converged = true;
    }
  }
  //dualVars.print("dual solution");
  if(inactiveAtoms.empty()) {
    matDEstimate = trans(solve(matZZT + diagmat(dualVars), matZXT));
  }
  else {
    mat matDActiveEstimate = trans(solve(matZZT + diagmat(dualVars), matZXT));
    matDEstimate = zeros(nDims, nAtoms);
    for(u32 i = 0; i < nActiveAtoms; i++) {
      matDEstimate.col(activeAtoms[i]) = matDActiveEstimate.col(i);
    }
    for(u32 i = 0; i < nInactiveAtoms; i++) {
      vec vecmatDi = matDEstimate.unsafe_col(inactiveAtoms[i]);
      RandomAtom(vecmatDi);
    }
  }
  matD = matDEstimate;
}


void SparseCoding::ProjectDictionary() {
  for(u32 j = 0; j < nAtoms; j++) {
    double normD_j = norm(matD.col(j), 2);
    if(normD_j > 1) {
      if(normD_j - 1.0 > 1e-9) {
	Log::Warn << "Norm Exceeded 1 by " << std::scientific << normD_j - 1.0
		  << "\n\tShrinking...\n";
	matD.col(j) /= normD_j;
      }
      // no need to normalize if the dictionary wasn't that infeasible
      //matD.col(j) /= normD_j;
    }
  }
}


double SparseCoding::Objective() {
  double l11NormZ = sum(sum(abs(matZ)));
  double froNormResidual = norm(matX - matD * matZ, "fro");
  if(lambda2 > 0) {
    double froNormZ = norm(matZ, "fro");
    return 
      0.5 * (froNormResidual * froNormResidual + lambda2 * froNormZ * froNormZ)
      + lambda1 * l11NormZ;
  }
  else {
    return 0.5 * froNormResidual * froNormResidual + lambda1 * l11NormZ;
  }
}


void SparseCoding::PrintDictionary() {
  matD.print("Dictionary");
}


void SparseCoding::PrintCoding() {
  matZ.print("Coding matrix");
}


void RemoveRows(const mat& X, uvec rows_to_remove, mat& X_mod) {

  u32 n_cols = X.n_cols;
  u32 n_rows = X.n_rows;
  u32 n_to_remove = rows_to_remove.n_elem;
  u32 n_to_keep = n_rows - n_to_remove;
  
  if(n_to_remove == 0) {
    X_mod = X;
  }
  else {
    X_mod.set_size(n_to_keep, n_cols);

    u32 cur_row = 0;
    u32 remove_ind = 0;
    // first, check 0 to first row to remove
    if(rows_to_remove(0) > 0) {
      // note that this implies that n_rows > 1
      u32 height = rows_to_remove(0);
      X_mod(span(cur_row, cur_row + height - 1), span::all) =
	X(span(0, rows_to_remove(0) - 1), span::all);
      cur_row += height;
    }
    // now, check i'th row to remove to (i + 1)'th row to remove, until i = penultimate row
    while(remove_ind < n_to_remove - 1) {
      u32 height = 
	rows_to_remove[remove_ind + 1]
	- rows_to_remove[remove_ind]
	- 1;
      if(height > 0) {
	X_mod(span(cur_row, cur_row + height - 1), 
	      span::all) =
	  X(span(rows_to_remove[remove_ind] + 1,
		 rows_to_remove[remove_ind + 1] - 1), 
	    span::all);
	cur_row += height;
      }
      remove_ind++;
    }
    // now that i is last row to remove, check last row to remove to last row
    if(rows_to_remove[remove_ind] < n_rows - 1) {
      X_mod(span(cur_row, n_to_keep - 1), 
	    span::all) = 
	X(span(rows_to_remove[remove_ind] + 1, n_rows - 1), 
	  span::all);
    }
  }
}


}; // namespace sparse_coding
}; // namespace mlpack
