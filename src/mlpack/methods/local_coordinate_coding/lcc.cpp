/**
 * @file lcc.cpp
 * @author Nishant Mehta
 *
 * Implementation of Local Coordinate Coding
 */

#include "lcc.hpp"

using namespace arma;
using namespace std;
using namespace mlpack::regression;
using namespace mlpack::lcc;

#define OBJ_TOL 1e-2 // 1E-9

namespace mlpack {
namespace lcc {

LocalCoordinateCoding::LocalCoordinateCoding(const mat& matX, u32 nAtoms, double lambda) :
  nDims(matX.n_rows),
  nAtoms(nAtoms),
  nPoints(matX.n_cols),
  matX(matX),
  matZ(mat(nAtoms, nPoints)),
  lambda(lambda)
{ /* nothing left to do */ }


// void LocalCoordinateCoding::Init(const mat& matX, u32 nAtoms, double lambda) {
//   this->matX = matX;

//   nDims = matX.n_rows;
//   nPoints = matX.n_cols;

//   this->nAtoms = nAtoms;
//   matD = mat(nDims, nAtoms);
//   matZ = mat(nAtoms, nPoints);
  
//   this->lambda = lambda;
// }


void LocalCoordinateCoding::SetDictionary(const mat& matD) {
  this->matD = matD;
}


void LocalCoordinateCoding::InitDictionary() {  
  RandomInitDictionary();
}


void LocalCoordinateCoding::LoadDictionary(const char* dictionaryFilename) {  
  matD.load(dictionaryFilename);
}


void LocalCoordinateCoding::RandomInitDictionary() {
  matD = randn(nDims, nAtoms);
  for(u32 j = 0; j < nAtoms; j++) {
    matD.col(j) /= norm(matD.col(j), 2);
  }
}


void LocalCoordinateCoding::DataDependentRandomInitDictionary() {
  matD = mat(nDims, nAtoms);
  for(u32 j = 0; j < nAtoms; j++) {
    vec vecD_j = matD.unsafe_col(j);
    RandomAtom(vecD_j);
  }
}


void LocalCoordinateCoding::RandomAtom(vec& atom) {
  atom.zeros();
  const u32 nSeedAtoms = 3;
  for(u32 i = 0; i < nSeedAtoms; i++) {
    atom +=  matX.col(rand() % nPoints);
  }
  atom /= ((double) nSeedAtoms);
  atom /= norm(atom, 2);
}


void LocalCoordinateCoding::DoLCC(u32 nIterations) {

  bool converged = false;
  double lastObjVal = 1e99;
  
  Log::Info << "Initial Coding Step" << endl;
  OptimizeCode();
  uvec adjacencies = find(matZ);
  Log::Info << "\tSparsity level: " 
	    << 100.0 * ((double)(adjacencies.n_elem)) 
                     / ((double)(nAtoms * nPoints))
	    << "%\n";
  Log::Info << "\tObjective value: " << Objective(adjacencies) << endl;
  
  for(u32 t = 1; t <= nIterations && !converged; t++) {
    Log::Info << "Iteration " << t << " of " << nIterations << endl;

    Log::Info << "Dictionary Step\n";
    OptimizeDictionary(adjacencies);
    double dsObjVal = Objective(adjacencies);
    Log::Info << "\tObjective value: " << Objective(adjacencies) << endl;
    
    Log::Info << "Coding Step" << endl;
    OptimizeCode();
    adjacencies = find(matZ);
    Log::Info << "\tSparsity level: " 
	      << 100.0 * ((double)(adjacencies.n_elem)) 
                       / ((double)(nAtoms * nPoints))
	      << "%\n";
    double curObjVal = Objective(adjacencies);
    Log::Info << "\tObjective value: " << curObjVal << endl;

    if(curObjVal > dsObjVal) {
      Log::Fatal << "Objective increased in sparse coding step!" << endl;
    }
    
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


void LocalCoordinateCoding::OptimizeCode() {
  mat matSqDists = 
    repmat(trans(sum(square(matD))), 1, nPoints)
    + repmat(sum(square(matX)), nAtoms, 1)
    - 2 * trans(matD) * matX;			     
  
  mat matInvSqDists = 1.0 / matSqDists;
  
  mat matDTD = trans(matD) * matD;
  mat matDPrimeTDPrime(matDTD.n_rows, matDTD.n_cols);
  
  for(u32 i = 0; i < nPoints; i++) {
    // report progress
    if((i % 100) == 0) {
      Log::Debug << "\t" << i << endl;
    }
    
    vec w = matSqDists.unsafe_col(i);
    vec invW = matInvSqDists.unsafe_col(i);
    mat matDPrime = matD * diagmat(invW);
    
    mat matDPrimeTDPrime = diagmat(invW) * matDTD * diagmat(invW);
    
    //LARS lars;
    // do we still need 0.5 * lambda? yes, yes we do
    //lars.Init(matDPrime.memptr(), matX.colptr(i), nDims, nAtoms, true, 0.5 * lambda); // apparently not as fast as using the below duo
                                                                                       // this may change, depending on the dimensionality and sparsity

    // the duo
    /* lars.Init(matDPrime.memptr(), matX.colptr(i), nDims, nAtoms, false, 0.5 * lambda); */
    /* lars.SetGram(matDPrimeTDPrime.memptr(), nAtoms); */
    
    bool useCholesky = false;
    LARS lars(useCholesky, 0.5 * lambda);
    lars.SetGram(matDPrimeTDPrime);
    
    lars.DoLARS(matDPrime, matX.unsafe_col(i));
    vec beta;
    lars.Solution(beta);
    matZ.col(i) = beta % invW;
  }
}


void LocalCoordinateCoding::OptimizeDictionary(uvec adjacencies) {
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
  
  // build matXPrime := [X x^1 ... x^1 ... x^n ... x^n]
  // where each x^i is repeated for the number of neighbors x^i has
  mat matXPrime = zeros(nDims, nPoints + adjacencies.n_elem);
  matXPrime(span::all, span(0, nPoints - 1)) = matX;
  u32 curCol = nPoints;
  for(u32 i = 0; i < nPoints; i++) {
    if(neighborCounts(i) > 0) {
      matXPrime(span::all, span(curCol, curCol + neighborCounts(i) - 1)) =
	repmat(matX.col(i), 1, neighborCounts(i));
    }
    curCol += neighborCounts(i);
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
  
  mat matZPrime = zeros(nActiveAtoms, nPoints + adjacencies.n_elem);
  //Log::Debug << "adjacencies.n_elem = " << adjacencies.n_elem << endl;
  matZPrime(span::all, span(0, nPoints - 1)) = matActiveZ;
  
  vec wSquared = ones(nPoints + adjacencies.n_elem, 1);
  //Log::Debug << "building up matZPrime\n";
  for(u32 l = 0; l < adjacencies.n_elem; l++) {
    u32 atomInd = adjacencies(l) % nAtoms;
    u32 pointInd = (u32) (adjacencies(l) / nAtoms);
    matZPrime(atomReverseLookup(atomInd), nPoints + l) = 1.0;
    wSquared(nPoints + l) = matZ(atomInd, pointInd); 
  }
  
  wSquared.subvec(nPoints, wSquared.n_elem - 1) = 
    lambda * abs(wSquared.subvec(nPoints, wSquared.n_elem - 1));
  
  //Log::Debug << "about to solve\n";
  mat matDEstimate;
  if(inactiveAtoms.empty()) {
    mat A = matZPrime * diagmat(wSquared) * trans(matZPrime);
    mat B = matZPrime * diagmat(wSquared) * trans(matXPrime);
    
    //Log::Debug << "solving...\n";
    matDEstimate = 
      trans(solve(A, B));
    /*    
    matDEstimate = 
      trans(solve(matZPrime * diagmat(wSquared) * trans(matZPrime),
		  matZPrime * diagmat(wSquared) * trans(matXPrime)));
    */
  }
  else {
    matDEstimate = zeros(nDims, nAtoms);
    //Log::Debug << "solving...\n";
    mat matDActiveEstimate = 
      trans(solve(matZPrime * diagmat(wSquared) * trans(matZPrime),
		  matZPrime * diagmat(wSquared) * trans(matXPrime)));
    for(u32 j = 0; j < nActiveAtoms; j++) {
      matDEstimate.col(activeAtoms[j]) = matDActiveEstimate.col(j);
    }
    for(u32 j = 0; j < nInactiveAtoms; j++) {
      vec vecD_j = matDEstimate.unsafe_col(inactiveAtoms[j]);
      RandomAtom(vecD_j);
      /*
      vec new_atom = randn(nDims, 1);
      matDEstimate.col(inactiveAtoms[i]) = 
	new_atom / norm(new_atom, 2);
      */
    }
  }
  matD = matDEstimate;
}
// need to test above function, sleepy now, will resume soon!


double LocalCoordinateCoding::Objective(uvec adjacencies) {
  double weightedL1NormZ = 0;
  u32 nAdjacencies = adjacencies.n_elem;
  for(u32 l = 0; l < nAdjacencies; l++) {
    u32 atomInd = adjacencies(l) % nAtoms;
    u32 pointInd = (u32) (adjacencies(l) / nAtoms);
    weightedL1NormZ += fabs(matZ(atomInd, pointInd)) * as_scalar(sum(square(matD.col(atomInd) - matX.col(pointInd))));
  }
  double froNormResidual = norm(matX - matD * matZ, "fro");
  return froNormResidual * froNormResidual + lambda * weightedL1NormZ;
}


void LocalCoordinateCoding::PrintDictionary() {
  matD.print("Dictionary");
}


void LocalCoordinateCoding::PrintCoding() {
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


}; // namespace lcc
}; // namespace mlpack
