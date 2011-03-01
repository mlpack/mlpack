// Implementation for Sparse Linear Algebra

#include "sparsela.h"

//-------------------------Feature----------------------------//

//////////////////////////
// Create an empty feature
//////////////////////////
Feature::Feature() {
}

//////////////////////////
// Create a feature
//////////////////////////
Feature::Feature(T_IDX i, T_VAL v) : i_(i), v_(v) {
}

//---------------------General Sparse Vector-----------------//

////////////////////////////////
// Create an empty sparse vector
////////////////////////////////
Svector::Svector() {
}

//////////////////////////////////////////////
// Create a sparse vector from given features
//////////////////////////////////////////////
Svector::Svector(vector<Feature> Fs) : Fs_(Fs) {
}

////////////////////////////////////////////
// Create a (dense) vector of constant value
////////////////////////////////////////////
Svector::Svector(T_IDX n_f, T_VAL v) : Fs_(n_f, Feature()){
  T_IDX sz = Fs_.size();
  for (T_IDX i=0; i<sz; i++) {
    Fs_[i] = Feature(i, v);
  }
}

//////////////////////////////////
// Copy from a given sparse vector
//////////////////////////////////
void Svector::Copy(Svector V) {
  Fs_ = V.Fs_;
}

/////////////////////////////////////////////
// Set to be a vector of given constant value
/////////////////////////////////////////////
void Svector::SetAll(T_VAL v) {
  T_IDX sz = Fs_.size();
  for (T_IDX i=0; i<sz; i++) {
    Fs_[i] = Feature(i, v); // idx starts from 1
  }
}

//////////////////////////////////////////////////////////////
// Set to be a vector of given lenght and given constant value
//////////////////////////////////////////////////////////////
void Svector::SetAllResize(T_IDX n_f, T_VAL v) {
  Fs_.resize(n_f);
  for (T_IDX i=0; i<n_f; i++) {
    Fs_[i] = Feature(i, v); // idx starts from 1
  }
}

///////////////////////////////
// Push one feature at the end
///////////////////////////////
void Svector::PushBack(Feature F) {
  Fs_.push_back(F);
}

///////////////////////////////////////////
// Insert one feature at a given position p
///////////////////////////////////////////
void Svector::InsertOne(T_IDX p, Feature F) {
  Fs_.insert(Fs_.begin() + p, F);
}


///////////////////////////////////////////
// Erase one feature at a given position p
///////////////////////////////////////////
void Svector::EraseOne(T_IDX p) {
  Fs_.erase(Fs_.begin() + p);
}

////////////////
// Clear content
////////////////
void Svector::Clear() {
  Fs_.clear();
}

/////////////////
// Print content
/////////////////
void Svector::Print() {
  vector<Feature>::iterator it;
  for (it=Fs_.begin(); it<Fs_.end(); it++) {
    cout << it->i_ << ":" << it->v_ << " ";
  }
  cout << endl;
}

/////////////////////////////
// Sparse Dot Product: w^T x
/////////////////////////////
double Svector::SparseDot(Svector *x) {
  if (Fs_.empty() || x->Fs_.empty()) {
    return 0.0;
  }
  else {
    double dv = 0.0;
    vector<Feature>::iterator itw = Fs_.begin();
    vector<Feature>::iterator itx = x->Fs_.begin();
    while (itw<Fs_.end() && itx<x->Fs_.end()) {
      if (itw->i_ == itx->i_) {
	dv += itw->v_ * itx->v_;
	itw++; itx++;
      }
      else {
	if (itw->i_ > itx->i_) {
	  itx++;
	}
	else {
	  itw++;
	}
      }
    }
    return dv;
  }
}

////////////////////////////////////////
// Sparse squared L2 norm: return w^T w
////////////////////////////////////////
double Svector::SparseSqL2Norm() {
  vector<Feature>::iterator it;
  double v = 0.0;
  for (it=Fs_.begin(); it<Fs_.end(); it++) {
    if (it->v_ != 0)
      v += pow(it->v_, 2);
  }
  return v;
}


//---------------------------Example-------------------------//
//////////////////////////
// Create an empty feature
//////////////////////////
Example::Example() : y_(0), in_use_(false), ud_("") {
}

/////////////////////////////////////////////////////////
// Create an unused example from given features and label
////////////////////////////////////////////////////////
Example::Example(vector<Feature> Fs, T_LBL y) : 
  Svector(Fs), y_(y), in_use_(false), ud_("") {
}

/////////////////////////////////////////////////////////
// Create an unused example from given features and label
////////////////////////////////////////////////////////
Example::Example(vector<Feature> Fs, T_LBL y, string ud) : 
  Svector(Fs), y_(y), in_use_(false), ud_(ud) {
}

////////////////////////////
// Copy from a given example
////////////////////////////
void Example::Copy(Example X) {
  Fs_ = X.Fs_;
  y_ = X.y_;
  in_use_ = X.in_use_;
  ud_ = X.ud_;
}

////////////////
// Clear content
////////////////
void Example::Clear() {
  Svector::Clear();
  y_ = 0;
  in_use_ = 0;
  ud_ = "";
}

/////////////////
// Print content
/////////////////
void Example::Print() {
  cout << y_ << " ";
  Svector::Print();
}

