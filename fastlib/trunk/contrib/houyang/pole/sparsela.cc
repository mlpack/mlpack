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

/////////////////////////////////////
// Sparse vector scaling: w <= a * w
/////////////////////////////////////
void Svector::SparseScaleOverwrite(double a) {
  if (a == 0.0) {
    Fs_.clear();
    return;
  }
  else if (a == 1.0) {
    return;
  }
  else if (a == -1.0) {
    vector<Feature>::iterator it;
    for (it=Fs_.begin(); it<Fs_.end(); it++) {
      it->v_ = - it->v_;
    }
    return;
  }
  else {
    vector<Feature>::iterator it;
    for (it=Fs_.begin(); it<Fs_.end(); it++) {
      if (it->v_ != 0.0)
	it->v_ = a * it->v_;
    }
  }
}

/////////////////////////////
// Sparse Dot Product: w^T v
/////////////////////////////
double Svector::SparseDot(Svector *v) {
  if (Fs_.empty() || v->Fs_.empty()) {
    return 0.0;
  }
  else {
    double dv = 0.0;
    vector<Feature>::iterator itw = Fs_.begin();
    vector<Feature>::iterator itv = v->Fs_.begin();
    while (itw<Fs_.end() && itv<v->Fs_.end()) {
      if (itw->i_ == itv->i_) {
	dv += itw->v_ * itv->v_;
	itw++; itv++;
      }
      else {
	if (itw->i_ > itv->i_) {
	  itv++;
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

///////////////////////////////////////////
// Sparse vector scaled add: w<= w+ a * x
///////////////////////////////////////////
void Svector::SparseAddExpertOverwrite(double a, Svector *x) {
  size_t nz_x = x->Fs_.size();
  size_t ct_w = 0, ct_x = 0;

  if (a == 0.0 || x->Fs_.empty()) { // scale==0 or all-0 x: w unchanged
    return;
  }
  else if (Fs_.empty()) { // w: all-0
    Fs_ = x->Fs_;
    for (ct_w = 0; ct_w < Fs_.size(); ct_w ++) {
      Fs_[ct_w].v_ = a * Fs_[ct_w].v_;
    }
    return;
  }
  else { // neither w nor x is of all-0
    while (ct_w<Fs_.size() || ct_x<nz_x) {
      if (ct_w == Fs_.size()) { // w reaches end, while x still not
	Fs_.push_back( Feature(x->Fs_[ct_x].i_, a * x->Fs_[ct_x].v_) );
	++ct_w; ++ct_x;
      }
      else if (ct_x == nz_x) { // x reaches end, while w still not
	// the succeeding w remain unchanged
	break;
      }
      else { // neither w nor x reaches end
	if (Fs_[ct_w].i_ == x->Fs_[ct_x].i_) {
	  Fs_[ct_w].v_ += a * x->Fs_[ct_x].v_;
	  ++ct_w; ++ct_x;
	}
	else if (Fs_[ct_w].i_ > x->Fs_[ct_x].i_) {
	  Fs_.insert(Fs_.begin()+ct_w, Feature(x->Fs_[ct_x].i_, a * x->Fs_[ct_x].v_));
	  ++ct_w; ++ct_x;
	}
	else { // w.Fs[ct_w].i < x.Fs[ct_x].i
	  ++ct_w;
	}
      }
    }
  }
}

///////////////////////////////
// Sparse vector add: w<= w+ x
///////////////////////////////
void Svector::SparseAddOverwrite(Svector *x) {
  size_t nz_x = x->Fs_.size();
  size_t ct_w = 0, ct_x = 0;

  if (x->Fs_.empty()) { // scale==0 or all-0 x: w unchanged
    return;
  }
  else if (Fs_.empty()) { // w: all-0
    Fs_ = x->Fs_;
    return;
  }
  else { // neither w nor x is of all-0
    while (ct_w<Fs_.size() || ct_x<nz_x) {
      if (ct_w == Fs_.size()) { // w reaches end, while x still not
	Fs_.push_back(x->Fs_[ct_x]);
	++ct_w; ++ct_x;
      }
      else if (ct_x == nz_x) { // x reaches end, while w still not
	// the succeeding w remain unchanged
	break;
      }
      else { // neither w nor x reaches end
	if (Fs_[ct_w].i_ == x->Fs_[ct_x].i_) {
	  Fs_[ct_w].v_ += x->Fs_[ct_x].v_;
	  ++ct_w; ++ct_x;
	}
	else if (Fs_[ct_w].i_ > x->Fs_[ct_x].i_) {
	  Fs_.insert(Fs_.begin()+ct_w, x->Fs_[ct_x]);
	  ++ct_w; ++ct_x;
	}
	else { // w.Fs[ct_w].i < x.Fs[ct_x].i
	  ++ct_w;
	}
      }
    }
  }
}

///////////////////////////////
// Shrink: remove 0 values
///////////////////////////////
void Svector::Shrink(double threshold) {
  for (size_t i=0; i<Fs_.size(); i++) {
    if (fabs(Fs_[i].v_) < threshold) {
      Fs_.erase(Fs_.begin()+i);
      i--;
    }
  }
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

