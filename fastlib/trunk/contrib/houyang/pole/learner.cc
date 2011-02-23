// Implementation for Learner

#include "learner.h"

Learner::Learner() {
}

Learner::~Learner() {
  if (LF_)
    delete LF_;
}

void Learner::OnlineLearn() {
  if (v_)
    cout << "Online learning" << endl;

  // Get input data
  if (read_port_) {
    D_.TR_ = new Dataset(NULL, port_, false);
    D_.TR_->ReadFromPort();
  }
  else {
    if (fn_learn_ != "") {
      D_.TR_ = new Dataset(fn_learn_, 0, random_data_);
      D_.TR_->ReadFromFile();
    }
    else {
      cout << "No input file provided!" << endl;
      exit(1);
    }
  }
  Learn();
}

void Learner::BatchLearn() {
  if (v_)
    cout << "Batch learning" << endl;

  // Training
  if (fn_learn_ != "") {
    D_.TR_ = new Dataset(fn_learn_, 0, random_data_);
    D_.TR_->ReadFromFile();
  }
  else {
    cout << "No training file provided!" << endl;
    exit(1);
  }
  Learn();

  // Testing
  if (fn_predict_ != "") {
    if (fn_predict_ == fn_learn_) {
      D_.TE_ = D_.TR_;
    }
    else {
      D_.TE_ = new Dataset(fn_predict_, 0, false);
      D_.TE_->ReadFromFile();
    }
  }
  else {
    cout << "No testing file provided!" << endl;
    exit(1);
  }
  Test();
}
