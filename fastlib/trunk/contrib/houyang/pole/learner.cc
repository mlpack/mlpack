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
    TR_ = new Data(NULL, port_, false);
    TR_->ReadFromPort();
  }
  else {
    if (fn_learn_ != "") {
      TR_ = new Data(fn_learn_, 0, random_data_);
      TR_->ReadFromFile();
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
    TR_ = new Data(fn_learn_, 0, random_data_);
    TR_->ReadFromFile();
  }
  else {
    cout << "No training file provided!" << endl;
    exit(1);
  }
  Learn();

  // Testing
  if (fn_predict_ != "") {
    if (fn_predict_ == fn_learn_) {
      TE_ = TR_;
    }
    else {
      TE_ = new Data(fn_predict_, 0, false);
      TE_->ReadFromFile();
    }
  }
  else {
    cout << "No testing file provided!" << endl;
    exit(1);
  }
  Test();
}
