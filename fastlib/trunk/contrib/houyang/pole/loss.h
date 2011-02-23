#ifndef LOSS_H_
#define LOSS_H_

class Loss {
 public:
  // Returns the loss value
  virtual double GetLoss(double prediction, double label) = 0;
  // Returns the update scalar
  virtual double GetUpdate(double prediction, double label) = 0;
};

class SquaredLoss : public Loss {
 public:
  SquaredLoss();
  double GetLoss(double prediction, double label);
  double GetUpdate(double prediction, double label);
};

class HingeLoss : public Loss {
 public:
  HingeLoss();
  double GetLoss(double prediction, double label);
  double GetUpdate(double prediction, double label);
};

class SquaredhingeLoss : public Loss {
 public:
  SquaredhingeLoss();
  double GetLoss(double prediction, double label);
  double GetUpdate(double prediction, double label);
};

class LogisticLoss : public Loss {
 public:
  LogisticLoss();
  double GetLoss(double prediction, double label);
  double GetUpdate(double prediction, double label);
};

class QuantileLoss : public Loss {
 public:
  double tau_;

  QuantileLoss(double tau);
  double GetLoss(double prediction, double label);
  double GetUpdate(double prediction, double label);
};


#endif

