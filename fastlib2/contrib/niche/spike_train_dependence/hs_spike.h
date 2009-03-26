#ifndef HS_SPIKE_H
#define HS_SPIKE_H

#include "fastlib/fastlib.h"

class Spike {
 private:
  double time_;
  int label_;
  
 public:
  void Init(double time_in, int label_in) {
    time_ = time_in;
    label_ = label_in;
  }

  double time() {
    return time_;
  }
  
  int label() {
    return label_;
  }
};


class SpikeSeqPair {
 private:
  Vector x_;
  Vector y_;
  ArrayList<Spike> all_spikes;
  int tau_; // dependence horizon
  
 public:
  void Init(const char* x_filename,
	    const char* y_filename,
	    int tau_in) {
    tau_ = tau_in;
    LoadVector(x_filename, &x_);
    LoadVector(y_filename, &y_);
  }

  void LoadVector(const char* filename, Vector* vector) {
    ArrayList<double> linearized;
    linearized.Init();

    FILE* file = fopen(filename, "r");
    double num;
    while(fscanf(file, "%lf", &num) != EOF) {
      linearized.PushBackCopy(num);
    }
    fclose(file);

    linearized.Trim();
    
    int n_points = linearized.size();
    printf("n_points = %d\n", n_points);
    vector -> Own(linearized.ReleasePtr(), n_points);
  }

  void Merge() {
    all_spikes.Init(x_.length() + y_.length());
    int i_x = 0;
    int i_y = 0;
    for(int i_all = 0; (i_x < x_.length()) || (i_y < y_.length()); i_all++) {
      if((i_x != x_.length()) &&
	 ((i_y == y_.length()) || (x_[i_x] <= y_[i_y]))) {
	all_spikes[i_all].Init(x_[i_x], 0);
	i_x++;
      }
      else {
	all_spikes[i_all].Init(y_[i_y], 1);
	i_y++;
      }
    }
  }

  void PrintAllSpikes() {
    for(int i = 0; i < all_spikes.size(); i++) {
      char label;
      if(all_spikes[i].label() == 0) {
	label = 'X';
      }
      else {
	label = 'Y';
      }
      printf("%f %c\n", all_spikes[i].time(), label);
    }
  }

  void XRef() {
    int cur_x = x_.length() - 1;
    int cur_y = y_.length() - 1;

    int n_gap = 0;
    printf("(%d,%d) %d\n", cur_x, cur_y, n_gap);
    while(x_[cur_x] <= y_[cur_y]) {
      n_gap++;
      cur_y--;
      printf("(%d,%d) %d\n", cur_x, cur_y, n_gap);
    }

    while(n_gap < tau_) {
      if(cur_y == -1 || cur_x <= 0) {
	FATAL("Dependence horizon is too large for the spike sequences. Decrease it.");
      }
      cur_x--;
      printf("(%d,%d) %d\n", cur_x, cur_y, n_gap);
     
      while(x_[cur_x] <= y_[cur_y]) {
	n_gap++;
	cur_y--;
	printf("(%d,%d) %d\n", cur_x, cur_y, n_gap);
	if(cur_y == -1) {
	  break;
	}
      } 
    }

    printf("cur_x by y constraints = %d\n", cur_x);


  }

  void ConstructPoints() {
    int n_spikes = all_spikes.size();

    int cur_spike = n_spikes - 1;

    int n_x_spikes;
    int n_y_spikes;

    // referencing to X
    for(int cur_spike = n_spikes - 1;
	(n_x_spikes < tau_) || (n_y_spikes < tau_)) {
      if(all_spikes[cur_spike].label == 0) {
	n_x_spikes++;
      }
      else {
	n_y_spikes++;
      }
      cur_spike--;
    }




  }

};


#endif /* HS_SPIKE_H */
