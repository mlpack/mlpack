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
  int n_spikes_;
  ArrayList<Spike> all_spikes;
  int tau_; // dependence horizon

  void RandPerm(Vector* p_x) {
    Vector &x = *p_x;

    int length =- x.length();
    int length_minus_1 = length - 1;
    
    int swap_index;
    double temp;
    for(int i = 0; i < length_minus_1; i++) {
      swap_index = (rand() % (length - i)) + i;
      temp = x[i];
      x[i] = x[swap_index];
      x[swap_index] = temp;
    }
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
    //printf("n_points = %d\n", n_points);
    vector -> Own(linearized.ReleasePtr(), n_points);
  }

  void Merge() {
    n_spikes_ = x_.length() + y_.length();
    all_spikes.Init(n_spikes_);
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

  void ConstructPointsByRefLabel(int ref_label,
				 Matrix* primary_points, 
				 Matrix* secondary_points) {
    int min_ref_spike_num = FindMinSpikeReference(ref_label);
    //printf("min_ref_spike_num = %d\n", min_ref_spike_num);

    int n_points = 0;
    for(int i = min_ref_spike_num; i < n_spikes_; i++) {
      if(all_spikes[i].label() == ref_label) {
	n_points++;
      }
    }
    
    primary_points -> Init(tau_, n_points);
    ConstructPointsByRefAndQueryLabel(min_ref_spike_num,
				      ref_label,
				      ref_label,
				      primary_points);
    secondary_points -> Init(tau_, n_points);
    ConstructPointsByRefAndQueryLabel(min_ref_spike_num,
				      ref_label,
				      1 - ref_label,
				      secondary_points);

  }

  void ConstructPointsByRefAndQueryLabel(int min_ref_spike_num,
					 int ref_label,
					 int query_label,
					 Matrix* points) {
    int point_num = 0;
    for(int i = min_ref_spike_num; i < n_spikes_; i++) {
      if(all_spikes[i].label() == ref_label) {
	Vector point;
	points -> MakeColumnVector(point_num, &point);
	ConstructPoint(i, query_label, &point);
	point_num++;
      }
    }
  }
  
  void ConstructPoint(int ref_spike_num, int query_label, Vector* p_point) {
    //printf("ref_spike_num = %d\n", ref_spike_num);
    Vector& point = *p_point;

    double ref_spike_time = all_spikes[ref_spike_num].time();

    int n_complete = 0;
    for(int i = ref_spike_num - 1; n_complete < tau_; i--) {
      if(all_spikes[i].label() == query_label) {
	point[n_complete] = ref_spike_time - all_spikes[i].time();
	n_complete++;
      }
    }
  }

  int FindMinSpikeReference(int ref_label) {
    int cur_spike = 0;

    int n_primary_spikes = 0; // spikes with the same label as ref_label
    int n_secondary_spikes = 0; // spikes with the label other than ref_label

    int n_spikes_minus_1 = n_spikes_ - 1;
    for(;
	((n_primary_spikes < tau_) || (n_secondary_spikes < tau_))
	  && (cur_spike < n_spikes_minus_1);
	cur_spike++) {
      if(all_spikes[cur_spike].label() == ref_label) {
	n_primary_spikes++;
      }
      else {
	n_secondary_spikes++;
      }
    }

    if((n_primary_spikes < tau_) || (n_secondary_spikes < tau_)) {
      FATAL("Dependence horizon is too large for the spike sequences. Decrease it.");
    }
    
    while((cur_spike < n_spikes_)
	  && (all_spikes[cur_spike].label() != ref_label)) {
      cur_spike++;
    }

    if(cur_spike == n_spikes_) {
      FATAL("Dependence horizon is too large for the spike sequences. Decrease it.");
    }

    return cur_spike;
  }

  
  
 public:
  void Init(const char* x_filename,
	    const char* y_filename,
	    int tau_in) {
    tau_ = tau_in;
    LoadVector(x_filename, &x_);
    LoadVector(y_filename, &y_);
    Merge();
  }

  //void Init() {
  // }

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

  void ConstructPoints(Matrix* x_ref_primary_points,
		       Matrix* x_ref_secondary_points,
		       Matrix* y_ref_primary_points,
		       Matrix* y_ref_secondary_points) {
    ConstructPointsByRefLabel(0,
			      x_ref_primary_points, x_ref_secondary_points);
        
    ConstructPointsByRefLabel(1,
			      y_ref_primary_points, y_ref_secondary_points);
  }

  void CreatePermutation(SpikeSeqPair* p_permed_pair) {
    SpikeSeqPair &permed_pair = *p_permed_pair;
    //permed_pair.Init();
    permed_pair.x_.Copy(x_);
    int y_len = y_.length();
    Vector y_isi;
    y_isi.Init(y_len - 1);
    for(int i = 1; i < y_len; i++) {
      y_isi[i] = y_[i] - y_[i-1];
    }

    RandPerm(&y_isi);
    permed_pair.y_.Init(y_len);
    permed_pair.y_[0] = y_[0];
    for(int i = 1; i < y_len; i++) {
      permed_pair.y_[i] = permed_pair.y_[i-1] + y_isi[i];
    }

    permed_pair.tau_ = tau_;
    permed_pair.Merge();
  }

};


#endif /* HS_SPIKE_H */
