#include "new_hmm.h"

int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, &new_hmm_doc);

  HMM hmm;
  Vector pi; // prior state probabilities
  Matrix a;  // transition probabilities
  ArrayList<distribution> &state_distributions = hmm.state_distributions;
  dataset *datasets = hmm.datasets;
  Matrix all_data[K];
  
  hmm.pi.Init(N);
  pi.Alias(hmm.pi);
  
  hmm.a.Init(N, N);
  a.Alias(hmm.a);
  


  
  int i, k;



  // set constants allocate data structures


  state_distributions.Init(N);


  // load in data 
  
  char *filename = (char *)malloc(100 * sizeof(char));
  
  for(k = 0; k < K; k++) {
    
    if(k < 10) {
      sprintf(filename, "handwriting_data/a0%d.csv", k);
    }
    else {
      sprintf(filename, "handwriting_data/a%d.csv", k);
    }

    data::Load(filename, &(all_data[k]));
    
    datasets[k].T = all_data[k].n_cols();
    
  }
  
  D = all_data[0].n_rows();

  
  
  /*data::Load("generated_data.csv", &(all_data[0]));
  datasets[0].T = all_data[0].n_cols();
  data::Load("generated_data.csv", &(all_data[1]));
    datasets[1].T = all_data[1].n_cols();*/
  


  int *viterbi_paths[K];
  for(k = 0; k < K; k++) {
    viterbi_paths[k] = (int*)malloc(datasets[k].T * sizeof(int));
  }


  int sum_T = 0;
  for(k = 0; k < K; k++) {
    sum_T += datasets[k].T;
    /*printf("sequence %d:\n", k);
      display_matrix_no_label(all_data[k]); // massive */
  }
  
  ln_sum_T = log(((double)sum_T));


  for(k = 0; k < K; k++) {
    int T = datasets[k].T;

    datasets[k].b.Init(N, T);
    datasets[k].alpha.Init(N, T);
    datasets[k].beta.Init(N, T);
    datasets[k].xi_sum_over_t.Init(N, N);
    datasets[k].xi_self.Init(N, T - 1);
    datasets[k].gamma.Init(N, T);
    datasets[k].gamma_sum_over_t.Init(N);
    datasets[k].gamma_t_sum_over_i.Init(T);
    datasets[k].c.Init(T);
  }

  for(i = 0; i < N; i++) {
    state_distributions[i].mu.Init(D);
    state_distributions[i].sigma.Init(D);
  }



  
  // initialize pi, a, b

  if(HMM_TYPE == ERGODIC) {
    
    pi.SetAll(1 / ((double)N)); // set pi to be a uniform prior
    a.SetAll(1 / ((double)N));   // set a to uniform transition probabilities
    
    // generate initial state distributions using the first observation sequence
    state_distributions =
      init_state_distributions(state_distributions, all_data[0]);
  }
  else if(HMM_TYPE == LEFT_RIGHT) {
    
    init_left_right_pi(&pi);
    init_left_right_a(&a);
    
    // generate initial state distributions using all observation sequences
    state_distributions =
      init_left_right_state_distributions(state_distributions, all_data);
  }
  else {
    printf("error: HMM_TYPE must be set to ERGODIC or LEFT_RIGHT\n");
    exit(1);
  }
  
  
  //   display_vector("pi", pi);
  //   display_matrix("a", a);
  //   display_state_distributions(state_distributions);
  
  
  
  
  
  /*display_state_distributions(state_distributions);
  display_vector("pi", pi);
  display_matrix("a", a);*/
  
  // calculate b (P(x | S_i) using initial mu and sigma
  for(k = 0; k < K; k++) {
    update_b(&(datasets[k].b), all_data[k], state_distributions);
    /*printf("b for sequence %d:\n", k);
      display_matrix_no_label(datasets[k].b);*/ //massive
  }



  //set_phoney_test_values(&pi, &a, &(datasets[0].b));

  
  train_hmm(&hmm, all_data);


  // THE HMM IS NOW TRAINED WITH THE INITIAL TOPOLOGY
  
  free(filename);

  fx_done(root);
}
