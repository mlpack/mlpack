#ifndef PARSER_INPUT_H
#define PARSER_INPUT_H

#include <boost/program_options.hpp>
#include <netdb.h>
#include <netinet/tcp.h>

#include "parallel.h"

namespace boost_po = boost::program_options;

void ParserInput(boost_po::variables_map &vm) {
  // prepare to fetch data from port
  if ( vm.count("port") ) {
    int daemon = socket(PF_INET, SOCK_STREAM, 0);
    if (daemon < 0) {
      cerr << "Can't open a socket!" << endl;
      exit(1);
    }

    int on = 1;
    // set socket options
    if (setsockopt(daemon, SOL_SOCKET, SO_REUSEADDR, (char*)&on, sizeof(on)) < 0) {
      perror("setsockopt SO_REUSEADDR");
    }

    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    short unsigned int port = vm["port"].as<size_t>();
    port = port > 0 ? port : 39524; // default port

    address.sin_port = htons(port);
      
    if (bind(daemon,(sockaddr*)&address, sizeof(address)) < 0) {
      cerr << "Failure to bind!" << endl;
      exit(1);
    }
    cout << "Socket opened at port " << port << "." << endl;

    int source_count = 1;
    
    if (vm.count("num_port_sources")) {
      source_count = vm["num_port_sources"].as<size_t>();
    }

    listen(daemon, source_count);

    //sockaddr_in client_address;
    //socklen_t size = sizeof(client_address);

    // TODO: implement reading from multiple clients

  }
  // prepare to fetch data from file
  if (vm.count("data_train")) {
    global.train_data_fn = vm["data_train"].as<string>();
    if (global.train_data_fn.length() != 0){
      if (!global.quiet) {
	cerr << "Input data from " << global.train_data_fn << "... ";
      }
    }
  }

}

int space_or_null(int c) {
  if (c==0)
    return 1;
  return isspace((unsigned char)c);
}

/* Scan through input file and count # of examples(lines), maximum # of
   features per example, and the size of the longest line. */
void CountFile(string fn, size_t &num_examples, size_t &max_features_example, size_t &max_length_line) 
{
  FILE *fp;
  int ic;
  char c;
  size_t current_length, current_fe;
  
  if ((fp = fopen (fn.c_str(), "r")) == NULL) {
    cerr << "Cannot open input example file: " << fn << ". Porgram stopped!"<< endl;
    exit (1);
  }
  current_length = 0;
  current_fe = 0;
  max_length_line = 0;
  num_examples = 1;
  max_features_example = 0;
  while ((ic=getc(fp)) != EOF) {
    c = (char)ic;
    current_length ++;
    if (space_or_null((int)c)) {
      current_fe ++;
    }
    if (c == '\n') {
      num_examples ++;
      if (current_length > max_length_line) {
	max_length_line = current_length;
      }
      if (current_fe > max_features_example) {
	max_features_example = current_fe;
      }
      current_length = 0;
      current_fe = 0;
    }
  }
  fclose(fp);
}

/* Read features from a line buffer */
int ReadFeatures(char *line, FEATURE *feat, T_LBL *label, size_t &num_feats, size_t &max_features_example, char **comment)
{
  unsigned long wpos,pos;
  unsigned long w_idx;
  double w_val;
  int numread;
  char featurepair[1000],junk[1000];
  
  pos = 0;
  (*comment)=NULL;
  while(line[pos]) { // strip comment
    if((line[pos] == '#') && (!(*comment))) {
      line[pos]=0;
      (*comment) = &(line[pos+1]);
    }
    if(line[pos] == '\n') { // strip CR
      line[pos]=0;
    }
    pos++;
  }
  if(!(*comment)) {
    (*comment) = &(line[pos]);
  }

  wpos=0;
  // check, that line starts with target value or zero, but not with feature pair
  if(sscanf(line,"%s", featurepair) == EOF)
    return(0);
  pos=0;
  while((featurepair[pos] != ':') && featurepair[pos]) 
    pos++;
  if(featurepair[pos] == ':') {
    cerr << "Line must start with label or 0!!!" << endl;; 
    cerr << "Line: " << line << endl;
    exit (1); 
  }
  // read the target value
  double lbl;
  if(sscanf(line,"%lf", &lbl) == EOF)
    return(0);
  (*label) = (T_LBL)lbl;
  pos=0;
  while(space_or_null((int)line[pos]))
    pos++;
  while((!space_or_null((int)line[pos])) && line[pos])
    pos++;
  while(((numread=sscanf(line+pos,"%s",featurepair)) != EOF) && 
	(numread > 0) && (wpos<max_features_example)) {
    while(space_or_null((int)line[pos]))
      pos++;
    while((!space_or_null((int)line[pos])) && line[pos])
      pos++;
    if(sscanf(featurepair,"%ld:%lf%s", &w_idx, &w_val, junk)==2) {
      // it is a regular feature
      if(w_idx<=0) { 
	cerr << "Feature numbers must be larger or equal to 1!" << endl;
	cerr << "Line: " << line << endl;
	exit (1); 
      }
      if((wpos>0) && ((feat[wpos-1]).widx >= w_idx)) { 
	cerr << "Features must be in increasing order!" << endl;
	cerr << "Line: " << line << endl;
	exit (1); 
      }
      (feat[wpos]).widx = (T_IDX)w_idx; // feature index starts from 1
      (feat[wpos]).wval = (T_VAL)w_val; 
      wpos++;
    }
    else {
      cout << "Cannot parse feature/value pair!!!" << endl; 
      cerr << featurepair << " in LINE: " << line << endl;
      exit (1); 
    }
  }
  num_feats = wpos;
  
  return(1);
}


/* Batch read all data from a file */
void SerialRead(string &data_fn, EXAMPLE **examples, size_t &num_examples) {
  FILE *fp;
  char *line, *comment;
  size_t max_num_examples, max_features_example, max_length_line;
  size_t dnum=0, wpos, dpos=0, dneg=0, dunlab=0;
  FEATURE *feat_cache;
  T_LBL label_cache;
  size_t i;

  // scan size of input data file
  CountFile(data_fn, max_num_examples, max_features_example, max_length_line);
  max_num_examples --;
  max_features_example --;
  max_length_line += 2;
  (*examples) = (EXAMPLE *)my_malloc( sizeof(EXAMPLE)*max_num_examples ); // feature vectors
  line = (char *)my_malloc( sizeof(char)*max_length_line );
  
  feat_cache = (FEATURE *)my_malloc( sizeof(FEATURE)*max_features_example );

  dnum=0;
  num_examples = 0;

  if ((fp = fopen (data_fn.c_str(), "r")) == NULL) {
    cerr << "Cannot open " << data_fn << ". Porgram stopped!"<< endl;
    exit (1);
  }
  while((!feof(fp)) && fgets(line,(int)max_length_line,fp)) {
    if(line[0] == '#') {
      continue;  // comment line
    }
    for (i=0; i<max_features_example; i++) {
      feat_cache[i].widx = 0;
      feat_cache[i].wval = 0;
    }
    if(!ReadFeatures(line, feat_cache, &label_cache, wpos, max_features_example, &comment)) {
      cout << endl << "Cannot read line " << dnum << " !"<< endl << line << endl;
      exit(1);
    }

    if(label_cache > 0)
      dpos++;
    if (label_cache < 0)
      dneg++;
    if (label_cache == 0)
      dunlab++;

    // setup an example
    CreateExample((*examples)+dnum, feat_cache, label_cache, comment, max_features_example);

    //print_ex((*examples)+dnum);
    
    dnum++;  
    if(!global.quiet) {
      if((dnum % 10000) == 0) {
	cout << dnum << "..";
      }
    }
  }

  fclose(fp);
  free(line);
  free(feat_cache);
  if(!global.quiet) {
    cout << "done. " << dnum << " examples loaded: " << dpos << " positive and " << dneg <<" negative." << endl;
  }
  
  num_examples = dnum;
}

void *ParReadThread(void *) {
  // TODO
  return NULL;
}

/* Read data parallelly while training */
void ParallelRead(size_t num_threads) {
  pthread_create(&global.par_read_thread, NULL, ParReadThread, NULL);
}

void ReadData(boost_po::variables_map &vm) {
  size_t i, j, left_ct;
  if ( vm.count("par_read") ) {
    ParallelRead(global.num_threads);
    //ring_size = 1 << 11; // 2048
    //parsed_ct = 0;
  }
  else {
    SerialRead(global.train_data_fn, &train_exps, num_train_exps);
    if (num_train_exps == 0) {
      cout << "0 input samples!" << endl;
      exit(1);
    }
    parsed_ct = num_train_exps;

    /* To mimic the online learning senario, in each epoch, 
       we randomly permutate the training set, indexed by old_from_new */
    old_from_new = (size_t *)my_malloc(sizeof(size_t) * parsed_ct);
    for (i=0; i<parsed_ct; i++) {
      old_from_new[i] = i; 
    }
    for (i=0; i<parsed_ct; i++) {
      j = rand() % parsed_ct;
      swap(old_from_new[i], old_from_new[j]);
    }

    if (global.num_iter_res >= parsed_ct) {
      global.num_epoches = global.num_epoches + (size_t)(global.num_iter_res / parsed_ct);
      global.num_iter_res = global.num_iter_res % parsed_ct;
    }
    left_ct = global.num_threads*global.mb_size - (global.num_epoches * parsed_ct + global.num_iter_res) % (global.num_threads*global.mb_size);
    global.num_iter_res = global.num_iter_res + left_ct;
    cout << "n_epo= " <<global.num_epoches << ", n_iter_res= " << global.num_iter_res << endl;

    if (vm.count("C")) {
      l1.C = vm["C"].as<double>();
      if (l1.C <= 0.0) {
	cout << "Parameter C should be positive!" << endl;
	exit(1);
      }
      l1.reg_factor = 1.0 / (l1.C * num_train_exps);
      //cout << "C= " << l1.C << ", lambda= " << l1.reg_factor << endl << endl;
    }
    else if (vm.count("lambda")) {
      l1.reg_factor = vm["lambda"].as<double>();
      if (l1.reg_factor <= 0.0) {
	cout << "Parameter lambda should be positive!" << endl;
	exit(1);
      }
      l1.C = 1.0 / (l1.reg_factor * num_train_exps);
      //cout << "lambda= " << l1.reg_factor << ", C= " << l1.C << endl << endl;
    }
  }

  //delay_indicies = (size_t*) calloc(l1.num_threads, sizeof(size_t));
  //threads_to_use = (size_t*) calloc(ring_size, sizeof(size_t));
  //delay_ring = (EXAMPLE**)calloc(ring_size, sizeof(EXAMPLE*));
  //for (i = 0; i < ring_size; i++) {
  //  delay_ring[i] = NULL;
  //}
}

void FinishData() {
  for(size_t i=0; i<num_train_exps; i++) {
    FreeExample(train_exps+i);
  }
  free(train_exps);
}

int GetImmedExample(EXAMPLE** x_p, size_t tid, learner &l) {
  size_t i, j;
  pthread_mutex_lock(&examples_lock);
  if (epoch_ct < global.num_epoches) {
    size_t ring_index = used_ct % parsed_ct;
    if (ring_index == (parsed_ct-1)) { // one epoch finished
      epoch_ct ++;
      /* To mimic the online learning senario, in each epoch, 
	 we randomly permutate the training set, indexed by old_from_new */
      for (i=0; i<parsed_ct; i++) {
	old_from_new[i] = i; 
      }
      for (i=0; i<parsed_ct; i++) {
	j = rand() % parsed_ct;
	swap(old_from_new[i], old_from_new[j]);
      }
    }
    train_exps[old_from_new[ring_index]].in_use = true;
    used_ct ++;
    l.num_used_exp[tid] ++;
    (*x_p) = train_exps + old_from_new[ring_index];
    pthread_mutex_unlock(&examples_lock);
    return 1;
  }
  else if (iter_res_ct < global.num_iter_res) {
    size_t ring_index = used_ct % parsed_ct;
    train_exps[old_from_new[ring_index]].in_use = true;
    used_ct ++;
    l.num_used_exp[tid] ++;
    (*x_p) = train_exps + old_from_new[ring_index];
    iter_res_ct ++;
    pthread_mutex_unlock(&examples_lock);
    return 1;
  }
  else {
    pthread_mutex_unlock(&examples_lock);
    return 0;
  }
}

/*
example* GetDelayedExample(size_t thread) {
  if (delay_indicies[thread] == master_index) { // no delayed example available
    return NULL;
  }
  else {
    size_t index = delay_indicies[thread] % ring_size;
    example* ret = delay_ring[index];
    
    delay_indicies[thread]++;
    
    pthread_mutex_lock(&ret->lock);
    if (--threads_to_use[index] == 0) {
      pthread_mutex_lock(&example_delay);
      delay_ring[index] = NULL;
      pthread_mutex_unlock(&example_delay);
    }
    pthread_mutex_unlock(&ret->lock);
    return ret;
  }
}
*/

/*
bool thread_done(size_t thread)
{
  bool ret;
  pthread_mutex_lock(&example_delay);
  ret = (delay_indicies[thread] == master_index);
  pthread_mutex_unlock(&example_delay);
  return ret;
}
*/


#endif
