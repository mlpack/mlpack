#ifndef PARSER_ARG_H
#define PARSER_ARG_H

#include <boost/program_options.hpp>
#include "parser_input.h"

namespace boost_po = boost::program_options;

boost_po::variables_map ParseArgs(int argc, char *argv[], learner &learner, boost_po::options_description& desc) {
  // Declare supported options.
  desc.add_options()
    ("help,h","Produce help message")
    ("threads,t", boost_po::value<size_t>(&global.num_threads)->default_value(1), "Number of threads. Default: 1 thread.")
    ("data_train,d", boost_po::value<string>()->default_value(""), "File name of training example set")
    ("opt_method,m", boost_po::value<string>()->default_value("d_sgd"), "Optimization method")
    ("epoches,e", boost_po::value<size_t>(&global.num_epoches)->default_value(0), "Number of training epoches. Default: 0 epoch")
    ("iterations,i", boost_po::value<size_t>(&global.num_iter_res)->default_value(0), "Number of training iterations besides epoches. Default: 0")
    ("reg,r", boost_po::value<int>(&learner.reg)->default_value(2), "Which regularization term to use. Default: 2(squared l2 norm)")
    ("lambda", boost_po::value<double>(&learner.reg_factor)->default_value(1.0), "Regularization factor ('lambda' in avg_loss + lambda * regularization). Default: 1.0")
    ("C,c", boost_po::value<double>(&learner.C)->default_value(1.0), "Cost factor C ('C' in regularization + C*avg_loss). Default: 1.0")
    ("type", boost_po::value<string>()->default_value("classification"), 
                       "Type of learning: classification or regression or others. Default: classification.")
    ("loss_function,l", boost_po::value<string>()->default_value("hinge"), 
                       "Loss function to be used. Default: squared. Available: squared, hinge, logistic and quantile.")
    ("bias", "Add a bias term to examples")
    ("comm", boost_po::value<int>(&global.comm_method)->default_value(1), "How agents communicate with each other. Default: 1(full connected)")
    ("mini_batch,b", boost_po::value<int>(&global.mb_size)->default_value(1), "Size of a mini-batch. Default: 1")
    ("num_port_sources", boost_po::value<size_t>(), "Number of sources for daemon socket input")
    ("predictions,p", boost_po::value<string>(), "File to output predictions")
    ("port", boost_po::value<size_t>(),"Port to listen on")
    ("par_read", "Read data parallelly with training")
    ("calc_loss", "Calculate total loss")
    ("random", "Randomly permute the input examples")
    ("quiet,q", "Don't output diagnostics");

  global.final_prediction_sink = -1;
  global.raw_prediction = -1;
  global.local_prediction = -1;

  boost_po::positional_options_description p;
  
  boost_po::variables_map vm;

  boost_po::store(boost_po::command_line_parser(argc, argv).
	    options(desc).positional(p).run(), vm);

  boost_po::notify(vm);
  
  if (vm.count("help") || argc == 1) {
    cerr << "\n" << desc << "\n";
    exit(1);
  }

  if (vm.count("quiet")) {
    global.quiet = true;
  }
  else {
    global.quiet = false;
  }

  if (vm.count("calc_loss")) {
    global.calc_loss = true;
  }
  else {
    global.calc_loss = false;
  }

  if (vm.count("random")) {
    global.random_input = true;
  }
  else {
    global.random_input = false;
  }

  // parse input training data
  ParserInput(vm);

  if (vm.count("epoches")) {
    global.num_epoches = vm["epoches"].as<size_t>();
  }
  if (global.num_epoches < 0 ) {
    global.num_epoches = 0;
  }

  if (vm.count("iterations")) {
    global.num_iter_res = vm["iterations"].as<size_t>();
  }
  if (global.num_iter_res < 0 ) {
    global.num_iter_res = 0;
  }

  if (vm.count("reg")) {
    learner.reg = vm["reg"].as<int>();
  }

  if (vm.count("opt_method")) {
    global.opt_method = vm["opt_method"].as<string>();
  }

  if (vm.count("threads")) {
    global.num_threads = vm["threads"].as<size_t>();
  }

  string loss_func_str;
  if(vm.count("loss_function")) {
    loss_func_str = vm["loss_function"].as<string>();
  }
  else {
    loss_func_str = "squaredloss";
  }

  if (vm.count("bias")) {
    global.use_bias = true;
  }
  else {
    global.use_bias = false;
  }

  if (vm.count("comm")) {
    global.comm_method = vm["comm"].as<int>();
  }

  if (vm.count("mini_batch")) {
    global.mb_size = vm["mini_batch"].as<int>();
  }
  if (global.mb_size <= 0)
    global.mb_size = 1;

  if (vm.count("predictions")) {
    if (!global.quiet)
      cerr << "predictions = " <<  vm["predictions"].as<string>() << endl;
    if ( strcmp(vm["predictions"].as<string>().c_str(), "stdout") == 0 )
      global.final_prediction_sink = 1;//stdout
    else {
      const char* fstr = (vm["predictions"].as< string >().c_str());
      global.final_prediction_sink = fileno(fopen(fstr,"w"));
      if (global.final_prediction_sink < 0)
	cerr << "Error opening the predictions file: " << fstr << endl;
    }
  }

  if (vm.count("type")) {
    learner.type = vm["type"].as<string>();
  }
  else {
    learner.type = "classification";
  }
  learner.loss_func = getLossFunction(loss_func_str, 0.1);
  learner.loss_name = learner.loss_func->getName();
  learner.num_threads = global.num_threads;
  learner.num_epoches = global.num_epoches;

  // initialize weight vectors and messages carried by each thread
  learner.w_vec_pool = (SVEC**)malloc(learner.num_threads * sizeof(SVEC*));
  learner.msg_pool = (SVEC**)malloc(learner.num_threads * sizeof(SVEC*));
  learner.total_loss_pool = (double*)malloc(learner.num_threads * sizeof(double));
  learner.total_misp_pool = (size_t*)malloc(learner.num_threads * sizeof(size_t));
  // for OGD
  learner.bias_pool = (double*)malloc(learner.num_threads * sizeof(double));
  learner.t_pool = (double*)malloc(learner.num_threads * sizeof(double));
  learner.scale_pool = (double*)malloc(learner.num_threads * sizeof(double));

  learner.num_used_exp = (size_t*)malloc(learner.num_threads * sizeof(size_t));
  
  return vm;
}

#endif
