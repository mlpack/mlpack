// Implementation for Pole

#include "pole.h"

///////////////
// Construction
///////////////
Pole::Pole() : batch_(false){
}

///////////////
// Destruction
///////////////
Pole::~Pole(void) {
  if (L_)
    delete L_;
}
  
////////////////////////////////////////////////////////
// Parse input arguments using boost's program_options
////////////////////////////////////////////////////////
void Pole::ParseArgs(int argc, char *argv[]) {
  namespace po = boost::program_options;

  // determine optimization method first
  if (argc > 1) {
    m_["method"] = "null";
    m_["kernel"] = "null";
    for (int i=1; i<argc; i++) {
      if (string(argv[i]) == "-m" || string(argv[i]) == "--method") {
	if (i < argc-1) {
          m_["method"] = string(argv[i+1]);
	}
	else {
          m_["method"] = "";
	}
      }
      else if (string(argv[i]) == "-k" || string(argv[i]) == "--kernel") {
	if (i < argc-1) {
          m_["kernel"] = string(argv[i+1]);
	}
	else {
          m_["kernel"] = "";
	}
      }
      else if (string(argv[i]) == "-h" || string(argv[i]) == "--help") {
        m_["method"] = "help";
	break;
      }
    }

    if (m_["method"] == "null") {
      cout << "ERROR! Optimization method needs to be specified!" << endl;
      exit(1);
    }
    else if (m_["method"] == "ogd") {
      L_ = new OGD;
    }
    else if (m_["method"] == "ogdk") {
      if (m_["kernel"] == "null") {
        cout << "ERROR! Kernel type needs to be specified for OGDK!" << endl;
        exit(1);
      }
      else if (m_["kernel"] == "linear") {
        L_ = new OGDK<LinearKernel>;
      }
      else if (m_["kernel"] == "gaussian") {
        L_ = new OGDK<RBFKernel>;
      }
      else {
        cout << "ERROR! Kernel type needs to be [lienar, gaussian]" << endl;
        exit(1);
      }
    }
    else if (m_["method"] == "oeg") {
      L_ = new OEG;
    }
    else if (m_["method"] == "dwm_i" || m_["method"] == "dwm_a") {
      L_ = new WM;
    }
    else if (m_["method"] == "help") {
      L_ = new Learner;
    }
    else {
      cout << "ERROR! Optimization method needs to be [ogd, ogdk, oeg, dwm_i, or dwm_a]!" << endl;
      exit(1);
    }
  }
  else {
    L_ = new Learner;
  }

  po::options_description desc("POLE (Parallel Online Learning Experiments) options");
  // Declare supported options
  desc.add_options()
    ("help,h","Produce help message")
    ("method,m", po::value<string>(&L_->opt_name_)->default_value(""), 
     "Optimization method [ogd, ogdk, oeg, dwm_i, or dwm_a].")
    ("batch", po::value<bool>(&batch_)->default_value(false),
     "Online leaing or Batch learning. Default: Online.")
    ("threads", po::value<size_t>(&L_->n_thread_)->default_value(1), 
     "Number of threads. Default: 1 thread.")
    ("data_learn,d", po::value<string>(&L_->fn_learn_)->default_value(""), 
     "File name of training example set.")
    ("data_predict,t", po::value<string>(&L_->fn_predict_)->default_value(""), 
     "File name of training example set.")
    ("epoches,e", po::value<size_t>(&L_->n_epoch_)->default_value(0), 
     "Number of training epoches. Default: 0 epoch.")
    ("iterations,i", po::value<size_t>(&L_->n_iter_res_)->default_value(0), 
     "Number of training iterations besides epoches. Default: 0.")
    ("reg,r", po::value<int>(&L_->reg_type_)->default_value(2), 
     "Which regularization term to use. Default: 2(squared l2 norm).")
    ("lambda", po::value<double>(&L_->reg_factor_), 
     "Regularization factor ('lambda' in avg_loss + lambda * regularization). No default value.")
    ("C,c", po::value<double>(&L_->reg_C_)->default_value(1.0), 
     "Cost factor C ('C' in regularization + C*avg_loss). Default: 1.0.")
    ("type", po::value<string>(&L_->type_)->default_value("classification"), 
     "Type of learning: classification or regression or others. Default: classification.")
    ("loss_function,l", po::value<string>(&L_->lf_name_)->default_value("hinge"), 
     "Loss function to be used. Default: squared. Available: squared, hinge, logistic and quantile.")
    ("bias", po::value<bool>(&L_->use_bias_)->default_value(false),
     "Add a bias term to examples.")
    ("experts,p", po::value<size_t>(&L_->n_expert_)->default_value(1), 
     "Number of experts. Default: 0.")
    ("weak_learner", po::value<string>(&L_->wl_name_)->default_value("stump"), 
     "Name of weak learner. Default: decision stump.")
    ("alpha", po::value<double>(&L_->alpha_)->default_value(0.5), 
     "Multiplication factor in Weighte Majority. Default: 0.5.")
    ("kernel,k", po::value<string>(&L_->kernel_name_)->default_value("linear"), 
     "Kernel (linear, rbf). Default: linear kernel.")
    ("sigma", po::value<double>(&L_->sigma_)->default_value(1.0), 
     "Sigma in Gaussian RBF kernel. Default: 1.0.")
    ("comm", po::value<int>(&L_->comm_method_)->default_value(1), 
     "How agents communicate with each other. Default: 1(full connected).")
    ("mini_batch,b", po::value<size_t>(&L_->mb_size_)->default_value(1), 
     "Size of a mini-batch. Default: 1.")
    ("calc_loss", po::value<bool>(&L_->calc_loss_)->default_value(false), 
     "Calculate total loss.")
    ("random", po::value<bool>(&L_->random_data_)->default_value(true), 
     "Randomly permute the input examples.")
    ("read_port", po::value<bool>(&L_->read_port_)->default_value(false), 
     "Read data parallelly with training.")
    ("num_port_sources", po::value<size_t>(&L_->n_source_)->default_value(0), 
     "Number of sources for daemon socket input.")
    ("port", po::value<size_t>(&L_->port_)->default_value(0),
     "Port to listen on.")
    ("log", po::value<size_t>(&L_->n_log_)->default_value(0), 
     "How many log points. Default: 0(no logging).")
    ("verbose,v", po::value<bool>(&L_->v_)->default_value(false), 
     "Verbose debug info.");

  po::positional_options_description p;
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  po::notify(vm);
  if (vm.count("help") || argc == 1) {
    cerr << "\n" << desc << "\n";
    exit(1);
  }
  
  if (L_->lf_name_ == "squared") {
    L_->LF_ = new SquaredLoss();
  }
  else if (L_->lf_name_ == "hinge") {
    L_->LF_ = new HingeLoss();
  }
  else if (L_->lf_name_ == "squaredhinge") {
    L_->LF_ = new SquaredhingeLoss();
  }
  else if (L_->lf_name_ == "logistic") {
    L_->LF_ = new LogisticLoss();
  }
  else if (L_->lf_name_ == "quantile") {
    L_->LF_ = new QuantileLoss(0.5);
  }
  else {
    cout << "Invalid loss function name!" << endl;
    exit(1);
  }

  //L + \lambda/2 \|w\|^2 <=> CL + 1/2 \|w\|^2
  if (vm.count("C")) {
    L_->reg_C_ = vm["C"].as<double>();
    if (L_->reg_C_ <= 0.0) {
      cout << "Parameter C should be positive!" << endl;
      exit(1);
    }
    L_->reg_factor_ = 1.0 / L_->reg_C_;
  }
  if (vm.count("lambda")) {
    L_->reg_factor_ = vm["lambda"].as<double>();
    if (L_->reg_factor_ < 0.0) {
      cout << "Parameter lambda should be non-negative!" << endl;
      exit(1);
    }
    else if (L_->reg_factor_ == 0) {
      cout << "Regularization factor == 0. No regularization imposed!" << endl;
    }
    else {
      L_->reg_C_ = 1.0 / L_->reg_factor_;
    }
  }

  if (L_->random_data_)
    srand(time(NULL));

  ArgsSanityCheck();

}

//////////////////////////
// Check input parameters
//////////////////////////
void Pole::ArgsSanityCheck() {
  if (L_->n_epoch_ < 0 )
    L_->n_epoch_ = 0;

  if (L_->n_iter_res_ < 0 )
    L_->n_iter_res_ = 0;

  if (L_->mb_size_ <= 0) {
    cout << "Mini-batch size should be positive! Using default value: 1." << endl;
    L_->mb_size_ = 1;
  }

  if (L_->alpha_ <=0 || L_->alpha_ >= 1) {
    cout << "In WM, alpha should be within (0,1)! Using default value: 0.5." << endl;
    L_->alpha_ = 0.5;
  }

  if (L_->n_expert_ <= 0) {
    L_->n_expert_ = 1;
    cout << "Number of expert should be positive! Using default value: 1." << endl;
  }

  if (L_->n_log_ <0) {
    L_->n_log_ = 0;
  }

  if (L_->n_log_ >0 && L_->calc_loss_ == false) {
    L_->calc_loss_ = true;
  }
}

void Pole::Run() {
  if (!batch_) { // Online Learning
    L_->OnlineLearn();
  }
  else { // Batch Learning
    L_->BatchLearn();
  }
}
