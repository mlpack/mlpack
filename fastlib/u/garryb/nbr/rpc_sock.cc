things to think about
 - how to shut down cleanly and respond to signals
   - ssh within the c code or ssh externally?
     - try within c code
   - failures
     - use ssh to detect return code
     - if ssh fails, emit an error message and kill yourself
 - fastexec integration

in the far future, abstract out:
 - topology with scatter/gather/reduce mechanics

resolve address - getaddrinfo
register port numbers

namespace RpcImpl {
  void Init();
  void SendReceive(int channel, int destination, ArrayList<char> *data);
  // void Reduce(int channel, const ArrayList<char>& data); needs template
  void Broadcast(int channel, const ArrayList<char>& data);
  void Barrier();
  void Done();
};

class RpcSockImpl {
 private:
  struct Destination {
    int out_fd;
    int in_fd;
    uint32 addr;
    uint16 port;

    Destination()
     : out_fd(-1)
     , in_fd(-1) {}

    ~Destination() {
      if (out_fd >= 0) {
        (void) close(out_fd);
      }
      if (in_fd >= 0) {
        (void) close(out_fd);
      }
    }
  };

 private:
  static RpcSockImpl instance;

 private:
  struct datanode *module_;
  int rank_;
  int n_;
  uint16 port_;
  ArrayList<int> children_;
  int listen_fd_;
  ArrayList<Destination> machines_;

 public:
  void Init();

 private:
  void CalcChildren_();
};

void RpcSockImpl::Init() {
  module_ = fx_submodule(fx_root, "rpc", "rpc");
  rank_ = fx_param_int(module_, "rank", "0");
  n_ = fx_param_int_req(module_, "n");
  port_ = fax_param_int(module_, "port", 31415);

  CalcChildren_();
  Listen_();
  StartPollingThread_();
  InfectChildren_();

  if (rank_ == 0) {
    fx_timer_start(module_, "resolve_ip");
    resolve all the ip addresses
    fx_timer_stop(module_, "resolve_ip");
    broadcast ip addresses to children
  } else {
    
  }
}

void RpcSockImpl::CalcChildren_() {
  int m = min(unsigned(n_ - rank_ - 1), unsigned((~n_) & (n_-1))) + 1;

  for (int i = 1; i < m; i *= 2) {
    *children_.AddBack() = rank_ + m;
  }
}

void RpcSockImpl::Listen_() {
  struct sockaddr_in my_address;

  listen_fd_ = socket(AF_INET, SOCK_STREAM, PF_INET); // last param 0?
  mem::Zero(&my_address);
  my_address.sin_family = AF_INET;
  my_address.sin_port = htons(port_);
  my_address.sin_addr.s_addr = htonl(INADDR_ANY);

  if (0 > bind(listen_fd_, (struct sockaddr*)&my_address, sizeof(my_address))) {
    FATAL("Could not bind to selected port %d", port_);
  }

  if (0 > listen(listen_fd_, 10)) {
    FATAL("listen() failed, port %d", port_);
  }
}

void RpcSockImpl::StartPollingThread_() {
  /* Create another thread that polls. */
}

void RpcSockImpl::InfectChildren_() {
  open machine file
  for (int i = children_.size(); --i;) {
    int child_rank = children_[i];
    const char *machinename = machines[i];
    create ssh string to execute
    fork process
    if ssh fails, send a signal to the main process
  }
}

void RpcSockImpl::Cleanup_() {
  StopPollingThread_();
  close(listen_fd_);
  machines_.Resize(0); // automatically calls their destructors
}
