#ifndef NBR_NETCACHE_H
#define NBR_NETCACHE_H

#include "cache.h"
#include "cachearray.h"
#include "rpc.h"

/**
 * Protocol request for networked block devices.
 */
struct BlockRequest {
  BlockDevice::blockid_t blockid;
  BlockDevice::offset_t begin;
  BlockDevice::offset_t end;
  enum Operation { READ, WRITE, ALLOC, INFO } operation;
  ArrayList<char> payload;

  OT_DEF(BlockRequest) {
    OT_MY_OBJECT(blockid);
    OT_MY_OBJECT(begin);
    OT_MY_OBJECT(end);
    OT_MY_OBJECT(operation);
    OT_MY_OBJECT(payload);
  }
};

/**
 * Protocol response for networked block devices.
 */
struct BlockResponse {
  unsigned int n_block_bytes;
  BlockDevice::blockid_t blockid;
  ArrayList<char> payload;

  OT_DEF(BlockResponse) {
    OT_MY_OBJECT(n_block_bytes);
    OT_MY_OBJECT(blockid);
    OT_MY_OBJECT(payload);
  }
};

class RemoteBlockDeviceBackend
    : public RemoteObjectBackend<BlockRequest, BlockResponse> {
 private:
  BlockDevice *blockdev_;
  uint64 n_reads_;
  uint64 n_read_bytes_;
  uint64 n_writes_;
  uint64 n_write_bytes_;

 public:
  void Init(BlockDevice *device);
  void HandleRequest(const BlockRequest& request, BlockResponse *response);
  void Report(datanode *module);
};

/**
 * A block device sitting on another computer.
 *
 * Individual instances of this object are not thread safe, at least not
 * yet.
 */
class HashedRemoteBlockDevice
    : public BlockDevice {
 private:
  int channel_;
  int my_rank_;
  int n_machines_;
  BlockDevice *local_device_;
  RemoteBlockDeviceBackend server_;

 public:
  static const int MASTER_RANK = 0;

 public:
  HashedRemoteBlockDevice() {
    DEBUG_ONLY(channel_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(n_machines_ = BIG_BAD_NUMBER);
    DEBUG_POISON_PTR(local_device_);
  }

  void Init(int channel_in, int my_rank, int n_machines_in);
  void ConnectToMaster();
  void SetLocalDevice(BlockDevice *local_device_in);
  virtual void Read(blockid_t blockid,
      offset_t begin, offset_t end, char *data);
  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data);
  virtual blockid_t AllocBlocks(blockid_t n_blocks_to_alloc);

  Channel *server() {
    return &server_;
  }
  int channel() const {
    return channel_;
  }

 private:
  int RankHash_(blockid_t block) {
    return block % n_machines_;
  }
  int LocalBlockId_(blockid_t block) {
    return block / n_machines_;
  }
};

template<typename T>
class SimpleDistributedCacheArray : public CacheArray<T> {
 private:
  SmallCache small_cache_;
  HashedRemoteBlockDevice remote_device_;
  MemBlockDevice local_device_;

 public:
  void Configure(int channel) {
    remote_device_.Init(channel, rpc::rank(), rpc::n_peers());
  }

  void InitMaster(const T& default_obj, unsigned int n_block_elems) {
    CacheArrayBlockHandler<T> *handler = new CacheArrayBlockHandler<T>();
    handler->Init(default_obj);
    local_device_.Init(n_block_elems * handler->n_elem_bytes());
    remote_device_.SetLocalDevice(&local_device_);
    small_cache_.Init(&remote_device_, handler, BlockDevice::M_CREATE);
    CacheArray<T>::Init(&small_cache_, BlockDevice::M_CREATE);
    rpc::Register(channel(), server());
  }

  void InitWorker() {
    CacheArrayBlockHandler<T> *handler = new CacheArrayBlockHandler<T>();
    remote_device_.ConnectToMaster();
    local_device_.Init(remote_device_.n_block_bytes());
    remote_device_.SetLocalDevice(&local_device_);
    handler->InitFromDevice(&remote_device_);
    small_cache_.Init(&remote_device_, handler, BlockDevice::M_APPEND);
    CacheArray<T>::Init(&small_cache_, BlockDevice::M_APPEND);
    rpc::Register(channel(), server());
  }

  void Sync(BlockDevice::mode_t mode) {
    //fprintf(stderr, "%p old blocks = %d\n", this, small_cache_.n_blocks());
    //fprintf(stderr, "old rpc blocks = %d\n", remote_device_.n_blocks());
    CacheArray<T>::Flush(true);
    (void) small_cache_.AllocBlocks(0);
    //fprintf(stderr, "new blocks = %d\n", small_cache_.n_blocks());
    //fprintf(stderr, "new rpc blocks = %d\n", remote_device_.n_blocks());
    CacheArray<T>::Grow();
    small_cache_.Remode(mode);
    CacheArray<T>::Remode(mode);
  }

  Channel *server() {
    return remote_device_.server();
  }

  int channel() const {
    return remote_device_.channel();
  }
};

#endif
