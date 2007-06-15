#ifndef NBR_NETCACHE_H
#define NBR_NETCACHE_H

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
  int n_machines_;
  BlockDevice *local_device_;
  RemoteBlockDeviceBackend server_;

 public:
  const int MASTER_RANK = 0;

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
  virtual blockid_t AllocBlock();

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
 public:
  SmallCache small_cache_;
  HashedRemoteBlockDevice remote_device_;
  MemBlockDevice local_device_;

 private:
  void Configure(int channel, int rank, int n_machines) {
    remote_device_.Init(channel, rank, n_machines);
  }

  void InitMaster(const T& default_obj,
      index_t n_elems, unsigned int n_block_elems) {
    CacheArraySchema<T> *handler = new CacheArraySchema<T>;
    handler->Init(default_obj);
    local_device_.Init(n_block_elems * handler->n_elem_bytes());
    remote_device_.SetLocalDevice(&local_device);
    small_cache_.Init(&remote_device_, handler, BlockDevice::CREATE);
    CacheArray<T>::Init(&small_cache_, BlockDevice::CREATE, 0, n_elem);
  }

  void InitWorker() {
    remote_device_.ConnectToMaster();
    local_device_.Init(remote_device_.n_block_bytes());
    remote_device_.SetLocalDevice(&local_device);
    small_cache_.Init(&remote_device_, new CacheArraySchema<T>,
        BlockDevice::MODIFY);
    CacheArray<T>::Init(&small_cache_, BlockDevice::MODIFY);
  }

  void FlushClear(BlockDevice::mode_t mode) {
    Flush();
    mode_ = mode;
    small_cache_.Clear(mode);
  }
};

#endif
