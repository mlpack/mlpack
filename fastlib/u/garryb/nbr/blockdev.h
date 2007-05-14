#ifndef NBR_BLOCKDEV_H
#define NBR_BLOCKDEV_H

#include "fastlib/fastlib_int.h"

class BlockActionHandler {
  FORBID_COPY(BlockActionHandler);
 public:
  virtual ~BlockActionHandler() {}
  
  virtual void BlockInit(char *block) = 0;
  virtual void BlockRefreeze(char *block) = 0;
  virtual void BlockThaw(char *block) = 0;
};

class BlockDevice {
  FORBID_COPY(BlockDevice);

 public:
  typedef uint32 blockid_t;
  typedef uint32 offset_t;

 protected:
  blockid_t n_blocks_;
  offset_t n_block_bytes_;

 private:
  Mutex mutex_;

 public:
  virtual ~BlockDevice() {}
  
  blockid_t n_blocks() const {
    return n_blocks_;
  }
  offset_t n_block_bytes() const {
    return n_block_bytes_;
  }
  uint64 n_total_bytes() const {
    return uint64(n_blocks_) * n_block_bytes_;
  }

  // todo: rethink the init function
  virtual void Read(blockid_t blockid, char *data) = 0;
  virtual void Write(blockid_t blockid, const char *data) = 0;
  virtual blockid_t AllocBlock() = 0;

 public:
  void Lock() { mutex_.Lock(); }
  void Unlock() { mutex_.Unlock(); }
};

class BlankBlockDevice : public BlockDevice {
  FORBID_COPY(BlankBlockDevice);
  
 private:
  BlockActionHandler *handler_;
  
 public:
  // todo: rethink the init function
  void Init(BlockActionHandler *handler_in) {
    handler_ = handler_in;
  }
  
  virtual void Read(blockid_t blockid, char *data) {
    handler_->BlockInit(data);
  }
  virtual void Write(blockid_t blockid, const char *data) {
    FATAL("Writebacks to a BlankBlockDevice are invalid.");
  }
  virtual blockid_t AllocBlock() {
    blockid_t blockid = n_blocks_;
    n_blocks_ = blockid + 1;
    return blockid;
  }
};

class BlockDeviceWrapper : public BlockDevice {
  FORBID_COPY(BlockDeviceWrapper);
  
 protected:
  BlockDevice *inner_;
  
 public:
  void Init(BlockDevice *inner_in) {
    inner_ = inner_in;
    n_blocks_ = inner_->n_blocks();
    n_block_bytes_ = inner_->n_block_bytes();
  }
  
  virtual void Read(blockid_t blockid, char *data) {
    inner_->Read(blockid, data);
  }
  virtual void Write(blockid_t blockid, const char *data) {
    inner_->Write(blockid, data);
  }
  virtual blockid_t AllocBlock() {
    blockid_t blockid = inner_->AllocBlock();
    n_blocks_ = blockid + 1;
    return blockid;
  }
};

/*
class DiskBlockDevice : public BlockDevice {
  FORBID_COPY(DiskBlockDevice);
  
 private:
  int fd_;
  
 public:
  void Read(blockid_t blockid, char *data);
  void Write(blockid_t blockid, const char *data);
};

void DiskBlockDevice::Read(blockid_t blockid, char *data) {
  copy code from blockio to handle incomplete buffers
}

void DiskBlockDevice::Write(blockid_t blockid, const char *data) {
  copy code from blockio to handle incomplete buffers
}
*/

#endif
