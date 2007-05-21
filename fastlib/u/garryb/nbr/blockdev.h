#ifndef NBR_BLOCKDEV_H
#define NBR_BLOCKDEV_H

#include "fastlib/fastlib_int.h"

class BlockActionHandler {
  FORBID_COPY(BlockActionHandler);
 public:
  BlockActionHandler() {}
  virtual ~BlockActionHandler() {}
  
  virtual void BlockInitFrozen(size_t bytes, char *block) = 0;
  virtual void BlockRefreeze(size_t bytes, const char *old_location, char *block) = 0;
  virtual void BlockThaw(size_t bytes, char *block) = 0;
};

class BlockDevice {
  FORBID_COPY(BlockDevice);

 public:
  typedef uint32 blockid_t;
  typedef uint32 offset_t;
  enum mode_t {
    READ,
    MODIFY,
    CREATE,
    TEMP
  };

 protected:
  blockid_t n_blocks_;
  offset_t n_block_bytes_;

 private:
  Mutex mutex_;

 public:
  BlockDevice() {}
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

  void Read(blockid_t blockid, char *data) {
    Read(blockid, 0, n_block_bytes_, data);
  }
  void Write(blockid_t blockid, const char *data) {
    Write(blockid, 0, n_block_bytes_, data);
  }
  
  virtual void Read(blockid_t blockid,
      offset_t begin, offset_t end, char *data) = 0;
  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data) = 0;
  virtual blockid_t AllocBlock() = 0;
  virtual void Close() {}

 public:
  void Lock() { mutex_.Lock(); }
  void Unlock() { mutex_.Unlock(); }
};

class NullBlockDevice : public BlockDevice {
  FORBID_COPY(NullBlockDevice);
  
 public:
  NullBlockDevice() {}
  ~NullBlockDevice() {}

  void Init(blockid_t n_blocks_in, offset_t n_block_bytes_in) {
    n_blocks_ = n_blocks_in;
    n_block_bytes_ = n_block_bytes_in;
  }
  
  virtual void Read(blockid_t blockid,
      offset_t begin, offset_t end, char *data) {
    abort();
  }
  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data) {
    // ignore the data
  }
  virtual blockid_t AllocBlock() {
    blockid_t blockid = n_blocks_;
    n_blocks_ = blockid + 1;
    return blockid;
  }
  virtual void Close() {}
};

class BlockDeviceWrapper : public BlockDevice {
  FORBID_COPY(BlockDeviceWrapper);
  
 protected:
  BlockDevice *inner_;
  
 public:
  BlockDeviceWrapper() {}
  virtual ~BlockDeviceWrapper() {}
  
  void Init(BlockDevice *inner_in) {
    inner_ = inner_in;
    n_blocks_ = inner_->n_blocks();
    n_block_bytes_ = inner_->n_block_bytes();
  }
  
  virtual void Read(blockid_t blockid,
      offset_t begin, offset_t end, char *data) {
    inner_->Read(blockid, begin, end, data);
  }
  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data) {
    inner_->Write(blockid, begin, end, data);
  }
  virtual blockid_t AllocBlock() {
    blockid_t blockid = inner_->AllocBlock();
    DEBUG_ASSERT(blockid >= n_blocks_);
    n_blocks_ = blockid + 1;
    return blockid;
  }
  virtual void Close() {}
};

class RandomAccessFile {
 private:
  int fd_;

 public:
  RandomAccessFile() {}
  ~RandomAccessFile() {}
  
  void Init(const char *fname, BlockDevice::mode_t mode);

  void Read(off_t pos, size_t len, char *buffer);
  void Write(off_t pos, size_t len, const char *buffer);
  
  void Close();
  
  off_t FindSize() const;
};

class DiskBlockDevice : public BlockDevice {
  FORBID_COPY(DiskBlockDevice);

 private:
  RandomAccessFile file_;

 public:
  DiskBlockDevice() {}
  ~DiskBlockDevice() {}
  
  void Init(const char *fname, mode_t mode, offset_t block_size = 131072);

  void Read(blockid_t blockid, offset_t begin, offset_t end,
     char *data);

  void Write(blockid_t blockid, offset_t begin, offset_t end,
     const char *data);

  blockid_t AllocBlock();

  void Close();
};

#endif
