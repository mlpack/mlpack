#ifndef NBR_BLOCKDEV_H
#define NBR_BLOCKDEV_H

#include "fastlib/fastlib_int.h"

class Schema {
  FORBID_COPY(Schema);
 public:
  Schema() {}
  virtual ~Schema() {}

  virtual void InitFromHeader(size_t header_size, char *header) = 0;
  virtual void WriteHeader(size_t header_size, char *header) = 0;

  virtual void BlockInitFrozen(size_t bytes, char *block) = 0;
  virtual void BlockRefreeze(
      size_t bytes, const char *old_location, char *block) = 0;
  virtual void BlockThaw(size_t bytes, char *block) = 0;
};

class BlockDevice {
  FORBID_COPY(BlockDevice);

 public:
  typedef uint32 blockid_t;
  typedef uint32 offset_t;

  enum modeflag_t {
    F_READ      = 0x01,
    F_WRITABLE  = 0x02,
    F_WRITE     = 0x04|F_WRITABLE,
    F_DYNAMIC   = 0x08,
    F_INIT      = 0x10,
  };
  enum mode_t {
    /** Read existing data. */
    M_READ = F_READ,
    /** Overwrite existing data, as if the original data didn't exist. */
    M_OVERWRITE = F_WRITE,
    /** Modify existing data. */
    M_MODIFY = F_READ|F_WRITE,
    /** Create from scratch. */
    M_CREATE = F_INIT|F_WRITE|F_DYNAMIC,
    /** Add new data to the end. */
    M_APPEND = F_WRITE|F_DYNAMIC,
    /** Use the block device only as temporary storage. */
    M_TEMP = F_INIT|F_WRITABLE|F_DYNAMIC,
    /** Use prexisting data, but allow data to be used as scratch space. */
    M_DESTROY = F_READ|F_WRITABLE,
  };

  /**
   * Checks if a mode is creating a brand new "file," and header information
   * must be written.
   *
   * @param mode the mode to check
   */  
  static bool need_init(mode_t mode) {
    return (mode & F_INIT) != 0;
  }
  /**
   * Checks if a mode requires reads.
   *
   * If a mode does not require reads, block devices may return garbage, and
   * caches should initialize the block to its default values.
   *
   * @param mode the mode to check
   */
  static bool need_read(mode_t mode) {
    return (mode & F_READ) != 0;
  }
  /**
   * Check if a mode requires writes.
   *
   * In modes that require writes, data must be saved at the end of lifetime.
   * Modes may allow writes but not require them (see below).
   *
   * @param mode the mode to check
   */
  static bool need_write(mode_t mode) {
    return (mode & F_WRITE) != 0;
  }
  /**
   * Checks if a mode allows writes.
   *
   * In modes that allow writes and allows reads, data that is written must
   * be preserved if it is flushed from cache and read back again.
   * However it need not be stored after the cache is closed unless need_write
   * is true.
   *
   * @param mode the mode to check
   */
  static bool can_write(mode_t mode) {
    return (mode & F_WRITABLE) != 0;
  }
  /**
   * Checks if a mode requires read-after-write to be true.
   *
   * Effectively checks can_write && need_read.
   *
   * @param mode the mode to check
   */
  static bool need_writeread(mode_t mode) {
    return (mode & (F_READ|F_WRITABLE)) == (F_READ|F_WRITABLE);
  }
  /**
   * Checks if a mode is a dynamic mode.
   *
   * In dnyamic modes (create, append, temp), resizing is possible, and
   * writes are always performed on the entire-block granularity.  In static
   * modes, writes are confined to contiguous regions on a sub-block
   * granularity.
   *
   * @param mode the mode to check
   */
  static bool is_dynamic(mode_t mode) {
    return (mode & F_DYNAMIC) != 0;
  }

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
  virtual blockid_t AllocBlocks(blockid_t diff);

 public:
  void Lock() { mutex_.Lock(); }
  void Unlock() { mutex_.Unlock(); }
  
 protected:
  virtual void HandleGrowth() {}
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
      offset_t begin, offset_t end, char *data);
  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data);
  virtual blockid_t AllocBlocks(blockid_t i);
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
  mode_t mode_;
  RandomAccessFile file_;
  String path_;

 public:
  DiskBlockDevice() {}
  virtual ~DiskBlockDevice();

  void Init(const char *fname, mode_t mode, offset_t block_size = 131072);

  void Read(blockid_t blockid, offset_t begin, offset_t end,
     char *data);

  void Write(blockid_t blockid, offset_t begin, offset_t end,
     const char *data);

  blockid_t AllocBlock();
};

class MemBlockDevice : public BlockDevice {
 private:
  DenseIntMap<char*> blocks_;

 public:
  MemBlockDevice() {}
  virtual ~MemBlockDevice();

  void Init(offset_t block_size);

  virtual void Read(blockid_t blockid,
      offset_t begin, offset_t end, char *data);
  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data);
  virtual blockid_t AllocBlock();
};

#endif
