/**
 * @file blockdev.h
 *
 * Block-device abstractions, such as a disk-based block device.
 */

#ifndef THOR_BLOCKDEV_H
#define THOR_BLOCKDEV_H

#include "fastlib/base/base.h"
#include "fastlib/col/intmap.h"
#include "fastlib/col/string.h"
#include "fastlib/fx/fx.h"
#include "fastlib/par/thread.h"

/**
 * An abstracted block device.
 *
 * A block device is a storage element that allows reading and reading whole
 * blocks or subsets of a block.  A file on disk is a canonical block device,
 * but THOR has distributed caches which function as one large block device.
 */
class BlockDevice {
  FORBID_ACCIDENTAL_COPIES(BlockDevice);

 public:
  /**
   * A type representing the ID of a block.
   *
   * Wherever you see blockid_t you know that it is referring to a block
   * identifier.
   */
  typedef int32 blockid_t;
  /**
   * A type representing the ID of a block.
   *
   * Wherever you see offset_t you know that it is referring to an
   * offset, in bytes, within a block.
   */
  typedef int32 offset_t;

  /** Flags used to define modes -- mostly not used anymore. */
  enum modeflag_t {
    /** Flag whether a mode requires reading, if this is unset, things read
     * for the first time get a default value. */
    F_READ      = 0x01,
    /** Flag whether a mode allows writes. */
    F_WRITABLE  = 0x02,
    /** Flag whether a mode requires writes. */
    F_WRITE     = 0x04|F_WRITABLE,
    F_DYNAMIC   = 0x08,
    F_INIT      = 0x10,
  };

  /**
   * Mode type used when opening or accessing a block device.
   *
   * NOTE: At one time there was a plan to make modes very important.
   * They are no longer very important.
   */
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
  /** Number of blocks. */
  blockid_t n_blocks_;
  /** Number of bytes in a block. */
  offset_t n_block_bytes_;

 public:
  BlockDevice() {}
  virtual ~BlockDevice() {}
  
  /** Gets the number of blocks this device knows about. */
  blockid_t n_blocks() const {
    return n_blocks_;
  }
  /** Gets the block size, in bytes. */
  offset_t n_block_bytes() const {
    return n_block_bytes_;
  }
  /** Gets the total number of bytes in the block device. */
  uint64 n_total_bytes() const {
    return uint64(n_blocks_) * n_block_bytes_;
  }

  /** Reads a block. */
  void Read(blockid_t blockid, char *data) {
    Read(blockid, 0, n_block_bytes_, data);
  }
  /** Writes to a block. */
  void Write(blockid_t blockid, const char *data) {
    Write(blockid, 0, n_block_bytes_, data);
  }

  /**
   * Reads [begin,end) from the specified block.
   *
   * @param blockid the number of the block
   * @param begin the first byte within the block to read
   * @param end one past the last byte desired
   * @param data where to store the bytes
   */
  virtual void Read(blockid_t blockid,
      offset_t begin, offset_t end, char *data) = 0;
  /**
   * Writes [begin,end) to the specified block.
   *
   * @param blockid the number of the block
   * @param begin the first byte within the block to write
   * @param end one past the last byte desired
   * @param data where to copy data from
   */
  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data) = 0;
  /**
   * Allocates a number of contiguous new blocks.
   *
   * @param diff the number to allocate
   * @return the block ID of the first block in the contiguous chunk
   */
  virtual blockid_t AllocBlocks(blockid_t diff);
};

/**
 * A null block device that silently ignores writes and fails on reads.
 */
class NullBlockDevice : public BlockDevice {
  FORBID_ACCIDENTAL_COPIES(NullBlockDevice);
  
 public:
  NullBlockDevice() {}
  ~NullBlockDevice() {}

  /**
   * Initializes to the specified dimensions.
   */
  void Init(blockid_t n_blocks_in, offset_t n_block_bytes_in) {
    n_blocks_ = n_blocks_in;
    n_block_bytes_ = n_block_bytes_in;
  }

  /**
   * Aborts on read.
   */
  virtual void Read(blockid_t blockid,
      offset_t begin, offset_t end, char *data) {
    FATAL("Cannot read from a null block device.");
  }
  /**
   * Ignores writes.
   */
  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data) {}
};

/**
 * An encapsulation low-level random-access read-write.
 *
 * Basically just wraps files in stdio.
 */
class RandomAccessFile {
 private:
  /** The file descriptor open. */
  int fd_;
  /** The BlockDevice mode of the file. */
  mode_t mode_;
  /** The filename open. */
  String fname_;
  /** Mutex to protect the lseek-read sequence. */
  Mutex mutex_;

 public:
  RandomAccessFile() {}
  ~RandomAccessFile() { Close(); }
  
  /**
   * Opens a filename with the specified mode.
   *
   * If the filename is NULL a temporary file will be opened.
   * The directory for temporary files defaults to /tmp, but may be
   * modified through the parameter @c tmp_dir in @c FX_ROOT.
   * If a temporary file is created, it won't show up in @c ls, because
   * in UNIX it is best to delete a file right after you are opening it,
   * ensuring the file will be deleted no matter how the program terminates.
   *
   * @param fname the filename, or NULL for a temporary file
   * @param mode the mode to open the file for
   */
  void Init(const char *fname, BlockDevice::mode_t mode);

  void Read(off_t pos, size_t len, char *buffer);
  void Write(off_t pos, size_t len, const char *buffer);

  /**
   * Explicitly closes the file.
   *
   * The dstructor will, however, close the file automatically.
   */
  void Close();

  /**
   * Determines the file's size via a system call.
   */
  off_t FindSize() const;
};

/**
 * An on-disk block device.
 */
class DiskBlockDevice : public BlockDevice {
  FORBID_ACCIDENTAL_COPIES(DiskBlockDevice);

 private:
  mode_t mode_;
  RandomAccessFile file_;

 public:
  DiskBlockDevice() {}
  virtual ~DiskBlockDevice();

  /**
   * Opens a disk file with the given mode.
   *
   * If fname is NULL and mode is M_TEMP and the filename is NULL, a
   * temporary filename is automatically generated.  See RandomAccessFile.
   */
  void Init(const char *fname, mode_t mode, offset_t block_size);

  void Read(blockid_t blockid, offset_t begin, offset_t end,
     char *data);

  void Write(blockid_t blockid, offset_t begin, offset_t end,
     const char *data);
};

/**
 * A memory-based block device.
 */
class MemBlockDevice : public BlockDevice {
 private:
  DenseIntMap<char*> blocks_;

 public:
  MemBlockDevice() {}
  virtual ~MemBlockDevice();

  /** Initializes with a specific block size. */
  void Init(offset_t block_size);

  virtual void Read(blockid_t blockid,
      offset_t begin, offset_t end, char *data);
  virtual void Write(blockid_t blockid,
      offset_t begin, offset_t end, const char *data);
};

/**
 * Input-output statistics such as number and bytes of reads and writes.
 */
class IoStats {
 private:
  uint64 n_read_bytes_;
  uint64 n_write_bytes_;
  uint n_reads_;
  uint n_writes_;
  
  OT_DEF(IoStats) {
    OT_MY_OBJECT(n_read_bytes_);
    OT_MY_OBJECT(n_write_bytes_);
    OT_MY_OBJECT(n_reads_);
    OT_MY_OBJECT(n_writes_);
  }

 public:
  /** Initializes to no reads or writes. */
  void Init() {
    Reset();
  }

  /** Gets the number of read operations performed. */
  uint n_reads() const {
    return n_reads_;
  }
  /** Gets the number of bytes read. */
  uint64 n_read_bytes() const {
    return n_read_bytes_;
  }
  /** Gets the number of write operations performed. */
  uint n_writes() const {
    return n_writes_;
  }
  /** Gets the number of bytes written. */
  uint64 n_write_bytes() const {
    return n_write_bytes_;
  }
  /** Gets the number of read or write operations total. */
  uint n_io() const {
    return n_reads_ + n_writes_;
  }
  /** Gets the number bytes read or writen total. */
  uint64 n_io_bytes() const {
    return n_read_bytes_ + n_write_bytes_;
  }

  /** Adds another set of counts to this. */
  void Add(const IoStats& other) {
    n_reads_ += other.n_reads_;
    n_writes_ += other.n_writes_;
    n_read_bytes_ += other.n_read_bytes_;
    n_write_bytes_ += other.n_write_bytes_;
  }

  /** RecordS a single read operation of the specified number of bytes. */
  void RecordRead(uint64 n_bytes) {
    n_read_bytes_ += n_bytes;
    n_reads_++;
  }

  /** Records a single write operation of the specified number of bytes. */
  void RecordWrite(uint64 n_bytes) {
    n_write_bytes_ += n_bytes;
    n_writes_++;
  }

  /**
   * Reset all counts to zero.
   */
  void Reset() {
    n_reads_ = n_writes_ = 0;
    n_read_bytes_ = n_write_bytes_ = 0;
  }

  /**
   * Reports the statistics gathered, and include information based on the
   * percentage of total data.
   */
  void Report(BlockDevice::offset_t n_block_bytes,
      BlockDevice::blockid_t n_blocks,
      datanode *module) const;

  /**
   * Reports only the tallies.
   */
  void Report(datanode *module) const;
};

#endif
