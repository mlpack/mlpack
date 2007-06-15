#include "blockdev.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

void RandomAccessFile::Init(const char *fname, BlockDevice::mode_t mode) {
  int octal_mode;
  
  path_.Copy(fname);
  
  switch(mode) {
   case BlockDevice::READ:
    octal_mode = O_RDONLY;
    break;
   case BlockDevice::MODIFY:
    octal_mode = O_RDWR;
    break;
   case BlockDevice::CREATE:
   case BlockDevice::TEMP:
    octal_mode = O_RDWR|O_CREAT|O_TRUNC;
    break;
   default: abort();
  }
  
  fd_ = open(fname, octal_mode, 0666);
  if (fd_ <= 0) {
    FATAL("Could not open file '%s'.", fname);
  }
}

void RandomAccessFile::Close() {
  close(fd_);
}

void RandomAccessFile::Write(off_t pos, size_t len, const char *buffer) {
  off_t rv;

  rv = lseek(fd_, pos, SEEK_SET);

  assert(rv >= 0);

  for (;;) {
    ssize_t written = write(fd_, buffer, len);

    len -= written;

    if (len == 0) {
      break;
    }

    DEBUG_ASSERT_MSG(written > 0, "error writing");

    buffer += written;
  }
}

void RandomAccessFile::Read(off_t pos, size_t len, char *buffer) {
  off_t rv;

  rv = lseek(fd_, pos, SEEK_SET);

  assert(rv >= 0);

  for (;;) {
    ssize_t amount_read = read(fd_, buffer, len);

    len -= amount_read;

    if (len == 0) {
      break;
    }

    if (amount_read <= 0) {
      assert(amount_read == 0);
      /* end-of-file, fill with zeros */
      memset(buffer, 0, len);
      break;
    }

    buffer += amount_read;
  }
}

off_t RandomAccessFile::FindSize() const {
  struct stat s;
  fstat(fd_, &s);
  return s.st_size;
}
  
DiskBlockDevice::~DiskBlockDevice() {
  file_.Close();
  if (mode_ == BlockDevice::TEMP) {
    unlink(path_.c_str());
  }
}

void DiskBlockDevice::Init(
    const char *fname, mode_t mode_in, offset_t block_size) {
  mode_ = mode_in;
  
  file_.Init(fname, mode_);
  
  n_block_bytes_ = block_size;
  n_blocks_ = (file_.FindSize() + n_block_bytes_ - 1) / n_block_bytes_;
  
  path_.Copy(fname);
}

void DiskBlockDevice::Read(blockid_t blockid,
    offset_t begin, offset_t end, char *data) {
  if (unlikely(blockid >= n_blocks_)) {
    n_blocks_ = blockid + 1;
  }
  DEBUG_BOUNDS(end, n_block_bytes_ + 1);
  DEBUG_BOUNDS(begin, end + 1);
  file_.Read(off_t(blockid) * n_block_bytes_ + begin, end - begin, data);
}

void DiskBlockDevice::Write(blockid_t blockid,
    offset_t begin, offset_t end, const char *data) {
  if (unlikely(blockid >= n_blocks_)) {
    n_blocks_ = blockid + 1;
  }
  DEBUG_BOUNDS(end, n_block_bytes_ + 1);
  DEBUG_BOUNDS(begin, end + 1);
  file_.Write(off_t(blockid) * n_block_bytes_ + begin, end - begin, data);
}

DiskBlockDevice::blockid_t DiskBlockDevice::AllocBlock() {
  blockid_t blockid = n_blocks_;
  n_blocks_ = blockid + 1;
  return blockid;
}


void MemBlockDevice::Init(offset_t block_size) {
  n_blocks_ = 0;
  n_block_bytes_ = block_size;
  blocks_.Init();
}

void MemBlockDevice::Read(blockid_t blockid,
    offset_t begin, offset_t end, char *data) {
  CheckSize_(blockid);
  char *mydata = blocks_[blockid].data;
  
  if (likely(mydata != NULL)) {
    mem::Copy(data, end - begin, mydata + begin);
  } else {
    // initialize with random garbage
  }
}

void MemBlockDevice::Write(blockid_t blockid,
    offset_t begin, offset_t end, const char *data) {
  CheckSize_(blockid);
  char *mydata = blocks_[blockid].data;

  if (unlikely(data == NULL)) {
    blocks_[blockid] = data = mem::Alloc<char>(n_block_bytes_);
  }

  mem::Copy(mydata + begin, end - begin, data);
}

blockid_t MemBlockDevice::AllocBlock() {
  CheckSize_(n_blocks_);
  return n_blocks_ - 1;
}

MemBlockDevice::~MemBlockDevice() {
}
