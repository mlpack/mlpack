
void RandomAccessFile::Init(const char *fname, BlockDevice::mode_t mode) {
  int octal_mode;
  
  switch(mode) {
   BlockDevice::READ:
    octal_mode = O_RDONLY;
    break;
   BlockDevice::MODIFY:
    octal_mode = O_RDWR;
    break;
   BlockDevice::CREATE:
   BlockDevice::TEMP:
    octal_mode = O_RDWR|O_CREAT|O_TRUNC;
    break;
   default: abort();
  }
  
  fd_ = open(fname, octal_mode, 0666);
  if (fd_ <= 0) {
    FATAL("Could not open file '%s'.", fname);
  }
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

    DEBUG_ASSERT_MSG(written > 0, "error writing %lu bytes", len);

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

void DiskBlockDevice::Init(
    const char *fname, mode_t mode, offset_t block_size) {
  file_.Init(fname, mode);
  
  n_block_bytes_ = block_size;
  n_blocks_ = (file_.FindSize() + n_block_bytes_ - 1) / n_block_bytes_;
}

void DiskBlockDevice::Read(blockid_t blockid,
    offset_t begin, offset_t end, char *data) {
  DEBUG_ASSERT(blockid <= n_blocks_);
  DEBUG_BOUNDS(end, n_block_bytes_ + 1);
  DEBUG_BOUNDS(begin, end + 1);
  file_.Read(off_t(blockid) * n_block_bytes_ + begin, end - begin, data);
}

void DiskBlockDevice::Write(blockid_t blockid,
    offset_t begin, offset_t end, const char *data) {
  DEBUG_ASSERT(blockid <= n_blocks_);
  DEBUG_BOUNDS(end, n_block_bytes_ + 1);
  DEBUG_BOUNDS(begin, end + 1);
  file_.Write(off_t(blockid) * n_block_bytes_ + begin, end - begin, data);
}

blockid_t DiskBlockDevice::AllocBlock() {
  blockid_t blockid = n_blocks_;
  n_blocks_ = blockid + 1;
  return blockid;
}
