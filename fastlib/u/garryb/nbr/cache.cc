

#ifdef GRAVEYARD

----------------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------

class RandomAccessFile {
 private:
  int fd_;

 public:
  void Init(const char *fname);

  void Read(char *buffer, off_t pos, size_t len);
  void Write(const char *buffer, off_t pos, size_t len);
}

void RandomAccessFile::Init(const char *fname) {
  fd_ = open(fname, O_RDWR|OCREAT|O_TRUNC, 0666);
  if (fd_ <= 0) FATAL("Could not open file '%s'.", fname):
}

void RandomAccessFile::Write(const char *buffer, off_t pos, size_t len) {
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

void RandomAccessFile::Read(char *buffer, off_t pos, size_t len) {
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


void DiskBlockDevice::Read(blockid_t blockid, char *data) {
  copy code from blockio to handle incomplete buffers
}

void DiskBlockDevice::Write(blockid_t blockid, const char *data) {
  copy code from blockio to handle incomplete buffers
}

#endif
