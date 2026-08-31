#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

template <typename T>
class PosixSharedMemory {
public:
    PosixSharedMemory(const std::string& name, bool create)
        : name_(name), fd_(-1), ptr_(nullptr) {
        int flags = create ? (O_CREAT | O_RDWR) : O_RDWR;
        fd_ = shm_open(name_.c_str(), flags, 0666);
        if (fd_ < 0) {
            throw std::runtime_error("shm_open failed: " + std::string(std::strerror(errno)));
        }
        if (create) {
            if (ftruncate(fd_, sizeof(T)) < 0) {
                close(fd_);
                throw std::runtime_error("ftruncate failed: " + std::string(std::strerror(errno)));
            }
        }
        void* mapped = mmap(nullptr, sizeof(T), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapped == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("mmap failed: " + std::string(std::strerror(errno)));
        }
        ptr_ = static_cast<T*>(mapped);
    }

    ~PosixSharedMemory() {
        if (ptr_ != nullptr) {
            munmap(ptr_, sizeof(T));
            ptr_ = nullptr;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    PosixSharedMemory(const PosixSharedMemory&) = delete;
    PosixSharedMemory& operator=(const PosixSharedMemory&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }

private:
    std::string name_;
    int fd_;
    T* ptr_;
};
