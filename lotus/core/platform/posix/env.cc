/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <dlfcn.h>

#include <thread>
#include <vector>

#include "core/platform/env.h"
#include "core/platform/types.h"
#include "core/common/common.h"

namespace Lotus {

namespace {

class StdThread : public Thread {
 public:
  StdThread(std::function<void()> fn)
      : thread_(fn) {}
  ~StdThread() override { thread_.join(); }

 private:
  std::thread thread_;
};

class PosixEnv : public Env {
 public:
  static PosixEnv& Instance() {
    static PosixEnv default_env;
    return default_env;
  }

  int GetNumCpuCores() const override {
    // TODO if you need the number of physical cores you'll need to parse
    // /proc/cpuinfo and grep for "cpu cores".
    //However, that information is not always available(output of 'grep -i core /proc/cpuinfo' is empty)
    return std::thread::hardware_concurrency();
  }

  EnvThread* CreateThread(std::function<void()> fn) const override {
    return new StdThread(fn);
  }

  Task CreateTask(std::function<void()> f) const override {
    return Task{std::move(f)};
  }
  void ExecuteTask(const Task& t) const override {
    t.f();
  }

  void SleepForMicroseconds(int64 micros) const override {
    while (micros > 0) {
      timespec sleep_time;
      sleep_time.tv_sec = 0;
      sleep_time.tv_nsec = 0;

      if (micros >= 1e6) {
        sleep_time.tv_sec =
            std::min<int64>(micros / 1e6, std::numeric_limits<time_t>::max());
        micros -= static_cast<int64>(sleep_time.tv_sec) * 1e6;
      }
      if (micros < 1e6) {
        sleep_time.tv_nsec = 1000 * micros;
        micros = 0;
      }
      while (nanosleep(&sleep_time, &sleep_time) != 0 && errno == EINTR) {
        // Ignore signals and wait for the full interval to elapse.
      }
    }
  }

  Thread* StartThread(const ThreadOptions& /*thread_options*/, const std::string& /*name*/,
                      std::function<void()> fn) const override {
    return new StdThread(fn);
  }

  PIDType GetSelfPid() const override {
    return getpid();
  }

  Common::Status FileOpenRd(const std::string& path, /*out*/ gsl::not_null<int*> p_fd) const override {
    *p_fd = open(path.c_str(), O_RDONLY);
    if (0 > *p_fd) {
      return Common::Status(Common::SYSTEM, errno);
    }
    return Status::OK();
  }

  Common::Status FileOpenWr(const std::string& path, /*out*/ gsl::not_null<int*> p_fd) const override {
    *p_fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (0 > *p_fd) {
      return Common::Status(Common::SYSTEM, errno);
    }
    return Status::OK();
  }

  Common::Status FileClose(int fd) const override {
    int ret = close(fd);
    if (0 != ret) {
      return Common::Status(Common::SYSTEM, errno);
    }
    return Status::OK();
  }

  Common::Status FileExists(const char* /*fname*/) const override {
    return Common::Status(Common::LOTUS, Common::NOT_IMPLEMENTED, "NOT_IMPLEMENTED");
  }
  Common::Status ReadFileAsString(const char* fname, std::string* out) const override {
    if (!out) {
      return Common::Status(Common::LOTUS, Common::INVALID_ARGUMENT, "'out' cannot be NULL");
    }
    char errbuf[512];
    int fd = open(fname, O_RDONLY);
    if (fd < 0) {
      snprintf(errbuf, sizeof(errbuf), "%s:%d open file %s fail, errcode = %d", __FILE__, __LINE__, fname, errno);
      return Common::Status(Common::LOTUS, Common::FAIL, errbuf);
    }
    struct stat stbuf;
    if ((fstat(fd, &stbuf) != 0) || (!S_ISREG(stbuf.st_mode))) {
      close(fd);
      snprintf(errbuf, sizeof(errbuf), "%s:%d read file %s fail", __FILE__, __LINE__, fname);
      return Common::Status(Common::LOTUS, Common::FAIL, errbuf);
    }
    if (stbuf.st_size == 0) {
      out->clear();
    } else {
      out->resize(stbuf.st_size, '\0');
      ssize_t bytes_readed = read(fd, (void*)out->data(), stbuf.st_size);
      if (bytes_readed <= 0 || bytes_readed != stbuf.st_size) {
        close(fd);
        snprintf(errbuf,
                 sizeof(errbuf),
                 "%s:%d open file %s fail, errcode = %d",
                 __FILE__,
                 __LINE__,
                 fname,
                 errno);
        return Common::Status(Common::LOTUS, Common::FAIL, errbuf);
      }
      close(fd);
    }
    return Common::Status::OK();
  }

  virtual Common::Status LoadLibrary(const std::string& library_filename, void** handle) const override {
    *handle = dlopen(library_filename.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!*handle) {
      return Common::Status(Common::LOTUS, Common::FAIL,
                            "Failed to load library " + library_filename + " with error: " + dlerror());
    }
    return Common::Status::OK();
  }

  virtual Common::Status UnloadLibrary(void* handle) const override {
    if (!handle) {
      return Common::Status(Common::LOTUS, Common::FAIL, "Got null library handle");
    }
    int retval = dlclose(handle);
    if (retval != 0) {
      return Common::Status(Common::LOTUS, Common::FAIL, "Failed to unload library");
    }
    return Common::Status::OK();
  }

  virtual Common::Status GetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) const override {
    char* error_str = dlerror();  // clear any old error str
    *symbol = dlsym(handle, symbol_name.c_str());
    error_str = dlerror();
    if (!error_str) {
      return Common::Status(Common::LOTUS, Common::FAIL,
                            "Failed to get symbol " + symbol_name + " with error: " + error_str);
    }
    if (!*symbol) {
      return Common::Status(Common::LOTUS, Common::FAIL, "Got null symbol from library");
    }
    return Common::Status::OK();
  }

  virtual std::string FormatLibraryFileName(const std::string& name, const std::string& version) const override {
    std::string filename;
    if (version.empty()) {
      filename = "lib" + name + ".so";
    } else {
      filename = "lib" + name + ".so" + "." + version;
    }
    return filename;
  }

 private:
  PosixEnv() = default;
};

}  // namespace

#if defined(PLATFORM_POSIX) || defined(__ANDROID__)
// REGISTER_FILE_SYSTEM("", PosixFileSystem);
// REGISTER_FILE_SYSTEM("file", LocalPosixFileSystem);
const Env& Env::Default() {
  return PosixEnv::Instance();
}
#endif

}  // namespace Lotus
