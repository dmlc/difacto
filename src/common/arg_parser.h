#ifndef DIFACTO_COMMON_ARG_PARSER_H_
#define DIFACTO_COMMON_ARG_PARSER_H_
#include <string>
#include <sstream>
#include "dmlc/io.h"
#include "dmlc/config.h"
#include "difacto/base.h"
namespace difacto {
class ArgParser {
 public:
  ArgParser() { }
  ~ArgParser() { }

  /**
   * \brief read all args in a file
   */
  void AddArgFile(const char* const filename) {
    dmlc::Stream *fs = dmlc::Stream::Create(filename, "r");
    CHECK(fs != nullptr) << "failed to open " << filename;
    char buf[1000];
    while (true) {
      size_t r = fs->Read(buf, 1000);
      data_.append(buf, r);
      if (!r) break;
    }
  }
  /**
   * \brief add an arg
   */
  void AddArg(const char* argv) {
    data_.append(argv); data_.append(" ");
  }

  /**
   * \brief return parsed kwargs
   */
  KWArgs GetKWArgs() {
    std::stringstream ss(data_);
    dmlc::Config conf(ss);
    KWArgs kwargs;
    for (auto it : conf) {
      kwargs.push_back(it);
    }
    return kwargs;
  }

 private:
  std::string data_;
};
}  // namespace difacto
#endif /* DIFACTO_COMMON_ARG_PARSER_H_ */
