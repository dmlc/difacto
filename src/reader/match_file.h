/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_READER_MATCH_FILE_H_
#define DIFACTO_READER_MATCH_FILE_H_
#include <regex.h>
#include <vector>
#include <string>
#include "io/filesys.h"
#include "dmlc/logging.h"
namespace difacto {

/**
 *
 * \brief match file by regex patterns
 * such as s3://my_path/part-.*
 *
 * @param pattern the regex pattern
 * @param matched matched filenames
 */
inline void MatchFile(const std::string& pattern,
                      std::vector<std::string>* matched) {
  // get the path
  size_t pos = pattern.find_last_of("/\\");
  std::string path = "./";
  if (pos != std::string::npos) path = pattern.substr(0, pos);

  // find all files
  dmlc::io::URI path_uri(path.c_str());
  dmlc::io::FileSystem *fs =
      dmlc::io::FileSystem::GetInstance(path_uri.protocol);
  std::vector<io::FileInfo> info;
  fs->ListDirectory(path_uri, &info);

  // store all matached files
  regex_t pat;
  std::string file =
      pos == std::string::npos ? pattern : pattern.substr(pos+1);
  file = ".*" + file;
  int status = regcomp(&pat, file.c_str(), REG_EXTENDED|REG_NEWLINE);
  if (status != 0) {
    char error_message[1000];
    regerror(status, &pat, error_message, 1000);
    LOG(FATAL) << "error regex '" << pattern << "' : " << error_message;
  }

  regmatch_t m[1];
  CHECK_NOTNULL(matched);
  for (size_t i = 0; i < info.size(); ++i) {
    std::string file = info[i].path.str();
    if (regexec(&pat, file.c_str(), 1, m, 0)) continue;
    matched->push_back(file);
  }
}

}  // namespace difacto
#endif  // DIFACTO_READER_MATCH_FILE_H_
