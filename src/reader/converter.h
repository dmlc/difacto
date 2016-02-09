/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef DIFACTO_READER_CONVERTER_H_
#define DIFACTO_READER_CONVERTER_H_
#include <string>
#include "dmlc/parameter.h"
#include "reader/reader.h"
#include "dmlc/io.h"
namespace difacto {

struct ConverterParam : public dmlc::Parameter<ConverterParam> {
  /** \brief The input data, either a filename or a directory. */
  std::string data_in;
  /** \brief the input data format: libsvm, rec, criteo, criteo_test, adfea, ... */
  std::string data_format;
  /** \brief The prefix of output */
  std::string data_out;
  /** \brief the output data format: libsvm or rec */
  std::string data_out_format;
  /**
   * \brief split the output into multiple parts
   * each part <= part_size MB
   * the default value -1 means no splitting
   */
  int part_size;
  DMLC_DECLARE_PARAMETER(ConverterParam) {
    DMLC_DECLARE_FIELD(data_in);
    DMLC_DECLARE_FIELD(data_format);
    DMLC_DECLARE_FIELD(data_out);
    DMLC_DECLARE_FIELD(data_out_format);
    DMLC_DECLARE_FIELD(part_size).set_default(-1);
  };
};
/**
 * \brief data converter
 */
class Converter {
 public:
  KWArgs Init(const KWArgs& kwargs) {
    auto remain = param_.InitAllowUnknown(kwargs);
    return remain;
  }

  void Run() {
    using namespace dmlc;
    using namespace dmlc::data;
    Reader in(param_.data_in, param_.data_format, 0, 1, 8);

    LOG(INFO) << "reading data from " << param_.data_in
              << " in " << param_.data_format << " format";

    int ipart = 0;
    size_t limit = static_cast<size_t>(-1);
    size_t part_size = static_cast<size_t>(param_.part_size);
    size_t nwrite = limit;
    Stream *out = nullptr;
    RecordIOWriter* rec_writer = nullptr;
    ostream* libsvm_writer = nullptr;

    while (in.Next()) {
      auto out_format = param_.data_out_format;
      if (nwrite == limit || nwrite / 1000000 >= part_size) {
        if (nwrite != limit) {
          LOG(INFO) << "done. written " << nwrite << " bytes";
        }
        auto outfile = param_.data_out;
        if (part_size != limit) {
          outfile += "-part_" + std::to_string(ipart++);
        }
        delete libsvm_writer; libsvm_writer = nullptr;
        delete rec_writer; rec_writer = nullptr;
        delete out; out = CHECK_NOTNULL(Stream::Create(outfile.c_str(), "wb"));
        nwrite = 0;

        LOG(INFO) << "wrting data to " << outfile
                  << " in " << out_format << " format";
        if (out_format == "libsvm") {
          libsvm_writer = new ostream(out);
        } else if (out_format == "rec") {
          rec_writer = new RecordIOWriter(out);
        } else {
          LOG(FATAL) << "unknow output format: " << out_format;
        }
      }

      if (out_format == "libsvm") {
        auto blk = in.Value();
        for (size_t i = 0; i < blk.size; ++i) {
          *libsvm_writer << blk.label[i] << " ";
          for (size_t j = blk.offset[i]; j < blk.offset[i+1]; ++j) {
            *libsvm_writer << blk.index[j];
            if (blk.value) *libsvm_writer << ":" << blk.value[j];
            *libsvm_writer << " ";
          }
          *libsvm_writer << "\n";
        }
        nwrite = libsvm_writer->bytes_written();
      } else if (out_format == "crb") {
        std::string str;
        CompressedRowBlock cblk;
        cblk.Compress(in.Value(), &str);
        rec_writer->WriteRecord(str);
        nwrite += str.size();
      }
    }
    delete libsvm_writer;
    delete rec_writer;
    delete out;
    LOG(INFO) << "done. written " << nwrite << "bytes";
  }

 private:
  ConverterParam param_;
};

}  // namespace difacto


#endif  // DIFACTO_READER_CONVERTER_H_
