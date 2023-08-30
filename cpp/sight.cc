// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "googlex/cortex/sight/sight.h"

#include "analysis/dremel/core/capacitor/public/importer-options.proto.h"
#include "analysis/dremel/core/capacitor/public/message-importer.h"
#include "analysis/dremel/core/capacitor/public/schema.h"
#include "file/base/file.h"
#include "file/base/helpers.h"
#include "file/base/path.h"
#include "net/grpc/public/include/grpcpp/credentials_google.h"
#include "net/grpc/public/include/grpcpp/support/status.h"
#include "net/grpc/public/include/grpcpp/support/time_google.h"
#include "third_party/absl/memory/memory.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/str_replace.h"
#include "third_party/absl/strings/substitute.h"
#include "third_party/grpc/include/grpcpp/channel.h"
#include "third_party/grpc/include/grpcpp/client_context.h"
#include "third_party/grpc/include/grpcpp/create_channel.h"
#include "third_party/grpc/include/grpcpp/grpcpp.h"
#include "third_party/grpc/include/grpcpp/security/credentials.h"
#include "util/gtl/map_util.h"
#include "util/task/status_macros.h"

namespace sight {

using ::absl::StatusOr;
using ::absl::Substitute;
using x::proto::BlockEnd;
using x::proto::BlockStart;
using x::proto::Object;
using x::proto::SingleLogObject;

using dremel::capacitor::MessageImporter;
using dremel::capacitor::MessageImporterOptions;

Sight::Params::Params()
    : local_(false),
      text_output_(false),
      capacitor_output_(false) {}

Sight::Params Sight::Params::WithLocal() {
  Params params(*this);
  params.local_ = true;
  return params;
}

Sight::Params Sight::Params::WithLogDirPath(absl::string_view log_dir_path) {
  Params params(*this);
  params.log_dir_path_ = static_cast<std::string>(log_dir_path);
  return params;
}

Sight::Params Sight::Params::WithLabel(absl::string_view label) {
  Params params(*this);
  params.label_ = static_cast<std::string>(label);
  return params;
}

Sight::Params Sight::Params::WithTextOutput() {
  Params params(*this);
  params.text_output_ = true;
  return params;
}

Sight::Params Sight::Params::WithCapacitorOutput() {
  Params params(*this);
  params.capacitor_output_ = true;
  return params;
}

Sight::Params Sight::Params::WithLogOwner(absl::string_view log_owner) {
  Params params(*this);
  params.log_owner_ = static_cast<std::string>(log_owner);
  return params;
}

std::unique_ptr<Sight> Sight::Create() {
  return absl::WrapUnique<Sight>(new Sight());
}

StatusOr<std::unique_ptr<Sight>> Sight::Create(const Params& params) {
  auto sight = absl::WrapUnique<Sight>(new Sight());

  sight->params_ = params;

  // The path prefix common to all the file(s) that hold the log.
  std::string path_prefix;
  if (params.local()) {
    std::string file_name_prefix;
    if (params.label().empty()) {
      file_name_prefix = "log";
    } else {
      file_name_prefix = params.label();
    }
    path_prefix = file::JoinPath(
        static_cast<std::string>(params.log_dir_path()), file_name_prefix);
    sight->id_ = 0;
  } else {
    sight->sight_service_ =
        x::service::v1::grpc::SightService::NewStub(grpc::CreateChannel(
            //        "bronevet-sight.sandbox.googleapis.com",
            "/bns/wl/borg/wl/bns/bronevet/sight-service.server/0",
            grpc::Loas2Credentials(grpc::Loas2CredentialsOptions())));

    grpc::ClientContext context;
    absl::Time abs_deadline = absl::Now() + absl::Seconds(10);
    context.set_deadline(abs_deadline);

    x::service::v1::CreateRequest request;
    x::service::v1::CreateResponse response;
    if (params.capacitor_output()) {
      request.set_format(x::service::v1::LF_CAPACITOR);
    } else {
      request.set_format(x::service::v1::LF_COLUMNIO);
    }
    request.set_log_dir_path(params.log_dir_path());
    request.set_log_owner(params.log_owner());
    request.set_label(params.label());

    RETURN_IF_ERROR(
        sight->sight_service_->Create(&context, request, &response));

    sight->id_ = response.id();
    path_prefix = response.path_prefix();
  }

  if (params.text_output()) {
    RETURN_IF_ERROR(file::Open(absl::StrCat(path_prefix, ".txt"), "w",
                               &sight->log_file_, file::Defaults()));
  }

  if (params.capacitor_output()) {
    sight->capacitor_log_file_path_ =
        file::JoinPath(absl::StrCat(path_prefix, ".capacitor"));

    MessageImporterOptions options;
    RETURN_IF_ERROR(BuildSchemaProto(Object::descriptor(),
                                     options.schema_options(),
                                     options.mutable_native_schema()));
    sight->log_capacitor_writer_ =
        MessageImporter::Create(options).ValueOrDie();
    RETURN_IF_ERROR(sight->log_capacitor_writer_->StartSession());
  }

  sight->pause_logging_depth_ = 0;

  return std::move(sight);
}

void Sight::PauseLogging() { ++pause_logging_depth_; }

void Sight::ResumeLogging() { --pause_logging_depth_; }

bool Sight::IsLoggingEnabled() { return pause_logging_depth_ == 0; }

std::string Sight::Text(absl::string_view text) {
  if (pause_logging_depth_ > 0) {
    return "";
  }

  if (isBinaryLogged()) {
    Object object;
    object.set_sub_type(x::proto::Object::ST_TEXT);
    x::proto::Text* text_object = object.mutable_text();
    text_object->set_text(absl::StrReplaceAll(text, {{"\n", "\\n"}}));
    LogObject(&object, /*advance_location=*/true);
  }

  EmitTextToFile(text);

  return static_cast<std::string>(text);
}

std::string Sight::TextLine(absl::string_view text) {
  return TextLineWithMeta(text, /*file=*/"", /*line=*/0, /*func=*/"");
}

std::string Sight::TextLineWithMeta(absl::string_view text,
                                    absl::string_view file, int line,
                                    absl::string_view func) {
  if (pause_logging_depth_ > 0) {
    return "";
  }

  if (isBinaryLogged()) {
    Object object;
    if (!file.empty()) {
      object.set_file(file);
      object.set_line(line);
      object.set_func(func);
    }
    object.set_sub_type(x::proto::Object::ST_TEXT);
    x::proto::Text* text_object = object.mutable_text();
    text_object->set_text(
        absl::StrReplaceAll(absl::StrCat(text, "\n"), {{"\n", "\\n"}}));
    LogObject(&object, /*advance_location=*/true);
  }

  std::string full_text_line = absl::StrCat(
      file, ":", line, "/", func, " ", line_prefix_, text, line_suffix_, "\n");
  EmitTextToFile(full_text_line);

  return full_text_line;
}

SightCoreInterface::Location Sight::EnterBlock(absl::string_view label) {
  return EnterBlock(label, /*file=*/"", /*line=*/0, /*func=*/"");
}

SightCoreInterface::Location Sight::EnterBlock(absl::string_view label,
                                               absl::string_view file, int line,
                                               absl::string_view func) {
  Object object;
  return EnterBlock(label, file, line, func, &object);
}

SightCoreInterface::Location Sight::EnterBlock(absl::string_view label,
                                               absl::string_view file, int line,
                                               absl::string_view func,
                                               Object* object) {
  if (pause_logging_depth_ > 0) {
    return location_;
  }
  if (func != "") {
    SetAttribute("func", func);
  }

  active_block_labels_.push_back(static_cast<std::string>(label));
  EmitTextToFile(absl::StrCat(line_prefix_, label, "<<<<", line_suffix_, "\n"));
  LOG(INFO) << line_prefix_ << "<<<<";
  absl::StrAppend(&line_prefix_, label, ": ");

  const auto obj_location = location_;
  if (isBinaryLogged()) {
    if (!file.empty()) {
      object->set_file(file);
      object->set_line(line);
      object->set_func(func);
    }
    object->set_sub_type(x::proto::Object::ST_BLOCK_START);
    BlockStart* block_start = object->mutable_block_start();
    block_start->set_label(label);
    LogObject(object, /*advance_location=*/false);
    open_block_start_locations_.push_back(object->location());
  }

  num_direct_contents_.push_back(0);
  num_transitive_contents_.push_back(0);
  location_.push_back(0);
  return obj_location;
}

void Sight::ExitBlock(absl::string_view label) {
  Object object;
  ExitBlock(label, &object);
}

void Sight::ExitBlock(absl::string_view label, Object* object) {
  if (pause_logging_depth_ > 0) {
    return;
  }

  if (active_block_labels_.empty() || location_.size() == 1) {
    LOG(ERROR) << TextLine(
        Substitute("Exiting inactive Sight block \"$0\"", label));
    return;
  }
  active_block_labels_.pop_back();
  line_prefix_.clear();
  for (const auto& block_label : active_block_labels_) {
    absl::StrAppend(&line_prefix_, block_label, ": ");
  }

  location_.pop_back();
  ++location_.back();

  if (isBinaryLogged()) {
    if (open_block_start_locations_.empty()) {
      LOG(ERROR) << TextLine(
          Substitute("Exiting inactive Sight block \"$0\"", label));
      return;
    }

    object->set_sub_type(x::proto::Object::ST_BLOCK_END);
    BlockEnd* block_end = object->mutable_block_end();
    block_end->set_label(label);
    block_end->set_num_direct_contents(num_direct_contents_.back());
    block_end->set_num_transitive_contents(num_transitive_contents_.back());
    block_end->set_location_of_block_start(open_block_start_locations_.back());
    open_block_start_locations_.pop_back();
    LogObject(object, /*advance_location=*/true);
  }

  EmitTextToFile(absl::StrCat(line_prefix_, label, ">>>>", line_suffix_, "\n"));
  LOG(INFO) << line_prefix_ << ">>>>";

  num_direct_contents_.pop_back();
  num_transitive_contents_.pop_back();

  if (GetAttribute("func") != "") {
    UnsetAttribute("func");
  }
}

void Sight::SetAttribute(absl::string_view key, absl::string_view value) {
  attributes_[static_cast<std::string>(key)].push_back(
      static_cast<std::string>(value));
  line_suffix_ = "| ";
  for (auto attr = attributes_.begin(); attr != attributes_.end(); ++attr) {
    if (attr != attributes_.begin()) {
      absl::StrAppend(&line_suffix_, ",");
    }
    absl::StrAppend(&line_suffix_, attr->first, "=", attr->second.back());
  }
}

void Sight::UnsetAttribute(absl::string_view key) {
  auto attribute = attributes_.find(static_cast<std::string>(key));
  if (attribute == attributes_.end() || attribute->second.empty()) {
    LOG(ERROR) << TextLine(
        Substitute("Failed to unset attribute $0, which is not set.", key));
    return;
  }

  attribute->second.pop_back();
  if (attribute->second.empty()) {
    attributes_.erase(attribute);
  }
}

absl::string_view Sight::GetAttribute(absl::string_view key) {
  static std::string* default_attribute_value = new std::string();
  const auto* values =
      gtl::FindOrNull(attributes_, static_cast<std::string>(key));
  if (values == nullptr || values->empty()) {
    return *default_attribute_value;
  }
  return values->back();
}

SightCoreInterface::Location Sight::LogObject(x::proto::Object* object) {
  return LogObject(object, /*advance_location=*/true);
}

Sight::Sight() : log_file_(nullptr), pause_logging_depth_(0) {
  // Initialize a top-level location index at the top-level log scope.
  location_.push_back(0);
  index_ = 0;
}

Sight::~Sight() {
  if (log_file_ != nullptr) {
    if (!log_file_->Close(file::Defaults()).ok()) {
      LOG(ERROR) << "Error closing Sight log file.";
    }
  }

  if (isBinaryLogged()) {
    if (log_capacitor_writer_ != nullptr) {
      File* file_ptr =
          file::OpenOrDie(capacitor_log_file_path_, "wb", file::Defaults());
      auto finalize_s = log_capacitor_writer_->FinalizeToFile(file_ptr);
      if (!finalize_s.ok()) {
        LOG(ERROR) << "Finalizing Capacitor log file: " << finalize_s.status();
      }
    }

    if (!params_.local()) {
      grpc::ClientContext context;
      absl::Time abs_deadline = absl::Now() + absl::Seconds(10);
      context.set_deadline(abs_deadline);

      x::service::v1::FinalizeRequest request;
      x::service::v1::FinalizeResponse response;
      request.set_id(id_);

      const auto finalize_s =
          sight_service_->Finalize(&context, request, &response);
      if (!finalize_s.ok()) {
        LOG(ERROR) << "ERROR finalizing log in Sight service: " << finalize_s;
      }
      std::cout << "Log " << params_.label()
                << ": https://script.google.com/a/google.com/macros/s/"
                   "AKfycbwIPbs-xJomxbAKF_6Q9lci9rMrXqG37O862MaTfGSn/dev?logId="
                << id_ << std::endl;
    }
  }
}

SightCoreInterface::Location Sight::LogObject(Object* object,
                                              bool advance_location) {
  if (!num_direct_contents_.empty()) {
    ++num_direct_contents_.back();
  }
  for (auto& transitive_count : num_transitive_contents_) {
    ++transitive_count;
  }
  const auto obj_location = location_;
  if (isBinaryLogged()) {
    object->set_location(absl::StrJoin(location_, ":"));
    object->set_index(index_++);
    for (const auto& a : attributes_) {
      if (a.second.empty()) {
        LOG(ERROR) << "No attributes recorded for key " << a.first;
        continue;
      }
      x::proto::Attribute* attribute = object->add_attribute();
      attribute->set_key(a.first);
      *attribute->mutable_value() = a.second.back();
    }
    for (const auto& block_loc : open_block_start_locations_) {
      *object->add_ancestor_start_location() = block_loc;
    }
    *object->add_ancestor_start_location() = object->location();
    SingleLogObject single_object;
    *single_object.mutable_obj() = *object;
    if (log_capacitor_writer_ != nullptr) {
      absl::Status capacitor_write_s =
          log_capacitor_writer_->ImportMessageBytes(
              single_object.SerializeAsString());
      if (!capacitor_write_s.ok()) {
        LOG(ERROR) << "Writing Sight log object: " << capacitor_write_s;
      }
    }
  }
  if (advance_location) {
    ++location_.back();
  }
  return obj_location;
}

void Sight::EmitTextToFile(absl::string_view text) {
  if (log_file_ != nullptr) {
    const auto write_s = file::WriteString(log_file_, text, file::Defaults());
    if (!write_s.ok()) {
      LOG(ERROR) << "Writing \"" << text << "\" to log file: " << write_s;
    }
  }
}

bool Sight::isBinaryLogged() const {
  return (log_capacitor_writer_ != nullptr);
}

}  // namespace sight
