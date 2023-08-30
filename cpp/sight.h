/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Sight logging library.
#ifndef GOOGLEX_CORTEX_SIGHT_SIGHT_H_
#define GOOGLEX_CORTEX_SIGHT_SIGHT_H_

#include <vector>

#include "analysis/dremel/core/capacitor/public/message-importer.h"
#include "base/logging.h"
#include "base/timer.h"
#include "file/base/file.h"
#include "file/base/path.h"
#include "googlex/cortex/sight/proto/sight.proto.h"
#include "googlex/cortex/sight/service/v1/sight_service.grpc.pb.h"
#include "googlex/cortex/sight/sight_interface.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"

namespace sight {

class Sight : public SightInterface {
 public:
  ~Sight() override;

  // The configuration parameters of a Sight instance.
  class Params {
   public:
    Params();
    Params(const Params& params) = default;

    Params WithLocal();
    Params WithLogDirPath(absl::string_view log_dir_path);
    Params WithLabel(absl::string_view label);
    Params WithTextOutput();
    Params WithCapacitorOutput();
    Params WithLogOwner(absl::string_view log_owner);

    bool local() const { return local_; }
    const std::string& log_dir_path() const { return log_dir_path_; }
    const std::string& label() const { return label_; }
    bool text_output() const { return text_output_; }
    bool capacitor_output() const { return capacitor_output_; }
    const std::string& log_owner() const { return log_owner_; }

   private:
    // Indicates that the log will be stored without the use of the Sight
    // service or the benefits of its UI.
    bool local_;

    // The directory to which the log data will be writtn.
    std::string log_dir_path_;

    // Unique label used to differentiate the output log files from those
    // from other application runs.
    std::string label_;

    // Indicates whether a text-formatted file needs to be written to
    // log_dir_path_.
    bool text_output_;

    // Indicates whether a Capacitor-formatted file needs to be written to
    // log_dir_path.
    bool capacitor_output_;

    // The user/group that owns the log tables.
    std::string log_owner_;
  };

  // Creates a Sight object that writes all log data to the standard log.
  static std::unique_ptr<Sight> Create();

  // Creates a Sight object configured via params.
  static absl::StatusOr<std::unique_ptr<Sight>> Create(const Params& params);

  void PauseLogging() override;

  void ResumeLogging() override;

  bool IsLoggingEnabled() override;

  std::string Text(absl::string_view text) override;

  std::string TextLine(absl::string_view text) override;

  std::string TextLineWithMeta(absl::string_view text, absl::string_view file,
                               int line, absl::string_view func) override;

  Location EnterBlock(absl::string_view label) override;

  Location EnterBlock(absl::string_view label, absl::string_view file, int line,
                      absl::string_view func) override;

  Location EnterBlock(absl::string_view label, absl::string_view file, int line,
                      absl::string_view func,
                      x::proto::Object* object) override;

  void ExitBlock(absl::string_view label) override;

  void ExitBlock(absl::string_view label, x::proto::Object* object) override;

  void SetAttribute(absl::string_view key, absl::string_view value) override;

  void UnsetAttribute(absl::string_view key) override;

  absl::string_view GetAttribute(absl::string_view key) override;

  Location LogObject(x::proto::Object* object) override;

 private:
  Sight();

  // Emits object to the log.
  // advance_location: Indicates whether this method call should advance the
  //   current log location.
  // Returns the location of the emitted log object.
  Location LogObject(x::proto::Object* object, bool advance_location);

  // Emits text to the output text file, if one is being used.
  void EmitTextToFile(absl::string_view text);

  // Returns whether a binary proto representation is being logged.
  bool isBinaryLogged() const;

  // The configuration parameters of this object.
  Params params_;

  // The file to which the Capacitor representation of the log should be
  // written.
  std::string capacitor_log_file_path_;

  // The unique ID of this log, according to the Sight service.
  int64 id_;

  // If an output file path is specified, this writer is used to emit
  // the log in proto format.
  ::File* log_file_;

  // If an output file path is specified, this writer is used to emit
  // the log in Capacitor format.
  std::unique_ptr<dremel::capacitor::MessageImporter> log_capacitor_writer_;

  // The current location in the nesting hierarchy of the log.
  Location location_;

  // The index of the next log object.
  int index_;

  // The string to be prepended to every log line.
  std::string line_prefix_;

  // The string to be appended to every log line.
  std::string line_suffix_;

  // The stack of locations of all the blocks that have been entered but not
  // yet exited.
  std::vector<std::string> open_block_start_locations_;

  // The number of objects directly contained by currently active block
  // (organized as a stack with one level per block).
  std::vector<int> num_direct_contents_;

  // The number of objects transitively contained by currently active block
  // (organized as a stack with one level per block).
  std::vector<int> num_transitive_contents_;

  // The stack of the labels of blocks that are currently active (begun but not
  // yet ended).
  std::vector<std::string> active_block_labels_;

  // The key-value attribute mappings that currently in-force. Maps each key to
  // the stack of key-value mappings in the order they were set.
  std::map<std::string, std::vector<std::string>> attributes_;

  // Counts the numer of PauseLogging() calls with no matching calls to
  // ResumeLogging();
  int pause_logging_depth_;

  // Owned reference to a sub of the Sight service, if any.
  std::unique_ptr<x::service::v1::grpc::SightService::Stub> sight_service_;
};

}  // namespace sight

#endif  // GOOGLEX_CORTEX_SIGHT_SIGHT_H_
