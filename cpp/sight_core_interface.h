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

#ifndef GOOGLEX_CORTEX_SIGHT_SIGHT_CORE_INTERFACE_H_
#define GOOGLEX_CORTEX_SIGHT_SIGHT_CORE_INTERFACE_H_

#include "googlex/cortex/sight/proto/sight.proto.h"
#include "third_party/absl/strings/string_view.h"

namespace sight {

// Specifies a simplified version of the generic interface to structured
// logging that provides basic logging functionality without access to any
// of the widgets. It is this directly usable by the widgets themselves.
class SightCoreInterface {
 public:
  using Location = std::vector<int>;

  virtual ~SightCoreInterface() {}

  // Specifies that logging should not be performed until the next call to
  // ResumeLogging(). If PauseLogging() is called multiple times, logging is not
  // resumed until ResumeLogging() is called once for each call to
  // PauseLogging().
  virtual void PauseLogging() = 0;

  // Resumes logging after a call to PauseLogging().
  virtual void ResumeLogging() = 0;

  // Returns whether logging is currently enabled.
  virtual bool IsLoggingEnabled() = 0;

  // Emits to the log a line containing text.
  virtual std::string Text(absl::string_view text) = 0;

  // Emits to the log a line containing text, followed by a linebreak.
  virtual std::string TextLine(absl::string_view text) = 0;

  // Emits to the log a line containing text, followed by a linebreak,
  // documenting that it was emitted at the code location specified
  // by file, line and func.
  virtual std::string TextLineWithMeta(absl::string_view text,
                                       absl::string_view file, int line,
                                       absl::string_view func) = 0;

  // Enters a new block named label.
  // Returns the location of the emitted log object.
  virtual Location EnterBlock(absl::string_view label) = 0;

  // Enters a new block named label that starts at the code location specified
  // by file, line and func.
  // Returns the location of the emitted log object.
  virtual Location EnterBlock(absl::string_view label, absl::string_view file,
                              int line, absl::string_view func) = 0;

  // Enters a new block named label that starts at the code location specified
  // by file, line and func.
  // object is expected to describe an object that inherits from StartBlock,
  //     with all of its specialized fields set.
  // Returns the location of the emitted log object.
  virtual Location EnterBlock(absl::string_view label, absl::string_view file,
                              int line, absl::string_view func,
                              x::proto::Object* object) = 0;

  // Exits a previously-entered block named label.
  virtual void ExitBlock(absl::string_view label) = 0;

  // Exits a previously-entered block named label.
  // object is expected to describe an object that inherits from EndBlock,
  // with all of its specialized fields set.
  virtual void ExitBlock(absl::string_view label, x::proto::Object* object) = 0;

  // Marks the beginning of a log region where key is mapped to value val.
  // If key is currently set to some value, the new mapping supercedes
  // the prior one until the next call to UnsetAttribute() with the same key.
  virtual void SetAttribute(absl::string_view key, absl::string_view value) = 0;

  // Marks the end of a log region where key is mapped to value val. If at the
  // time of the last call to SetAttribute() with key, key was already mapped
  // to a value, the Sight reverts to using that mapping.
  virtual void UnsetAttribute(absl::string_view key) = 0;

  // Returns teh value currently mapped to key or an empty string if key
  // is currently unmapped.
  virtual absl::string_view GetAttribute(absl::string_view key) = 0;

  // Emits object to the log.
  // Returns the location of the emitted log object.
  virtual Location LogObject(x::proto::Object* object) = 0;
};

// Abstract class for helper objects that annotate the log during a given
// lexical scope. This class documents the API that must be provided by all
// such objects.
class SightScopedHelper {
 public:
  virtual ~SightScopedHelper() {}

  // Specifies the code location where the object was created, as well as the
  // instanec of Sight that manages the log.
  virtual void SetEnterCodeLocation(absl::string_view file, int line,
                                    absl::string_view func,
                                    SightCoreInterface* sight) = 0;
};

// Class makes it easy to match the lifetimes of blocks in the log to
// lexical scopes in the source code since they will match the lifetime of
// Block-type variable objects.
class Block : public SightScopedHelper {
 public:
  explicit Block(absl::string_view label, SightCoreInterface* sight);
  explicit Block(absl::string_view label);
  ~Block() override;

  void SetEnterCodeLocation(absl::string_view file, int line,
                            absl::string_view func,
                            SightCoreInterface* sight) override;

  const std::string& label() const { return label_; }

 private:
  std::string label_;
  SightCoreInterface* sight_;
};

// Class that makes it easy to match the lifetimes of attributes in the log to
// lexical scopes in the source code since they will match the lifetime of
// Attribute-type variable objects.
class Attribute : public SightScopedHelper {
 public:
  explicit Attribute(absl::string_view key, absl::string_view val,
                     SightCoreInterface* sight);
  explicit Attribute(absl::string_view key, absl::string_view val);
  ~Attribute() override;

  void SetEnterCodeLocation(absl::string_view file, int line,
                            absl::string_view func,
                            SightCoreInterface* sight) override;

  const std::string& key() const { return key_; }
  const std::string& val() const { return val_; }

 private:
  std::string key_;
  std::string val_;
  SightCoreInterface* sight_;
};

// Class that makes it easy to pause logging for the duration of a given
// lexical scope.
class Quiet : public SightScopedHelper {
 public:
  explicit Quiet(SightCoreInterface* sight);
  explicit Quiet();
  ~Quiet() override;

  void SetEnterCodeLocation(absl::string_view file, int line,
                            absl::string_view func,
                            SightCoreInterface* sight) override;

 private:
  SightCoreInterface* sight_;
};

// Drop-in-replacement for the current LOG(*) macros that preserves line number
// information provided by LOG(*) while also making it possible to process the
// same log text via Sight.
#define SEE(log_level, log_text, sight_obj)                                   \
  LOG(log_level) << sight_obj->TextLineWithMeta(log_text, __FILE__, __LINE__, \
                                                __func__);

// Macro for using helper objects that annotate the log from a lexical scope.
// SEEOBJ(ObjectType, ObjectParams, sight::Sight instance).
// The macro takes care of creating a uniquely-named temporary variable that
// has the lifetime of the current lexical scope and documents the code location
// of the macro.
#define CONCAT_IMPL(x, y) x##y
#define MACRO_CONCAT(x, y) CONCAT_IMPL(x, y)
#define SEEOBJ(type, params, sight_obj)                      \
  type MACRO_CONCAT(sight_logging_object_, __LINE__) params; \
  MACRO_CONCAT(sight_logging_object_, __LINE__)              \
      .SetEnterCodeLocation(__FILE__, __LINE__, __func__, sight_obj);

}  // namespace sight

#endif  // GOOGLEX_CORTEX_SIGHT_SIGHT_CORE_INTERFACE_H_
