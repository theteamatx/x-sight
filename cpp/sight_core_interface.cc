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

#include "googlex/cortex/sight/sight_core_interface.h"

#include "file/base/file.h"
#include "file/base/helpers.h"
#include "file/base/path.h"
#include "third_party/absl/memory/memory.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/str_replace.h"
#include "third_party/absl/strings/substitute.h"
#include "util/gtl/map_util.h"
#include "util/task/status_macros.h"

namespace sight {

Block::Block(absl::string_view label, SightCoreInterface* sight)
    : label_(static_cast<std::string>(label)), sight_(sight) {
  sight_->EnterBlock(label_);
}

Block::Block(absl::string_view label)
    : label_(static_cast<std::string>(label)) {}

Block::~Block() { sight_->ExitBlock(label_); }

void Block::SetEnterCodeLocation(absl::string_view file, int line,
                                 absl::string_view func,
                                 SightCoreInterface* sight) {
  sight_ = sight;
  sight_->EnterBlock(label_, file, line, func);
}

Attribute::Attribute(absl::string_view key, absl::string_view val,
                     SightCoreInterface* sight)
    : key_(static_cast<std::string>(key)), val_(val), sight_(sight) {
  sight_->SetAttribute(key_, val_);
}

Attribute::Attribute(absl::string_view key, absl::string_view val)
    : key_(static_cast<std::string>(key)), val_(val) {}

void Attribute::SetEnterCodeLocation(absl::string_view file, int line,
                                     absl::string_view func,
                                     SightCoreInterface* sight) {
  sight_ = sight;
  sight_->SetAttribute(key_, val_);
}

Attribute::~Attribute() { sight_->UnsetAttribute(key_); }

Quiet::Quiet(SightCoreInterface* sight) : sight_(sight) {
  sight_->PauseLogging();
}

Quiet::Quiet() {}

void Quiet::SetEnterCodeLocation(absl::string_view file, int line,
                                 absl::string_view func,
                                 SightCoreInterface* sight) {
  sight_ = sight;
  sight_->PauseLogging();
}

Quiet::~Quiet() { sight_->ResumeLogging(); }

}  // namespace sight
