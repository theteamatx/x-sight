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

#ifndef GOOGLEX_CORTEX_SIGHT_SIGHT_INTERFACE_H_
#define GOOGLEX_CORTEX_SIGHT_SIGHT_INTERFACE_H_

#include <vector>

#include "base/logging.h"
#include "base/timer.h"
#include "file/base/file.h"
#include "file/base/path.h"
#include "googlex/cortex/sight/sight_core_interface.h"
#include "third_party/absl/strings/string_view.h"

namespace sight {

// Specifies a complete version of the generic interface to structured
// logging that basic logging functionality as well as access to any
// of the widgets.
class SightInterface : public SightCoreInterface {
 public:
  ~SightInterface() override {}
};

}  // namespace sight

#endif  // GOOGLEX_CORTEX_SIGHT_SIGHT_INTERFACE_H_
