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

#include "base/commandlineflags.h"
#include "base/init_google.h"
#include "googlex/cortex/sight/sight.h"
#include "googlex/cortex/sight/sight_core_interface.h"

ABSL_FLAG(std::string, log_path, "/tmp", "Path of the output log file");
ABSL_FLAG(std::string, label, "Generic Demo", "Unique label of this run");

using sight::Attribute;
using sight::Block;
using sight::Sight;

int main(int argc, char** argv) {
  InitGoogle(argv[0], &argc, &argv, false);

  auto sight_s = Sight::Create(Sight::Params()
                                   .WithLabel(absl::GetFlag(FLAGS_label))
                                   .WithTextOutput()
                                   .WithCapacitorOutput()
                                   .WithLogOwner("user@domain.com"));
  if (!sight_s.ok()) {
    LOG(ERROR) << sight_s.status();
    return -1;
  }
  auto sight = std::move(sight_s).ValueOrDie();

  {
    SEEOBJ(Block, ("A"), sight.get());
    {
      SEEOBJ(Attribute, ("key", "A"), sight.get());
      SEEOBJ(Block, ("A1"), sight.get());
      {
        SEEOBJ(Block, ("A1.1"), sight.get());
        SEE(INFO, "A1.1 text", sight);
      }
    }
  }
  {
    SEEOBJ(Block, ("B"), sight.get()) {
      SEEOBJ(Attribute, ("key", "B"), sight.get());
      SEEOBJ(Attribute, ("key1", "B"), sight.get());
      SEEOBJ(Attribute, ("key2", "B"), sight.get());
      SEEOBJ(Attribute, ("key3", "B"), sight.get());
      SEEOBJ(Block, ("B1"), sight.get()) {
        SEEOBJ(Block, ("B1.1"), sight.get());
        SEE(INFO, "B1.1 text", sight);
      }
      SEEOBJ(Block, ("B2"), sight.get()) {
        SEEOBJ(Attribute, ("keyin", "X"), sight.get());
        SEEOBJ(Attribute, ("keyin1", "X"), sight.get());
        SEEOBJ(Attribute, ("keyin2", "X"), sight.get());
        SEEOBJ(Attribute, ("keyin3", "X"), sight.get());
        {
          SEEOBJ(Block, ("B2.1"), sight.get())
          SEE(INFO, "B2.1 text", sight);
        }
      }
      SEEOBJ(Block, ("B3"), sight.get()) {
        SEEOBJ(Block, ("B3.1"), sight.get())
        SEE(INFO, "B3.1 text", sight);
      }
    }
  }

  return 0;
}
