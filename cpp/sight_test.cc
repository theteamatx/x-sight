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

#include "analysis/dremel/core/capacitor/public/record-reader.h"
#include "file/base/file.h"
#include "file/base/filesystem.h"
#include "file/base/helpers.h"
#include "file/base/path.h"
#include "file/columnio/public/record-reader.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace sight {
namespace {

using ::testing::Eq;
using ::testing::EqualsProto;

TEST(SightTest, SeeTextOutput) {
  std::string log_dir_path =
      file::JoinPath(absl::GetFlag(FLAGS_test_tmpdir), "SeeTextOutput");
  ASSERT_OK(file::CreateDir(log_dir_path, file::Overwrite()));
  auto sight_s = Sight::Create(Sight::Params()
                                   .WithLocal()
                                   .WithLogDirPath(log_dir_path)
                                   .WithTextOutput());
  ASSERT_OK(sight_s.status());
  {
    auto sight = std::move(sight_s).ValueOrDie();
    SEE(INFO, "text", sight)
  }

  std::string actual_log_contents;
  ASSERT_OK(file::GetContents(file::JoinPath(log_dir_path, "log.txt"),
                              &actual_log_contents, file::Defaults()));
  EXPECT_THAT(actual_log_contents,
              Eq(
                  R"(googlex/cortex/sight/sight_test.cc:29/TestBody text
)"));
}

TEST(SightTest, SeeCapacitorOutput) {
  std::string log_dir_path =
      file::JoinPath(absl::GetFlag(FLAGS_test_tmpdir), "SeeCapacitorOutput");
  ASSERT_OK(file::CreateDir(log_dir_path, file::Overwrite()));
  auto sight_s = Sight::Create(Sight::Params()
                                   .WithLocal()
                                   .WithLogDirPath(log_dir_path)
                                   .WithCapacitorOutput());
  ASSERT_OK(sight_s.status());
  {
    auto sight = std::move(sight_s).ValueOrDie();
    SEE(INFO, "text", sight)
  }

  File* actual_file;
  EXPECT_OK(file::Open(file::JoinPath(log_dir_path, "log.capacitor"), "r",
                       &actual_file, file::Defaults()));
  std::string error;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<dremel::capacitor::RecordReader> actual_log_reader,
      dremel::capacitor::RecordReader::Create(
          actual_file, TAKE_OWNERSHIP,
          dremel::capacitor::RecordReader::DefaultOptionsAllColumns()));
  EXPECT_EQ(actual_log_reader->NumRecords(), 1);

  x::proto::SingleLogObject object;
  ASSERT_OK_AND_ASSIGN(bool read_success,
                       actual_log_reader->ReadToMessage(&object));
  ASSERT_TRUE(read_success);
  EXPECT_THAT(object, EqualsProto(R"pb(
                obj {
                  location: '0'
                  sub_type: ST_TEXT
                  text { text: 'text\\n' }
                  file: 'googlex/cortex/sight/sight_test.cc'
                  line: 52
                  func: 'TestBody'
                  ancestor_start_location: '0'
                }
              )pb"));
}

TEST(SightTest, BlockTextOutput) {
  std::string log_dir_path =
      file::JoinPath(absl::GetFlag(FLAGS_test_tmpdir), "BlockTextOutput");
  ASSERT_OK(file::CreateDir(log_dir_path, file::Overwrite()));
  auto sight_s = Sight::Create(Sight::Params()
                                   .WithLocal()
                                   .WithLogDirPath(log_dir_path)
                                   .WithTextOutput());
  ASSERT_OK(sight_s.status());
  {
    auto sight = std::move(sight_s).ValueOrDie();
    { Block b("block", sight.get()); }
  }

  std::string actual_log_contents;
  ASSERT_OK(file::GetContents(file::JoinPath(log_dir_path, "log.txt"),
                              &actual_log_contents, file::Defaults()));
  EXPECT_THAT(actual_log_contents, Eq(
                                       R"(block<<<<
block>>>>
)"));
}

TEST(SightTest, BlockCapacitorOutput) {
  std::string log_dir_path =
      file::JoinPath(absl::GetFlag(FLAGS_test_tmpdir), "BlockCapacitorOutput");
  ASSERT_OK(file::CreateDir(log_dir_path, file::Overwrite()));
  auto sight_s = Sight::Create(Sight::Params()
                                   .WithLocal()
                                   .WithLogDirPath(log_dir_path)
                                   .WithCapacitorOutput()
                                   .WithLogOwner("owner"));
  ASSERT_OK(sight_s.status());
  {
    auto sight = std::move(sight_s).ValueOrDie();
    { Block b("block", sight.get()); }
  }

  std::string error;
  std::unique_ptr<::file::columnio::RecordReader> actual_log_reader =
      absl::WrapUnique<::file::columnio::RecordReader>(
          ::file::columnio::RecordReader::Create(
              file::JoinPath(log_dir_path, "log.capacitor"), "*", &error));
  ASSERT_TRUE(actual_log_reader != nullptr) << error;

  EXPECT_EQ(actual_log_reader->NumRows(), 2);
  x::proto::SingleLogObject object;
  EXPECT_TRUE(actual_log_reader->ReadToMessage(&object));
  EXPECT_THAT(object, EqualsProto(R"pb(
                obj {
                  location: '0'
                  sub_type: ST_BLOCK_START
                  block_start { label: 'block' }
                  ancestor_start_location: '0'
                })pb"));
  EXPECT_TRUE(actual_log_reader->ReadToMessage(&object));
  EXPECT_THAT(object, EqualsProto(R"pb(
                obj {
                  location: '1'
                  sub_type: ST_BLOCK_END
                  block_end { label: 'block' location_of_block_start: '0' }
                  index: 1
                  ancestor_start_location: '1'
                })pb"));
}

TEST(SightTest, NestedBlockTextOutput) {
  std::string log_dir_path =
      file::JoinPath(absl::GetFlag(FLAGS_test_tmpdir), "NestedBlockTextOutput");
  ASSERT_OK(file::CreateDir(log_dir_path, file::Overwrite()));
  auto sight_s = Sight::Create(Sight::Params()
                                   .WithLocal()
                                   .WithLogDirPath(log_dir_path)
                                   .WithTextOutput());
  ASSERT_OK(sight_s.status());
  {
    auto sight = std::move(sight_s).ValueOrDie();
    {
      Block b("block1", sight.get());
      {
        Block b("block2", sight.get());
        SEE(INFO, "text", sight)
      }
    }
  }

  std::string actual_log_contents;
  ASSERT_OK(file::GetContents(file::JoinPath(log_dir_path, "log.txt"),
                              &actual_log_contents, file::Defaults()));
  EXPECT_THAT(actual_log_contents, Eq(
                                       R"(block1<<<<
block1: block2<<<<
googlex/cortex/sight/sight_test.cc:164/TestBody block1: block2: text
block1: block2>>>>
block1>>>>
)"));
}

TEST(SightTest, NestedBlockCapacitorOutput) {
  std::string log_dir_path = file::JoinPath(absl::GetFlag(FLAGS_test_tmpdir),
                                            "NestedBlockCapacitorOutput");
  ASSERT_OK(file::CreateDir(log_dir_path, file::Overwrite()));
  auto sight_s = Sight::Create(Sight::Params()
                                   .WithLocal()
                                   .WithLogDirPath(log_dir_path)
                                   .WithCapacitorOutput()
                                   .WithLogOwner("owner"));
  ASSERT_OK(sight_s.status());
  {
    auto sight = std::move(sight_s).ValueOrDie();
    {
      Block b("block1", sight.get());
      {
        Block b("block2", sight.get());
        SEE(INFO, "text", sight)
      }
    }
  }

  std::string error;
  std::unique_ptr<::file::columnio::RecordReader> actual_log_reader =
      absl::WrapUnique<::file::columnio::RecordReader>(
          ::file::columnio::RecordReader::Create(
              file::JoinPath(log_dir_path, "log.capacitor"), "*", &error));
  ASSERT_TRUE(actual_log_reader != nullptr) << error;

  EXPECT_EQ(actual_log_reader->NumRows(), 5);
  x::proto::SingleLogObject object;
  EXPECT_TRUE(actual_log_reader->ReadToMessage(&object));
  EXPECT_THAT(object, EqualsProto(R"pb(
                obj {
                  location: "0"
                  sub_type: ST_BLOCK_START
                  block_start { label: "block1" }
                  ancestor_start_location: '0'
                })pb"));
  EXPECT_TRUE(actual_log_reader->ReadToMessage(&object));
  EXPECT_THAT(object, EqualsProto(R"pb(
                obj {
                  location: "0:0"
                  sub_type: ST_BLOCK_START
                  block_start { label: "block2" }
                  index: 1
                  ancestor_start_location: '0'
                  ancestor_start_location: '0:0'
                })pb"));
  EXPECT_TRUE(actual_log_reader->ReadToMessage(&object));
  EXPECT_THAT(object, EqualsProto(R"pb(
                obj {
                  location: "0:0:0"
                  sub_type: ST_TEXT
                  text { text: "text\\n" }
                  index: 2
                  file: "googlex/cortex/sight/sight_test.cc"
                  line: 197
                  func: "TestBody"
                  ancestor_start_location: '0'
                  ancestor_start_location: '0:0'
                  ancestor_start_location: '0:0:0'
                })pb"));
  EXPECT_TRUE(actual_log_reader->ReadToMessage(&object));
  EXPECT_THAT(object, EqualsProto(R"pb(
                obj {
                  location: "0:1"
                  sub_type: ST_BLOCK_END
                  block_end {
                    label: "block2"
                    location_of_block_start: "0:0"
                    num_direct_contents: 1
                    num_transitive_contents: 1
                  }
                  index: 3
                  ancestor_start_location: '0'
                  ancestor_start_location: '0:1'
                })pb"));
  EXPECT_TRUE(actual_log_reader->ReadToMessage(&object));
  EXPECT_THAT(object, EqualsProto(R"pb(
                obj {
                  location: "1"
                  sub_type: ST_BLOCK_END
                  block_end {
                    label: "block1"
                    location_of_block_start: "0"
                    num_direct_contents: 1
                    num_transitive_contents: 3
                  }
                  index: 4
                  ancestor_start_location: '1'
                })pb"));
}

TEST(SightTest, AttributesTextOutput) {
  std::string log_dir_path =
      file::JoinPath(absl::GetFlag(FLAGS_test_tmpdir), "Attributes");
  ASSERT_OK(file::CreateDir(log_dir_path, file::Overwrite()));
  auto sight_s = Sight::Create(Sight::Params()
                                   .WithLocal()
                                   .WithLogDirPath(log_dir_path)
                                   .WithTextOutput());
  ASSERT_OK(sight_s.status());
  {
    auto sight = std::move(sight_s).ValueOrDie();
    {
      Attribute b("key", "val", sight.get());
      SEE(INFO, "text", sight)
    }
  }
  std::string actual_log_contents;
  ASSERT_OK(file::GetContents(file::JoinPath(log_dir_path, "log.txt"),
                              &actual_log_contents, file::Defaults()));
  EXPECT_THAT(
      actual_log_contents,
      Eq(
          R"(googlex/cortex/sight/sight_test.cc:287/TestBody text| key=val
)"));
}

TEST(SightTest, AttributesCapacitorOutput) {
  std::string log_dir_path =
      file::JoinPath(absl::GetFlag(FLAGS_test_tmpdir), "See");
  ASSERT_OK(file::CreateDir(log_dir_path, file::Overwrite()));
  auto sight_s = Sight::Create(Sight::Params()
                                   .WithLocal()
                                   .WithLogDirPath(log_dir_path)
                                   .WithCapacitorOutput()
                                   .WithLogOwner("owner"));
  ASSERT_OK(sight_s.status());
  {
    auto sight = std::move(sight_s).ValueOrDie();
    {
      Attribute b("key", "val", sight.get());
      SEE(INFO, "text", sight)
    }
  }

  std::string error;
  std::unique_ptr<::file::columnio::RecordReader> actual_log_reader =
      absl::WrapUnique<::file::columnio::RecordReader>(
          ::file::columnio::RecordReader::Create(
              file::JoinPath(log_dir_path, "log.capacitor"), "*", &error));
  ASSERT_TRUE(actual_log_reader != nullptr) << error;

  EXPECT_EQ(actual_log_reader->NumRows(), 1);
  x::proto::SingleLogObject object;
  EXPECT_TRUE(actual_log_reader->ReadToMessage(&object));
  EXPECT_THAT(object, EqualsProto(R"pb(
                obj {
                  location: '0'
                  attribute { key: 'key' value: 'val' }
                  sub_type: ST_TEXT
                  text { text: 'text\\n' }
                  index: 0
                  file: 'googlex/cortex/sight/sight_test.cc'
                  line: 314
                  func: 'TestBody'
                  ancestor_start_location: '0'
                })pb"));
}

TEST(SightTest, Quiet) {
  std::string log_dir_path =
      file::JoinPath(absl::GetFlag(FLAGS_test_tmpdir), "Quiet");
  ASSERT_OK(file::CreateDir(log_dir_path, file::Overwrite()));
  auto sight_s = Sight::Create(Sight::Params()
                                   .WithLocal()
                                   .WithLogDirPath(log_dir_path)
                                   .WithTextOutput());
  ASSERT_OK(sight_s.status());
  {
    auto sight = std::move(sight_s).ValueOrDie();
    SEE(INFO, "loud text", sight);
    {
      sight::Quiet q(sight.get());
      SEE(INFO, "quiet text", sight);
    }
  }

  std::string actual_log_contents;
  ASSERT_OK(file::GetContents(file::JoinPath(log_dir_path, "log.txt"),
                              &actual_log_contents, file::Defaults()));
  EXPECT_THAT(actual_log_contents,
              Eq(
                  R"(googlex/cortex/sight/sight_test.cc:353/TestBody loud text
)"));
}

}  // namespace
}  // namespace sight
