#include "core/graph/record.h"
#include "gtest/gtest.h"

using namespace Lotus::Common;

namespace Lotus {
namespace Test {
TEST(RecordTest, CommonDataStructureTest) {
  Record<std::string> stringRecord;
  const std::string* n2;
  EXPECT_FALSE(stringRecord.GetName(0, &n2).IsOK());

  // One way to store feature vector using Record.
  std::vector<std::string> names = {"featureName", "featureValue"};
  std::tuple<std::string, float> values("streamLength", 2.0f);
  Record<std::string, float> record(names, values);

  const std::string* name;
  auto status = record.GetName(2, &name);
  EXPECT_FALSE(status.IsOK());

  record.GetName(0, &name);
  auto& value = std::get<0>(record.GetValues());
  EXPECT_EQ("featureName", *name);
  EXPECT_EQ("streamLength", value);

  record.GetName(1, &name);
  auto& value2 = std::get<1>(record.GetValues());
  EXPECT_EQ("featureValue", *name);
  EXPECT_EQ(2.0f, value2);

  // Another way to store feature vector using Record.
  names = {"streamLength"};
  std::tuple<float> values2(2.0f);
  Record<float> record2(names, values2);

  record2.GetName(0, &name);
  auto& value3 = std::get<0>(record2.GetValues());
  EXPECT_EQ("streamLength", *name);
  EXPECT_EQ(2.0f, value3);
}
}  // namespace Test
}  // namespace Lotus