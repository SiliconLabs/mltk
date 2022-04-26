#pragma once

#include "gtest/gtest.h"


namespace testing
{
 namespace internal
 {
  enum class GTestColor { kDefault, kRed, kGreen, kYellow };
  extern void ColoredPrintf(GTestColor color, const char* fmt, ...);
 }
}
#define GTEST_PRINTF(fmt, ...)  do { \
::testing::internal::ColoredPrintf(::testing::internal::GTestColor::kYellow, "[          ] " fmt "\n", ## __VA_ARGS__); \
} while(0)

// C++ stream interface
class TestCout : public std::stringstream
{
public:
    ~TestCout()
    {
        GTEST_PRINTF("%s\n",str().c_str());
    }
};

#define GTEST_COUT  TestCout()
#define GTEST_SKIP_TEST(message) GTEST_MESSAGE_(message, ::testing::TestPartResult::kSkip)
#define GTEST_FAILED() ::testing::Test::HasFailure()
#define GTEST_SHOULD_CONTINUE() (!(::testing::Test::HasFailure() || ::testing::Test::IsSkipped()))