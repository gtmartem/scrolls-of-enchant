#include <math.h>
#include "gtest/gtest.h"

double squareRoot(double x)
{
    if (x < 0)
        return -1;
    else
        {
            return sqrt(x);
        }
}

TEST(SquareRootTest, PositiveNos)
{
    EXPECT_EQ(18.0, squareRoot(324.0));
    EXPECT_EQ(25.4, squareRoot(645.16));
    EXPECT_EQ(2.0, squareRoot(4.0));
    EXPECT_EQ(2.0, squareRoot(5.0));
}

TEST(SquareRootTest, ZeroAndNegativeNox)
{
    ASSERT_EQ(0.0, squareRoot(0.0));
    ASSERT_EQ(-1, squareRoot(-22.0));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}