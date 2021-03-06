// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "shared_test_classes/base/layer_test_utils.hpp"

int main(int argc, char* argv[]) {
    FuncTestUtils::SkipTestsConfig::disable_tests_skipping = false;
    LayerTestsUtils::extendReport = false;
    bool print_custom_help = false;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--disable_tests_skipping") {
            FuncTestUtils::SkipTestsConfig::disable_tests_skipping = true;
        } else if (std::string(argv[i]) == "--extend_report") {
            LayerTestsUtils::extendReport = true;
        } else if (std::string(argv[i]) == "--help") {
            print_custom_help = true;
        }
    }
    if (print_custom_help) {
        std::cout << "Custom command line argument:" << std::endl;
        std::cout << "  --disable_tests_skipping" << std::endl;
        std::cout << "       Ignore tests skipping rules and run all the test" << std::endl;
        std::cout << "       (except those which are skipped with DISABLED prefix)" << std::endl;
        std::cout << "  --extend_report" << std::endl;
        std::cout << "       Extend operation coverage report without overwriting the device results" << std::endl;
        std::cout << std::endl;
    }
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new LayerTestsUtils::TestEnvironment);
    auto retcode = RUN_ALL_TESTS();

    return retcode;
}
