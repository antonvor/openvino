// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <shared_test_classes/single_layer/convolution_backprop_data.hpp>


using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {
using LayerTestsDefinitions::convBackpropDataSpecificParams;
using LayerTestsDefinitions::convBackpropDataLayerTestParamsSet;

typedef std::tuple<
    convBackpropDataLayerTestParamsSet,
    bool,
    CPUSpecificParams,
    fusingSpecificParams,
    std::map<std::string, std::string> > deconvLayerCPUTestParamsSet;

class DeconvolutionLayerCPUTest : public testing::WithParamInterface<deconvLayerCPUTestParamsSet>,
    virtual public LayerTestsUtils::LayerTestsCommon, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<deconvLayerCPUTestParamsSet> obj) {
        convBackpropDataLayerTestParamsSet basicParamsSet;
        bool withBiases;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, withBiases, cpuParams, fusingParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::ConvolutionBackpropDataLayerTest::getTestCaseName(testing::TestParamInfo<convBackpropDataLayerTestParamsSet>(
            basicParamsSet, 0));

        result << "_withBiases=" << withBiases;
        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                result << "_" << item.first << "=" << item.second;
            }
        }

        return result.str();
    }
protected:
    void SetUp() override {
        using namespace ngraph;
        convBackpropDataLayerTestParamsSet basicParamsSet;
        bool withBiases;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(basicParamsSet, withBiases, cpuParams, fusingParams, additionalConfig) = this->GetParam();

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        convBackpropDataSpecificParams convParams;
        std::vector<size_t> inputShape;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = basicParamsSet;

        if (inPrc == Precision::UNSPECIFIED)
            inPrc = Precision::FP32;
        if (outPrc == Precision::UNSPECIFIED)
            outPrc = Precision::FP32;

        if (inPrc == Precision::U8) {
            selectedType += std::string("_") + Precision(Precision::I8).name();
        } else {
            selectedType += std::string("_") + inPrc.name();
        }

        op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;
        auto inElementType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
        auto outElementType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(outPrc);

        auto inputParams = builder::makeParams(inElementType, { inputShape });
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(inputParams));

        auto weiPrc = (inElementType == element::u8) ? element::i8 : inElementType;
        auto deconvolutionNode = builder::makeConvolutionBackpropDataRelaxed(paramOuts.front(), weiPrc, outElementType, kernel, stride, padBegin,
            padEnd, dilation, padType, convOutChannels, withBiases);

        function = makeNgraphFunction(element::f32, inputParams, deconvolutionNode, "convolutionBackpropData");

        if (inPrc == Precision::U8 || inPrc == Precision::I8) {
            additionalPasses.push_back(std::make_shared<pass::ConvertPrecision<element::i8, element::f32>>());
            additionalPasses.push_back(std::make_shared<pass::ConvertPrecision<element::u8, element::f32>>());
        }

        if (outPrc != Precision::FP32) {
            additionalPasses.push_back(std::make_shared<ConvertPrecision<opset1::ConvolutionBackpropData>>());
            if (withBiases)
                additionalPasses.push_back(std::make_shared<pass::ConvertPrecision<element::i32, element::f32>>());
        }
    }
};

TEST_P(DeconvolutionLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Deconvolution");
}

namespace {

/* COMMON PARAMS */
const std::vector<fusingSpecificParams> fusingParamsSet{
        emptyFusingSpec,
//        fusingScaleShift
};

const std::map<std::string, std::string> cpuEmptyPluginConfig;
const std::map<std::string, std::string> cpuBF16PluginConfig = { { PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES } };

/* ============= Deconvolution params (planar layout) ============= */
const SizeVector numOutChannels_Planar = { 6 };

/* ============= Deconvolution params (blocked layout) ============= */
const SizeVector numOutChannels_Blocked = { 64 };

/* ============= Deconvolution params (2D) ============= */
const std::vector<SizeVector> kernels2d = { {3, 3}, {1, 1} };
const std::vector<SizeVector> strides2d = { {1, 1}, {2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins2d = { {0, 0} };
const std::vector<std::vector<ptrdiff_t>> padEnds2d = { {0, 0} };
const std::vector<SizeVector> dilations2d = { {1, 1} };

/* ============= Deconvolution params (3D) ============= */
const std::vector<SizeVector> kernels3d = { {3, 3, 3}, {1, 1, 1} };
const std::vector<SizeVector> strides3d = { {1, 1, 1}, {2, 2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins3d = { {0, 0, 0} };
const std::vector<std::vector<ptrdiff_t>> padEnds3d = { {0, 0, 0} };
const std::vector<SizeVector> dilations3d = { {1, 1, 1} };
/* ============= */

/* INSTANCES */
/* ============= Deconvolution (Planar 2D) ============= */
const auto convParams_ExplicitPadding_Planar_2D = ::testing::Combine(
    ::testing::ValuesIn(kernels2d),
    ::testing::ValuesIn(strides2d),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::ValuesIn(numOutChannels_Planar),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_2D_Planar_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Planar_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::Values(false),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_2D_Planar_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Planar_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::Values(false),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_2D_U8, DeconvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        convParams_ExplicitPadding_Planar_2D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::U8/*, Precision::I8 */),
                                        ::testing::Values(Precision::FP32, Precision::I32),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7 })),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::Values(false, true),
                                ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_I8})),
                                ::testing::ValuesIn(fusingParamsSet),
                                ::testing::Values(cpuEmptyPluginConfig)),
                        DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Deconvolution (Planar 3D) ============= */
const auto convParams_ExplicitPadding_Planar_3D = ::testing::Combine(
    ::testing::ValuesIn(kernels3d),
    ::testing::ValuesIn(strides3d),
    ::testing::ValuesIn(padBegins3d),
    ::testing::ValuesIn(padEnds3d),
    ::testing::ValuesIn(dilations3d),
    ::testing::ValuesIn(numOutChannels_Planar),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_3D_Planar_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Planar_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::Values(false),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_3D_Planar_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Planar_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::Values(false),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_3D_I8, DeconvolutionLayerCPUTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        convParams_ExplicitPadding_Planar_3D,
                                        ::testing::Values(Precision::FP32),
                                        ::testing::Values(Precision::U8/*, Precision::I8 */),
                                        ::testing::Values(Precision::UNSPECIFIED),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(Layout::ANY),
                                        ::testing::Values(std::vector<size_t >({ 2, 12, 7, 7, 7 })),
                                        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                ::testing::Values(false),
                                ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D_I8})),
                                ::testing::ValuesIn(fusingParamsSet),
                                ::testing::Values(cpuEmptyPluginConfig)),
                        DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupDeconvolution (Blocked 2D) ============= */
const auto convParams_ExplicitPadding_Blocked_2D = ::testing::Combine(
    ::testing::ValuesIn(kernels2d),
    ::testing::ValuesIn(strides2d),
    ::testing::ValuesIn(padBegins2d),
    ::testing::ValuesIn(padEnds2d),
    ::testing::ValuesIn(dilations2d),
    ::testing::ValuesIn(numOutChannels_Blocked),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_2D_Blocked_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Blocked_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 67, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::Values(false),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_2D_Blocked_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Blocked_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 67, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::Values(false),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= GroupDeconvolution (Blocked 3D) ============= */
const auto convParams_ExplicitPadding_Blocked_3D = ::testing::Combine(
    ::testing::ValuesIn(kernels3d),
    ::testing::ValuesIn(strides3d),
    ::testing::ValuesIn(padBegins3d),
    ::testing::ValuesIn(padEnds3d),
    ::testing::ValuesIn(dilations3d),
    ::testing::ValuesIn(numOutChannels_Blocked),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_3D_Blocked_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Blocked_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 67, 7, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::Values(false),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_3D_Blocked_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_Blocked_3D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 67, 7, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::Values(false),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Kernel_1x1 (2D) ============= */

const auto convParams_ExplicitPadding_1x1_2D = ::testing::Combine(
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_2D_1x1_FP32, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_1x1_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Precision::UNSPECIFIED),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 67, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::Values(false),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_1x1})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuEmptyPluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Deconv_2D_1x1_BF16, DeconvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(
            convParams_ExplicitPadding_1x1_2D,
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Precision::BF16),
            ::testing::Values(Layout::ANY),
            ::testing::Values(Layout::ANY),
            ::testing::Values(std::vector<size_t >({ 2, 67, 7, 7 })),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ::testing::Values(false),
        ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_1x1})),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::Values(cpuBF16PluginConfig)),
    DeconvolutionLayerCPUTest::getTestCaseName);

/* ========= */

} // namespace
} // namespace CPULayerTestsDefinitions
