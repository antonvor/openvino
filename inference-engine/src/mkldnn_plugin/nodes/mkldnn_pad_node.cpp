// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_pad_node.h"
#include <legacy/ie_layers.h>
#include <string>
#include <cmath>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <limits>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNPadNode::MKLDNNPadNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNPadNode::getSupportedDescriptors() {
    auto* padLayer = dynamic_cast<PadLayer*>(getCnnLayer().get());
    if (padLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert Pad layer.";

    padsBegin = padLayer->GetParamAsUInts("pads_begin");
    padsEnd = padLayer->GetParamAsUInts("pads_end");

    SizeVector srcDims = padLayer->insData[0].lock()->getTensorDesc().getDims();
    SizeVector dstDims = padLayer->outData[0]->getTensorDesc().getDims();
    if (srcDims.size() != dstDims.size() || padsBegin.size() != srcDims.size() || padsEnd.size() != srcDims.size())
        THROW_IE_EXCEPTION << padLayer->name << " Incorrect number of input/output dimensions!";

    std::string pad_mode = padLayer->GetParamAsString("pad_mode");
    if (pad_mode == "constant") {
        padMode = CONSTANT;
        padValue = padLayer->GetParamAsFloat("pad_value", 0.f);
    } else if (pad_mode == "edge") {
        padMode = EDGE;
    } else if (pad_mode == "reflect") {
        padMode = REFLECT;
        for (size_t i = 0; i < srcDims.size(); i++) {
            if ((srcDims[i] - 1) < padsBegin[i] || (srcDims[i] - 1) < padsEnd[i])
                THROW_IE_EXCEPTION << padLayer->name << " Incorrect padsBegin or padsEnd for 'reflect' pad mode";
        }
    } else if (pad_mode == "symmetric") {
        padMode = SYMMETRIC;
        for (size_t i = 0; i < srcDims.size(); i++) {
            if (srcDims[i] < padsBegin[i] || srcDims[i] < padsEnd[i])
                THROW_IE_EXCEPTION << padLayer->name << " Incorrect padsBegin or padsEnd for 'symmetric' pad mode";
        }
    } else {
        THROW_IE_EXCEPTION << padLayer->name
                           << " Incorrect pad_mode. Only constants|edge|reflect|symmetric modes are supported!";
    }

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();
}

void MKLDNNPadNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto srcDims = getParentEdgeAt(0)->getDims();

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;

    auto memoryFormat = MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims());
    config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), dataType, memoryFormat);
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), dataType, memoryFormat);
    supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memoryFormat});
}

void MKLDNNPadNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    SizeVector srcDims = getParentEdgeAt(0)->getBlob()->getTensorDesc().getDims();
    SizeVector dstDims = getChildEdgeAt(0)->getBlob()->getTensorDesc().getDims();

    params.srcStrides = getParentEdgeAt(0)->getBlob()->getTensorDesc().getBlockingDesc().getStrides();
    params.dstStrides = getChildEdgeAt(0)->getBlob()->getTensorDesc().getBlockingDesc().getStrides();

    params.srcODims.resize(srcDims.size());
    params.padDims.resize(padsBegin.size());
    for (size_t i = 0; i < srcDims.size(); i++) {
        params.srcODims[i] = srcDims[i] + padsBegin[i];
        params.padDims[i] = padsBegin[i] + padsEnd[i];
        params.padPointsNum += padsBegin[i] + padsEnd[i];
    }
}

void MKLDNNPadNode::execute(mkldnn::stream strm) {
    const float *srcData = getParentEdgeAt(0)->getBlob()->cbuffer().as<const float*>() +
            getParentEdgeAt(0)->getBlob()->getTensorDesc().getBlockingDesc().getOffsetPadding();
    float* dstData = getChildEdgeAt(0)->getBlob()->buffer().as<float*>() +
            getChildEdgeAt(0)->getBlob()->getTensorDesc().getBlockingDesc().getOffsetPadding();

    auto srcDims = getParentEdgeAt(0)->getDims().ToSizeVector();
    auto dstDims = getChildEdgeAt(0)->getDims().ToSizeVector();

    switch (padMode) {
        case CONSTANT:
            padConstant(srcData, dstData, srcDims, dstDims);
            break;
        case EDGE:
            padEdge(srcData, dstData, srcDims, dstDims);
            break;
        case REFLECT:
            padReflect(srcData, dstData, srcDims, dstDims);
            break;
        case SYMMETRIC:
            padSymmetric(srcData, dstData, srcDims, dstDims);
            break;
    }
}

inline size_t parallel_init(size_t start, size_t size, std::vector<size_t> &counters, std::vector<size_t> &dims) {
    for (int j = size - 1; j >= 0; j--) {
        counters[j] = start % dims[j];
        start = start / dims[j];
    }
    return start;
}

inline void parallel_step(size_t size, std::vector<size_t> &counters, std::vector<size_t> &dims) {
    for (int j = size - 1; j >= 0; j--) {
        counters[j] = (counters[j] + 1) % dims[j];
        if (counters[j] != 0)
            return;
    }
}

void MKLDNNPadNode::padConstant(const float* srcData, float* dstData, SizeVector srcDims, SizeVector dstDims) {
    size_t dimsSize_1 = dstDims.size() - 1;
    size_t inputSV = srcDims[dimsSize_1];
    size_t workAmountSrc = params.srcStrides[0] * srcDims[0] / srcDims[dimsSize_1];

    int offset = 0;
    for (size_t i = 0; i < params.srcStrides.size(); ++i)
        offset += padsBegin[i] * params.dstStrides[i];
    std::fill_n(dstData, offset, padValue);

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector counters(dimsSize_1, 0);
        splitter(workAmountSrc, nthr, ithr, start, end);
        SizeVector countersNext(dimsSize_1, 0);

        parallel_init(start, dimsSize_1, counters, srcDims);
        parallel_init(start, dimsSize_1, countersNext, srcDims);
        parallel_step(dimsSize_1, countersNext, srcDims);
        int srcIdx = 0;
        int dstIdx = padsBegin[dimsSize_1];
        int dstIdxNext = padsBegin[dimsSize_1];
        for (size_t i = 0; i < dimsSize_1; ++i) {
            srcIdx += counters[i] * params.srcStrides[i];
            dstIdx += (padsBegin[i] + counters[i]) * params.dstStrides[i];
            dstIdxNext += (padsBegin[i] + countersNext[i]) * params.dstStrides[i];
        }
        if (dstIdxNext <= dstIdx) dstIdxNext = params.dstStrides[0] * dstDims[0];

        for (size_t iwork = start; iwork < end; ++iwork, srcIdx += inputSV) {
            cpu_memcpy(&dstData[dstIdx], &srcData[srcIdx], sizeof(float) * inputSV);
            std::fill_n(&dstData[dstIdx + inputSV], dstIdxNext - dstIdx - inputSV, padValue);

            for (int j = dimsSize_1 - 1; j >= 0; j--) {
                counters[j] = (counters[j] + 1) % srcDims[j];
                if (counters[j] != 0) {
                    dstIdx += params.dstStrides[j];
                    break;
                } else {
                    dstIdx = padsBegin[dimsSize_1];
                    for (size_t i = 0; i < dimsSize_1; ++i)
                        dstIdx += (padsBegin[i] + counters[i]) * params.dstStrides[i];
                }
            }

            for (int j = dimsSize_1 - 1; j >= 0; j--) {
                countersNext[j] = (countersNext[j] + 1) % srcDims[j];
                if (countersNext[j] != 0) {
                    dstIdxNext += params.dstStrides[j];
                    break;
                } else {
                    dstIdxNext = padsBegin[dimsSize_1];
                    for (size_t i = 0; i < dimsSize_1; ++i)
                        dstIdxNext += (padsBegin[i] + countersNext[i]) * params.dstStrides[i];
                }
            }

            if (dstIdxNext <= dstIdx) dstIdxNext = params.dstStrides[0] * dstDims[0];
        }
    });
}

void MKLDNNPadNode::padEdge(const float* srcData, float* dstData, SizeVector srcDims, SizeVector dstDims) {
    size_t dimsSize_1 = dstDims.size() - 1;
    size_t inputSV = dstDims[dimsSize_1];
    size_t workAmountDst = params.dstStrides[0] * dstDims[0] / dstDims[dimsSize_1];

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector counters(dimsSize_1, 0);
        splitter(workAmountDst, nthr, ithr, start, end);

        parallel_init(start, dimsSize_1, counters, dstDims);
        int dstIdx = 0;
        for (size_t i = 0; i < dimsSize_1; ++i)
            dstIdx += counters[i] * params.dstStrides[i];

        for (size_t iwork = start; iwork < end; ++iwork, dstIdx += inputSV) {
            int srcIdx = 0;
            for (size_t i = 0; i < dimsSize_1; ++i) {
                int idx = (counters[i] < padsBegin[i]) ? 0 :
                          ((counters[i] >= params.srcODims[i]) ? (srcDims[i] - 1) : (counters[i] - padsBegin[i]));
                srcIdx += idx * params.srcStrides[i];
            }

            std::fill_n(&dstData[dstIdx], padsBegin[dimsSize_1], srcData[srcIdx]);
            cpu_memcpy(&dstData[dstIdx + padsBegin[dimsSize_1]], &srcData[srcIdx],
                       srcDims[dimsSize_1] * sizeof(float));
            std::fill_n(&dstData[dstIdx + params.srcODims[dimsSize_1]], dstDims[dimsSize_1] - params.srcODims[dimsSize_1],
                        srcData[srcIdx + srcDims[dimsSize_1] - 1]);

            parallel_step(dimsSize_1, counters, dstDims);
        }
    });
}

void MKLDNNPadNode::padReflect(const float *srcData, float* dstData, SizeVector srcDims, SizeVector dstDims) {
    SizeVector src_2;
    for (size_t i = 0; i < srcDims.size(); i++)
        src_2.push_back(srcDims[i] + params.srcODims[i] - 2);

    size_t dimsSize_1 = dstDims.size() - 1;
    size_t inputSV = dstDims[dimsSize_1];
    size_t workAmountDst = params.dstStrides[0] * dstDims[0] / dstDims[dimsSize_1];

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector counters(dimsSize_1, 0);
        splitter(workAmountDst, nthr, ithr, start, end);

        parallel_init(start, dimsSize_1, counters, dstDims);
        int dstIdx = 0;
        for (size_t i = 0; i < dimsSize_1; ++i)
            dstIdx += counters[i] * params.dstStrides[i];

        for (size_t iwork = start; iwork < end; ++iwork, dstIdx += inputSV) {
            int srcIdx = 0;
            for (size_t i = 0; i < dimsSize_1; ++i) {
                int idx = (counters[i] < padsBegin[i]) ? (padsBegin[i] - counters[i]) :
                          ((counters[i] >= params.srcODims[i]) ? (src_2[i] - counters[i]) : (counters[i] - padsBegin[i]));
                srcIdx += idx * params.srcStrides[i];
            }

            for (size_t i = 0; i < padsBegin[dimsSize_1]; ++i)
                dstData[dstIdx + i] = srcData[srcIdx + padsBegin[dimsSize_1] - i];

            cpu_memcpy(&dstData[dstIdx + padsBegin[dimsSize_1]], &srcData[srcIdx], sizeof(float) * srcDims[dimsSize_1]);

            for (size_t i = params.srcODims[dimsSize_1]; i < dstDims[dimsSize_1]; ++i)
                dstData[dstIdx + i] = srcData[srcIdx + src_2[dimsSize_1] - i];

            parallel_step(dimsSize_1, counters, dstDims);
        }
    });
}

void MKLDNNPadNode::padSymmetric(const float *srcData, float* dstData, SizeVector srcDims, SizeVector dstDims) {
    SizeVector src_2;
    for (size_t i = 0; i < srcDims.size(); i++)
        src_2.push_back(srcDims[i] + params.srcODims[i] - 1);

    size_t dimsSize_1 = dstDims.size() - 1;
    size_t inputSV = dstDims[dimsSize_1];
    size_t workAmountDst = params.dstStrides[0] * dstDims[0] / dstDims[dimsSize_1];

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector counters(dimsSize_1, 0);
        splitter(workAmountDst, nthr, ithr, start, end);

        parallel_init(start, dimsSize_1, counters, dstDims);
        int dstIdx = 0;
        for (size_t i = 0; i < dimsSize_1; ++i)
            dstIdx += counters[i] * params.dstStrides[i];

        for (size_t iwork = start; iwork < end; ++iwork, dstIdx += inputSV) {
            int srcIdx = 0;
            for (size_t i = 0; i < dimsSize_1; ++i) {
                int idx = (counters[i] < padsBegin[i]) ? (padsBegin[i] - 1 - counters[i]) :
                          ((counters[i] >= params.srcODims[i]) ? (src_2[i] - counters[i]) : (counters[i] - padsBegin[i]));
                srcIdx += idx * params.srcStrides[i];
            }

            for (size_t i = 0; i < padsBegin[dimsSize_1]; ++i)
                dstData[dstIdx + i] = srcData[srcIdx + padsBegin[dimsSize_1] -1 - i];

            cpu_memcpy(&dstData[dstIdx + padsBegin[dimsSize_1]], &srcData[srcIdx],
                       sizeof(float) * srcDims[dimsSize_1]);

            for (size_t i = params.srcODims[dimsSize_1]; i < dstDims[dimsSize_1]; ++i)
                dstData[dstIdx + i] = srcData[srcIdx + src_2[dimsSize_1] - i];

            parallel_step(dimsSize_1, counters, dstDims);
        }
    });
}

bool MKLDNNPadNode::created() const {
    return getType() == Pad;
}
REG_MKLDNN_PRIM_FOR(MKLDNNPadNode, Pad);
