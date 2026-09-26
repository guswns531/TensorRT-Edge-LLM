/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "runtime/kvCacheManager.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/pagedKvTypes.h"
#include <limits>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace rt
{

std::vector<int32_t> KVCacheManager::resolveLayerOwners(Config const& config)
{
    check::check(
        config.numAttentionLayers >= 0 && config.layerConfigs.size() == static_cast<size_t>(config.numAttentionLayers),
        "KV layer schema must match numAttentionLayers.");
    check::check(config.sharingDonors.empty() || config.sharingDonors.size() == config.layerConfigs.size(),
        "KV sharingDonors must be empty or match the logical attention layer count.");
    check::check(
        config.numAttentionLayers == 0 || config.kvCacheType == DataType::kHALF || config.kvCacheType == DataType::kFP8,
        "Unsupported KV cache dtype.");
    for (int32_t layer = 0; layer < config.numAttentionLayers; ++layer)
    {
        auto const& shape = config.layerConfigs[layer];
        check::check(shape.numKVHeads > 0 && shape.headDim > 0, "KV layer dimensions must be positive.");
        if (!config.sharingDonors.empty())
        {
            int32_t const donor = config.sharingDonors[layer];
            check::check(donor >= -1 && donor < config.numAttentionLayers, "KV donor index is out of range.");
            if (donor >= 0)
            {
                auto const& donorShape = config.layerConfigs[donor];
                check::check(shape.numKVHeads == donorShape.numKVHeads && shape.headDim == donorShape.headDim,
                    "KV donor and borrower layouts must match.");
            }
        }
    }
    std::vector<int32_t> owners(static_cast<size_t>(config.numAttentionLayers));
    for (int32_t layer = 0; layer < config.numAttentionLayers; ++layer)
    {
        int32_t owner = layer;
        int32_t hops = 0;
        while (!config.sharingDonors.empty() && config.sharingDonors[owner] >= 0)
        {
            check::check(hops++ < config.numAttentionLayers, "KV donor graph contains a cycle.");
            owner = config.sharingDonors[owner];
        }
        owners[layer] = owner;
    }
    return owners;
}

KVCacheManager::KVCacheManager(Config const& config, cudaStream_t stream)
    : mConfig(config)
    , mLayerOwners(resolveLayerOwners(config))
{
    check::check(mConfig.kvCacheType == nvinfer1::DataType::kHALF || mConfig.kvCacheType == nvinfer1::DataType::kFP8,
        "Unsupported KV cache dtype.");
    check::check(mConfig.numAttentionLayers >= 0, "numAttentionLayers must be non-negative.");
    check::check(mConfig.maxBatchSize > 0, "maxBatchSize must be positive.");
    check::check(mConfig.maxSequenceLength > 0, "maxSequenceLength must be positive.");
    check::check(mConfig.maxSequenceLength <= kMAX_KV_CACHE_CAPACITY,
        "maxSequenceLength exceeds the largest value that remains int32 after page alignment.");
    check::check(static_cast<int32_t>(mConfig.layerConfigs.size()) == mConfig.numAttentionLayers,
        "layerConfigs size must equal numAttentionLayers.");

    // Token capacity of each active-slot K/V view, padded to a whole number of pages.
    mCapPadded = static_cast<int32_t>(
        ((static_cast<int64_t>(mConfig.maxSequenceLength) + kTOKENS_PER_PAGE - 1) / kTOKENS_PER_PAGE)
        * kTOKENS_PER_PAGE);

    // numPages defaults to the active-slot pages. A larger override retains pages across requests.
    int64_t const minimumActivePages = computeMinimumKvPoolPages(mConfig.maxBatchSize, mConfig.maxSequenceLength);
    check::check(minimumActivePages <= kMAX_KV_POOL_PAGES,
        "KVCacheManager: minimum active pages exceed the largest int32-addressable paged-KV pool.");
    check::check(mConfig.numPages == 0 || mConfig.allowPoolUndercommit
            || static_cast<int64_t>(mConfig.numPages) >= minimumActivePages,
        "KVCacheManager: Config::numPages (" + std::to_string(mConfig.numPages)
            + ") must be >= the minimum active pages (" + std::to_string(minimumActivePages) + ") when non-zero.");
    check::check(mConfig.numPages >= 0 && mConfig.numPages <= kMAX_KV_POOL_PAGES,
        "KVCacheManager: Config::numPages exceeds the largest supported paged-KV pool.");
    mNumPages = (mConfig.numPages == 0) ? static_cast<int32_t>(minimumActivePages) : mConfig.numPages;

    // Pure-Mamba / pure-recurrent models legitimately have zero attention layers.
    // Leave mLayerCaches empty and skip uniformity detection.
    if (mConfig.numAttentionLayers == 0)
    {
        mIsUniform = true;
        return;
    }

    size_t const elemSize = rt::utils::getTypeSize(mConfig.kvCacheType);
    char const* kvCacheTypeStr = (mConfig.kvCacheType == nvinfer1::DataType::kHALF) ? "kHALF" : "kFP8";

    // Determine uniformity: check if all layers share the same numKVHeads and headDim.
    mIsUniform = true;
    for (int32_t i = 1; i < mConfig.numAttentionLayers; ++i)
    {
        if (mConfig.layerConfigs[i].numKVHeads != mConfig.layerConfigs[0].numKVHeads
            || mConfig.layerConfigs[i].headDim != mConfig.layerConfigs[0].headDim)
        {
            mIsUniform = false;
            break;
        }
    }

    mLayerCaches.resize(mConfig.numAttentionLayers);
    for (int32_t i = 0; i < mConfig.numAttentionLayers; ++i)
    {
        if (mLayerOwners[i] != i)
        {
            continue;
        }
        KVLayerConfig const& lc = mConfig.layerConfigs[i];
        check::check(lc.numKVHeads > 0, "numKVHeads must be positive for layer " + std::to_string(i) + ".");
        check::check(lc.headDim > 0, "headDim must be positive for layer " + std::to_string(i) + ".");

        size_t layerBytes = elemSize;
        for (int64_t dimension : {int64_t{2}, static_cast<int64_t>(mNumPages), static_cast<int64_t>(kTOKENS_PER_PAGE),
                 static_cast<int64_t>(lc.numKVHeads), static_cast<int64_t>(lc.headDim)})
        {
            check::check(layerBytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max() / dimension),
                "KV pool allocation size overflows int64.");
            layerBytes *= static_cast<size_t>(dimension);
        }
        check::check(mAllocatedBytes <= std::numeric_limits<size_t>::max() - layerBytes,
            "Total KV pool allocation size overflows size_t.");
        mAllocatedBytes += layerBytes;
        mPhysicalOwnerLayers.push_back(i);

        mLayerCaches[i] = rt::Tensor({2, mNumPages, kTOKENS_PER_PAGE, lc.numKVHeads, lc.headDim}, DeviceType::kGPU,
            mConfig.kvCacheType, "KVCacheManager::layer_" + std::to_string(i));
    }

    LOG_DEBUG("KVCacheManager(dtype=%s, layers=%d, owners=%d, uniform=%s) allocated %.2f MB total GPU memory",
        kvCacheTypeStr, mConfig.numAttentionLayers, numPhysicalOwners(), mIsUniform ? "true" : "false",
        static_cast<float>(mAllocatedBytes) / (1024.0f * 1024.0f));
}

KVCacheManager::~KVCacheManager() noexcept {}

KVCacheManager::KVCacheManager(KVCacheManager&& other) noexcept
{
    mConfig = std::move(other.mConfig);
    mLayerCaches = std::move(other.mLayerCaches);
    mLayerOwners = std::move(other.mLayerOwners);
    mPhysicalOwnerLayers = std::move(other.mPhysicalOwnerLayers);
    mAllocatedBytes = std::exchange(other.mAllocatedBytes, 0U);
    mIsUniform = other.mIsUniform;
    mCapPadded = other.mCapPadded;
    mNumPages = other.mNumPages;

    other.mConfig = Config{};
    other.mLayerOwners.clear();
    other.mPhysicalOwnerLayers.clear();
    other.mIsUniform = true;
    other.mCapPadded = 0;
    other.mNumPages = 0;
}

KVCacheManager& KVCacheManager::operator=(KVCacheManager&& other) noexcept
{
    if (this != &other)
    {
        mConfig = std::move(other.mConfig);
        mLayerCaches = std::move(other.mLayerCaches);
        mLayerOwners = std::move(other.mLayerOwners);
        mPhysicalOwnerLayers = std::move(other.mPhysicalOwnerLayers);
        mAllocatedBytes = std::exchange(other.mAllocatedBytes, 0U);
        mIsUniform = other.mIsUniform;
        mCapPadded = other.mCapPadded;
        mNumPages = other.mNumPages;

        other.mConfig = Config{};
        other.mLayerOwners.clear();
        other.mPhysicalOwnerLayers.clear();
        other.mIsUniform = true;
        other.mCapPadded = 0;
        other.mNumPages = 0;
    }
    return *this;
}

rt::Tensor& KVCacheManager::getCombinedKVCache(int32_t attnLayerIdx) noexcept
{
    return mLayerCaches[mLayerOwners[attnLayerIdx]];
}

rt::Tensor const& KVCacheManager::getCombinedKVCache(int32_t attnLayerIdx) const noexcept
{
    return mLayerCaches[mLayerOwners[attnLayerIdx]];
}

std::pair<rt::Tensor, rt::Tensor> KVCacheManager::getSeparateKVCache(int32_t attnLayerIdx) const noexcept
{
    KVLayerConfig const& lc = mConfig.layerConfigs[attnLayerIdx];
    rt::Tensor kView(kPoolPtr(attnLayerIdx), {mConfig.maxBatchSize, mCapPadded, lc.numKVHeads, lc.headDim},
        DeviceType::kGPU, mConfig.kvCacheType);
    rt::Tensor vView(vPoolPtr(attnLayerIdx), {mConfig.maxBatchSize, mCapPadded, lc.numKVHeads, lc.headDim},
        DeviceType::kGPU, mConfig.kvCacheType);
    return {std::move(kView), std::move(vView)};
}

int32_t KVCacheManager::maxCapPadded() const noexcept
{
    return mCapPadded;
}

int32_t KVCacheManager::numPages() const noexcept
{
    return mNumPages;
}

void* KVCacheManager::kPoolPtr(int32_t attnLayerIdx) const noexcept
{
    return const_cast<void*>(getCombinedKVCache(attnLayerIdx).rawPointer());
}

void* KVCacheManager::vPoolPtr(int32_t attnLayerIdx) const noexcept
{
    KVLayerConfig const& lc = mConfig.layerConfigs[attnLayerIdx];
    size_t const elemSize = rt::utils::getTypeSize(mConfig.kvCacheType);
    int64_t const kCacheElems = static_cast<int64_t>(numPages()) * kTOKENS_PER_PAGE * lc.numKVHeads * lc.headDim;
    return static_cast<char*>(kPoolPtr(attnLayerIdx)) + kCacheElems * static_cast<int64_t>(elemSize);
}

KVLayerConfig const& KVCacheManager::getLayerConfig(int32_t attnLayerIdx) const noexcept
{
    return mConfig.layerConfigs[attnLayerIdx];
}

int32_t KVCacheManager::numLayers() const noexcept
{
    return mConfig.numAttentionLayers;
}

bool KVCacheManager::isUniform() const noexcept
{
    return mIsUniform;
}

KVCacheManager::Config const& KVCacheManager::getConfig() const noexcept
{
    return mConfig;
}

int32_t KVCacheManager::physicalOwner(int32_t attnLayerIdx) const noexcept
{
    return mLayerOwners[attnLayerIdx];
}

std::vector<int32_t> const& KVCacheManager::physicalOwnerLayerIndices() const noexcept
{
    return mPhysicalOwnerLayers;
}

int32_t KVCacheManager::numPhysicalOwners() const noexcept
{
    return static_cast<int32_t>(mPhysicalOwnerLayers.size());
}

size_t KVCacheManager::allocatedBytes() const noexcept
{
    return mAllocatedBytes;
}

size_t KVCacheManager::bytesPerPage() const noexcept
{
    return mNumPages > 0 ? mAllocatedBytes / static_cast<size_t>(mNumPages) : 0U;
}

} // namespace rt
} // namespace trt_edgellm
