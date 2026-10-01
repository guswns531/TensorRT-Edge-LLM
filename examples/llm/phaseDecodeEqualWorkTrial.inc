/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

std::ifstream trialInput(trialPath);
nlohmann::json trialConfig;
trialInput >> trialConfig;
int32_t const rows = trialConfig.at("rows");
int32_t const split = trialConfig.at("split");
int32_t const past = trialConfig.at("past_tokens");
int32_t const warmup = trialConfig.at("warmup");
int32_t const iterations = trialConfig.at("iterations");
int32_t const rounds = trialConfig.at("rounds");
std::string const trialMode = trialConfig.value("mode", std::string{"dense_split"});
int32_t const contextStepTokens = trialConfig.value("context_step_tokens", 0);
int32_t const contextBuckets = trialConfig.value("context_buckets", 1);
ELLM_CHECK(trialMode == "dense_split" || trialMode == "row_order", "Equal-work trial mode is invalid");
ELLM_CHECK(rows > 1 && rows <= decodeBatchCapacity && split > 0 && split < rows,
    "Equal-work trial batches exceed decode capacity");
ELLM_CHECK(past > 0 && past < config.maxKVCacheCapacity - 1 && contextStepTokens >= 0 && contextBuckets > 0,
    "Equal-work trial prefix or context sweep is invalid");
ELLM_CHECK(trialMode == "row_order" || contextStepTokens == 0,
    "Variable context lengths require the row_order trial mode");
ELLM_CHECK(trialMode == "row_order" || past <= prefillSequenceCapacity,
    "Dense/split trial prefix exceeds its single-prefill input profile");
ELLM_CHECK(warmup >= 0 && warmup <= 1000 && iterations > 0 && iterations <= 1000 && rounds > 0 && rounds <= 20,
    "Equal-work trial sample budget is invalid");
ELLM_CHECK(!pair->sharedExecutionContext(), "Equal-work trial requires independent contexts");
CUDA_CHECK(cudaStreamSynchronize(setupStream));
rt::EmbeddingPreprocessor trialEmbedding(embedding, config);
rt::Tensor hostTokens({std::max(rows, past)}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "trial_host_ids");
rt::Tensor deviceTokens({std::max(rows, past)}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "trial_ids");
rt::Tensor sampledIds({rows, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "trial_samples");
rt::Tensor hostSamples({rows}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "trial_host_samples");
rt::PhaseKVActiveView trialPrefillKV(config.maxSupportedPrefillBatchSize, ownership, prefillMap, "trial_p");
rt::PhaseKVActiveView trialDecodeKV(decodeBatchCapacity, ownership, decodeMap, "trial_d");
trialDecodeKV.setPersistentDecodeSelectEnabled(true);
trialDecodeKV.setPersistentPageBindingsEnabled(true);
if (config.ropeConfig.type == rt::RopeType::kMRope)
{
    kernel::initializeTextOnlyMRopeCosSin(prefillIO->mropeCosSin.dataPointer<float>(), config.ropeConfig.rotaryTheta,
        config.rotaryDim, config.maxKVCacheCapacity, config.maxSupportedPrefillBatchSize, prefillStream);
    kernel::initializeTextOnlyMRopeCosSin(decodeIO->mropeCosSin.dataPointer<float>(), config.ropeConfig.rotaryTheta,
        config.rotaryDim, config.maxKVCacheCapacity, decodeBatchCapacity, decodeStream);
}
auto stageTrialTokens = [&](std::vector<int32_t> const& ids, rt::PipelineIO& io,
                            rt::Gemma4EmbeddingPreprocessor* ple, bool prefill, cudaStream_t stream) {
    rt::Coords const shape = prefill ? rt::Coords{1, static_cast<int64_t>(ids.size())}
                                    : rt::Coords{static_cast<int64_t>(ids.size()), 1};
    ELLM_CHECK(hostTokens.reshape(shape) && deviceTokens.reshape(shape), "Trial token reshape failed");
    ELLM_CHECK(io.inputsEmbeds.reshape({shape[0], shape[1], config.hiddenSize}),
        "Trial embedding reshape failed");
    std::copy(ids.begin(), ids.end(), hostTokens.dataPointer<int32_t>());
    CUDA_CHECK(cudaMemcpyAsync(deviceTokens.rawPointer(), hostTokens.rawPointer(), ids.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    trialEmbedding.embed(deviceTokens, std::nullopt, std::nullopt, io, stream);
    trialEmbedding.prepareDeepstack(deviceTokens, {}, io, stream);
    if (ple != nullptr)
    {
        ple->embed(deviceTokens, stream);
    }
};
std::vector<int32_t> slots;
std::vector<int32_t> firstIds;
std::vector<int32_t> rowContextTokens;
for (int32_t row{}; row < rows; ++row)
{
    int32_t const slot = ownership.reserve();
    slots.push_back(slot);
    firstIds.push_back(1 + row % 17);
    int32_t const rowPastTokens = past + (row % contextBuckets) * contextStepTokens;
    rowContextTokens.push_back(rowPastTokens);
    ELLM_CHECK(rowPastTokens <= config.maxSupportedInputLength
            && rowPastTokens < config.maxKVCacheCapacity - 1
            && (trialMode != "row_order" || rowPastTokens % ownership.config().tokensPerPage == 0),
        "Equal-work context sweep exceeds the model or page-aligned fixture contract");
    ownership.ensureCapacity(slot, rowPastTokens + 2);
    ownership.setLength(slot, 0);
    int32_t const chunkLimit = config.packedPrefill ? config.maxPackedPrefillChunkTokens : prefillSequenceCapacity;
    for (int32_t offset{}; offset < rowPastTokens;)
    {
        int32_t const chunk = std::min(chunkLimit, rowPastTokens - offset);
        trialPrefillKV.prepare({slot}, prefillStream);
        stageTrialTokens(std::vector<int32_t>(static_cast<size_t>(chunk), firstIds.back()), *prefillIO, prefillPle.get(),
            true, prefillStream);
        trialPrefillKV.preparePrefillMetadata(*prefillIO, {chunk}, prefillStream, config.packedPrefill);
        auto const dims = config.packedPrefill ? config.packedPrefillDims(1, chunk, chunk, rt::ExecutionPhase::kContextChunk)
                                              : config.prefillDims(1, chunk, rt::ExecutionPhase::kContextChunk);
        ELLM_CHECK(pair->prefillExecutor().prepare(0, dims, prefillMap, prefillStream)
                && pair->prefillExecutor().execute(prefillStream),
            "Trial prefix initialization failed");
        CUDA_CHECK(cudaStreamSynchronize(prefillStream));
        trialPrefillKV.commitLengths({offset + chunk});
        trialPrefillKV.complete();
        offset += chunk;
    }
}
auto& trialExecutor = pair->decodeExecutor();
auto prepareTrial = [&](int32_t offset, int32_t count, std::vector<int32_t> const& ids) {
    trialDecodeKV.prepare(std::vector<int32_t>(slots.begin() + offset, slots.begin() + offset + count), decodeStream);
    stageTrialTokens(std::vector<int32_t>(ids.begin() + offset, ids.begin() + offset + count), *decodeIO,
        decodePle.get(), false, decodeStream);
    trialDecodeKV.prepareDecodeMetadata(*decodeIO, decodeStream);
    ELLM_CHECK(trialExecutor.prepare(1, config.decodeDims(count), decodeMap, decodeStream),
        "Trial decode binding failed");
};
std::vector<int32_t> const graphBatches
    = trialMode == "row_order" ? std::vector<int32_t>{rows} : std::vector<int32_t>{rows, split, rows - split};
for (int32_t const batch : graphBatches)
{
    prepareTrial(0, batch, firstIds);
    ELLM_CHECK(trialExecutor.captureGraph(decodeStream), "Trial requires graph coverage for every batch");
    CUDA_CHECK(cudaStreamSynchronize(decodeStream));
    trialDecodeKV.complete();
}
cudaEvent_t trialStart{};
cudaEvent_t trialEnd{};
CUDA_CHECK(cudaEventCreate(&trialStart));
CUDA_CHECK(cudaEventCreate(&trialEnd));
auto dispatchTrial = [&](int32_t offset, int32_t count, std::vector<int32_t> const& ids,
                         std::vector<int32_t>& outputs) {
    prepareTrial(offset, count, ids);
    CUDA_CHECK(cudaEventRecord(trialStart, decodeStream));
    ELLM_CHECK(trialExecutor.execute(decodeStream), "Trial decode execution failed");
    CUDA_CHECK(cudaEventRecord(trialEnd, decodeStream));
    ELLM_CHECK(decodeIO->outputLogits.reshape({count, config.outputVocabSize})
            && sampledIds.reshape({count, 1}) && hostSamples.reshape({count}),
        "Trial sampling reshape failed");
    selectArgmax(decodeIO->outputLogits, sampledIds, decodeStream);
    CUDA_CHECK(cudaMemcpyAsync(hostSamples.rawPointer(), sampledIds.rawPointer(),
        static_cast<size_t>(count) * sizeof(int32_t), cudaMemcpyDeviceToHost, decodeStream));
    // The next dispatch reuses this context, pinned staging buffers, and sampled IDs.
    CUDA_CHECK(cudaStreamSynchronize(decodeStream));
    std::copy_n(hostSamples.dataPointer<int32_t>(), count, outputs.begin() + offset);
    float gpuMs{};
    CUDA_CHECK(cudaEventElapsedTime(&gpuMs, trialStart, trialEnd));
    trialDecodeKV.complete();
    return gpuMs;
};
if (trialMode == "row_order")
{
    auto prepareOrderedTrial = [&](std::vector<int32_t> const& rowOrder, std::vector<int32_t> const& ids) {
        std::vector<int32_t> orderedSlots;
        std::vector<int32_t> orderedIds;
        orderedSlots.reserve(rowOrder.size());
        orderedIds.reserve(rowOrder.size());
        for (int32_t const row : rowOrder)
        {
            ELLM_CHECK(row >= 0 && row < rows, "Row-order trial contains an invalid row");
            orderedSlots.push_back(slots[static_cast<size_t>(row)]);
            orderedIds.push_back(ids[static_cast<size_t>(row)]);
        }
        trialDecodeKV.prepare(orderedSlots, decodeStream);
        stageTrialTokens(orderedIds, *decodeIO, decodePle.get(), false, decodeStream);
        trialDecodeKV.prepareDecodeMetadata(*decodeIO, decodeStream);
        ELLM_CHECK(trialExecutor.prepare(1, config.decodeDims(static_cast<int32_t>(rowOrder.size())), decodeMap,
                       decodeStream),
            "Row-order trial decode binding failed");
    };
    auto dispatchOrderedTrial = [&](std::vector<int32_t> const& rowOrder, std::vector<int32_t> const& ids,
                                    std::vector<int32_t>& outputs) {
        prepareOrderedTrial(rowOrder, ids);
        CUDA_CHECK(cudaEventRecord(trialStart, decodeStream));
        ELLM_CHECK(trialExecutor.execute(decodeStream), "Row-order trial decode execution failed");
        CUDA_CHECK(cudaEventRecord(trialEnd, decodeStream));
        int32_t const count = static_cast<int32_t>(rowOrder.size());
        ELLM_CHECK(decodeIO->outputLogits.reshape({count, config.outputVocabSize})
                && sampledIds.reshape({count, 1}) && hostSamples.reshape({count}),
            "Row-order trial sampling reshape failed");
        selectArgmax(decodeIO->outputLogits, sampledIds, decodeStream);
        CUDA_CHECK(cudaMemcpyAsync(hostSamples.rawPointer(), sampledIds.rawPointer(),
            static_cast<size_t>(count) * sizeof(int32_t), cudaMemcpyDeviceToHost, decodeStream));
        CUDA_CHECK(cudaStreamSynchronize(decodeStream));
        int32_t const* const selected = hostSamples.dataPointer<int32_t>();
        for (int32_t row{}; row < count; ++row)
        {
            outputs[static_cast<size_t>(rowOrder[static_cast<size_t>(row)])] = selected[row];
        }
        float gpuMs{};
        CUDA_CHECK(cudaEventElapsedTime(&gpuMs, trialStart, trialEnd));
        trialDecodeKV.complete();
        return gpuMs;
    };
    std::vector<int32_t> identityOrder(static_cast<size_t>(rows));
    std::vector<int32_t> reversedOrder(static_cast<size_t>(rows));
    for (int32_t row{}; row < rows; ++row)
    {
        identityOrder[static_cast<size_t>(row)] = row;
        reversedOrder[static_cast<size_t>(row)] = rows - row - 1;
    }
    std::array<std::vector<int32_t>, 2> const orders{identityOrder, reversedOrder};
    std::array<char const*, 2> constexpr kORDER_NAMES{"identity", "reversed"};
    nlohmann::json rowOrderMeasurements = nlohmann::json::array();
    size_t rowOrderMismatches{};
    size_t rowOrderComparisons{};
    auto const rowOrderGraphBefore = trialExecutor.graphCacheStats();
    for (int32_t round{}; round < rounds; ++round)
    {
        for (int32_t iteration = -warmup; iteration < iterations; ++iteration)
        {
            std::array<std::vector<int32_t>, 2> outputs;
            for (int32_t order{}; order < 2; ++order)
            {
                int32_t const variant = (round + iteration + warmup + order) % 2;
                outputs[static_cast<size_t>(variant)].resize(static_cast<size_t>(rows));
                for (size_t row{}; row < slots.size(); ++row)
                {
                    ownership.setLength(slots[row], rowContextTokens[row]);
                }
                auto const begin = std::chrono::steady_clock::now();
                float const gpuMs = dispatchOrderedTrial(orders[static_cast<size_t>(variant)], firstIds,
                    outputs[static_cast<size_t>(variant)]);
                auto const end = std::chrono::steady_clock::now();
                if (iteration >= 0)
                {
                    rowOrderMeasurements.push_back({{"round", round}, {"iteration", iteration},
                        {"variant", kORDER_NAMES[variant]}, {"order", order}, {"gpu_ms", gpuMs},
                        {"drain_ms", std::chrono::duration<double, std::milli>(end - begin).count()}});
                }
            }
            if (iteration >= 0)
            {
                for (int32_t row{}; row < rows; ++row)
                {
                    bool const mismatch = outputs[0][static_cast<size_t>(row)] != outputs[1][static_cast<size_t>(row)];
                    rowOrderMismatches += mismatch;
                    rowOrderComparisons += 1U;
                }
            }
        }
    }
    auto const rowOrderGraphAfter = trialExecutor.graphCacheStats();
    ELLM_CHECK(rowOrderGraphAfter.misses == rowOrderGraphBefore.misses
            && rowOrderGraphAfter.launchFailures == rowOrderGraphBefore.launchFailures,
        "Row-order trial unexpectedly fell back from CUDA graph replay");
    CUDA_CHECK(cudaEventDestroy(trialStart));
    CUDA_CHECK(cudaEventDestroy(trialEnd));
    for (int32_t const slot : slots)
    {
        ownership.release(slot);
    }
    nlohmann::json const rowOrderReport{{"config", trialConfig}, {"policy_applied", false},
        {"scope", "fixed_page_aligned_prefix_decode_row_permutation"},
        {"row_context_tokens", rowContextTokens},
        {"graph_hits", rowOrderGraphAfter.hits - rowOrderGraphBefore.hits},
        {"graph_misses", rowOrderGraphAfter.misses - rowOrderGraphBefore.misses},
        {"compared_tokens", rowOrderComparisons}, {"mismatched_tokens", rowOrderMismatches},
        {"remaining_slots", ownership.availableSlots()}, {"samples", std::move(rowOrderMeasurements)}};
    std::ofstream rowOrderOutput(trialConfig.at("output").get<std::string>());
    rowOrderOutput << rowOrderReport.dump(2) << '\n';
    ELLM_CHECK(rowOrderOutput.good(), "Failed to write row-order trial report");
    return EXIT_SUCCESS;
}
nlohmann::json measurements = nlohmann::json::array();
size_t mismatchedTokens{};
size_t comparedTokens{};
auto const graphBefore = trialExecutor.graphCacheStats();
for (int32_t round{}; round < rounds; ++round)
{
    for (int32_t iteration = -warmup; iteration < iterations; ++iteration)
    {
        std::array<std::vector<int32_t>, 2> initialOutputs;
        std::array<std::vector<int32_t>, 2> successorOutputs;
        for (int32_t order{}; order < 2; ++order)
        {
            int32_t const variant = (round + iteration + warmup + order) % 2;
            auto& outputs = initialOutputs[variant];
            auto& nextOutputs = successorOutputs[variant];
            outputs.resize(static_cast<size_t>(rows));
            nextOutputs.resize(static_cast<size_t>(rows));
            // Prefix KV is immutable; replay only overwrites positions past and past+1.
            for (int32_t const slot : slots)
            {
                ownership.setLength(slot, past);
            }
            auto const begin = std::chrono::steady_clock::now();
            float gpuMs{};
            if (variant == 0)
            {
                gpuMs = dispatchTrial(0, rows, firstIds, outputs);
            }
            else
            {
                gpuMs = dispatchTrial(0, split, firstIds, outputs);
                gpuMs += dispatchTrial(split, rows - split, firstIds, outputs);
            }
            auto const drained = std::chrono::steady_clock::now();
            for (int32_t const slot : slots)
            {
                ownership.setLength(slot, past + 1);
            }
            float const nextGpuMs = dispatchTrial(0, rows, outputs, nextOutputs);
            auto const finished = std::chrono::steady_clock::now();
            if (iteration >= 0)
            {
                measurements.push_back({{"round", round}, {"iteration", iteration},
                    {"variant", variant == 0 ? "dense" : "split"}, {"order", order}, {"gpu_ms", gpuMs},
                    {"drain_ms", std::chrono::duration<double, std::milli>(drained - begin).count()},
                    {"successor_gpu_ms", nextGpuMs},
                    {"two_tick_ms", std::chrono::duration<double, std::milli>(finished - begin).count()}});
            }
        }
        if (iteration >= 0)
        {
            for (int32_t row{}; row < rows; ++row)
            {
                mismatchedTokens += initialOutputs[0][row] != initialOutputs[1][row];
                mismatchedTokens += successorOutputs[0][row] != successorOutputs[1][row];
                comparedTokens += 2U;
            }
        }
    }
}
auto const graphAfter = trialExecutor.graphCacheStats();
ELLM_CHECK(graphAfter.misses == graphBefore.misses && graphAfter.launchFailures == graphBefore.launchFailures,
    "Equal-work trial unexpectedly fell back from CUDA graph replay");
CUDA_CHECK(cudaEventDestroy(trialStart));
CUDA_CHECK(cudaEventDestroy(trialEnd));
for (int32_t const slot : slots)
{
    ownership.release(slot);
}
nlohmann::json const trialReport{{"config", trialConfig}, {"policy_applied", false},
    {"scope", "synthetic_equal_prefix_decode_plus_forced_dense_successor"},
    {"graph_hits", graphAfter.hits - graphBefore.hits}, {"graph_misses", graphAfter.misses - graphBefore.misses},
    {"compared_tokens", comparedTokens}, {"mismatched_tokens", mismatchedTokens},
    {"remaining_slots", ownership.availableSlots()}, {"samples", std::move(measurements)}};
std::ofstream trialOutput(trialConfig.at("output").get<std::string>());
trialOutput << trialReport.dump(2) << '\n';
ELLM_CHECK(trialOutput.good(), "Failed to write equal-work trial report");
