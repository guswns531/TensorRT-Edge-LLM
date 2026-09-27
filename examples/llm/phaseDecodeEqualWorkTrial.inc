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
ELLM_CHECK(rows > 1 && rows <= decodeBatchCapacity && split > 0 && split < rows,
    "Equal-work trial batches exceed decode capacity");
ELLM_CHECK(past > 0 && past < config.maxKVCacheCapacity - 1 && past <= prefillSequenceCapacity,
    "Equal-work trial prefix exceeds prefill or KV capacity");
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
for (int32_t row{}; row < rows; ++row)
{
    int32_t const slot = ownership.reserve();
    slots.push_back(slot);
    firstIds.push_back(1 + row % 17);
    ownership.ensureCapacity(slot, past + 2);
    ownership.setLength(slot, 0);
    trialPrefillKV.prepare({slot}, prefillStream);
    stageTrialTokens(std::vector<int32_t>(static_cast<size_t>(past), firstIds.back()), *prefillIO, prefillPle.get(),
        true, prefillStream);
    trialPrefillKV.preparePrefillMetadata(*prefillIO, {past}, prefillStream, config.packedPrefill);
    auto const dims = config.packedPrefill ? config.packedPrefillDims(1, past, past)
                                         : config.prefillDims(1, past, false);
    ELLM_CHECK(pair->prefillExecutor().prepare(0, dims, prefillMap, prefillStream)
            && pair->prefillExecutor().execute(prefillStream),
        "Trial prefix initialization failed");
    CUDA_CHECK(cudaStreamSynchronize(prefillStream));
    trialPrefillKV.commitLengths({past});
    trialPrefillKV.complete();
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
for (int32_t const batch : std::vector<int32_t>{rows, split, rows - split})
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
