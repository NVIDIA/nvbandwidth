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

#include <cuda.h>
#include <algorithm>
#include <memory>
#include <vector>
#include <cuda_runtime_api.h>

#include "common.h"
#include "output.h"
#include "testcase.h"
#include "test_settings.h"
#include "memcpy.h"
#include "memcpy_le.h"

// ============================================================================
// HostDeviceTransfer
// ============================================================================

HostDeviceTransfer::HostDeviceTransfer(CopyInitiator initiator, Direction direction)
    : Testcase(generateKey(initiator, direction), generateDesc(initiator, direction))
    , initiator_(initiator)
    , directionStrategy_(createDirectionStrategy(direction)) {}

void HostDeviceTransfer::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer(size, deviceId);
        DeviceBuffer deviceBuffer(size, deviceId);

        auto [src, dst] = directionStrategy_->getOrderedBuffers(hostBuffer, deviceBuffer);
        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(*src, *dst);
    }

    recordStability(bandwidthValues, memcpyInstance);
    output->addTestcaseResults(bandwidthValues, buildOutputLabel());
}

FilterResult HostDeviceTransfer::filter() {
    // Waive SM host<->device tests for bounce-buffer CC since SM accesses to
    // sysmem in bounce-buffer CC modes result in page faults and migration of
    // the accessed pages to vidmem.
    if (initiator_ == CopyInitiator::CE) {
        return FilterResult::pass();
    }
    return filterBounceBufferConfComputeEnabled();
}

MemcpyInitiator* HostDeviceTransfer::createMemcpyInitiator() const {
    switch (initiator_) {
        case CopyInitiator::CE:  return new MemcpyInitiatorCE();
        case CopyInitiator::SM:  return new MemcpyInitiatorSM();
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for host-device transfers");
            return nullptr;
        default:
            ASSERT(false && "Unknown CopyInitiator");
            return nullptr;
    }
}

std::unique_ptr<HostDeviceTransferDirectionStrategy> HostDeviceTransfer::createDirectionStrategy(Direction dir) {
    switch (dir) {
        case Direction::HostToDevice: return std::make_unique<HostToDeviceDirection>();
        case Direction::DeviceToHost: return std::make_unique<DeviceToHostDirection>();
        default:
            ASSERT(false && "Unknown Direction");
            return nullptr;
    }
}

std::string HostDeviceTransfer::generateKey(CopyInitiator initiator, Direction direction) {
    std::string result = (direction == Direction::HostToDevice)
                         ? "host_to_device" : "device_to_host";
    result += "_memcpy_";
    switch (initiator) {
        case CopyInitiator::CE:  result += "ce"; break;
        case CopyInitiator::SM:  result += "sm"; break;
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for host-device transfers");
            break;
        default:
            ASSERT(false && "Unknown CopyInitiator");
            break;
    }
    return result;
}

// ============================================================================
// HostDeviceBidirTransfer
// ============================================================================

HostDeviceBidirTransfer::HostDeviceBidirTransfer(CopyInitiator initiator, Direction direction)
    : Testcase(generateKey(initiator, direction), generateDesc(initiator, direction))
    , initiator_(initiator)
    , direction_(direction) {}

MemcpyInitiator* HostDeviceBidirTransfer::createMemcpyInitiator() const {
    switch (initiator_) {
        case CopyInitiator::CE:  return new MemcpyInitiatorCE();
        case CopyInitiator::SM:  return new MemcpyInitiatorSMSplitWarp();
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for host-device bidirectional transfers");
            return nullptr;
        default:
            ASSERT(false && "Unknown CopyInitiator");
            break;
    }
    return nullptr;
}

std::string HostDeviceBidirTransfer::generateKey(CopyInitiator initiator, Direction direction) {
    std::string result = (direction == Direction::HostToDevice)
                         ? "host_to_device" : "device_to_host";
    result += "_bidirectional_memcpy_";
    switch (initiator) {
        case CopyInitiator::CE:  result += "ce"; break;
        case CopyInitiator::SM:  result += "sm"; break;
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for host-device bidirectional transfers");
            return nullptr;
        default:
            ASSERT(false && "Unknown CopyInitiator");
            break;
    }
    return result;
}

std::string HostDeviceTransfer::generateDesc(CopyInitiator initiator, Direction direction) {
    std::string result = "\t";
    result += (direction == Direction::HostToDevice) ? "Host to device" : "Device to host";
    switch (initiator) {
        case CopyInitiator::CE:  result += " CE memcpy using cuMemcpyAsync"; break;
        case CopyInitiator::SM:  result += " SM memcpy using a copy kernel"; break;
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for host-device transfers");
            break;
        default:
            ASSERT(false && "Unknown CopyInitiator");
            break;
    }
    return result;
}

std::string HostDeviceTransfer::buildOutputLabel() const {
    std::string initiatorName;
    switch (initiator_) {
        case CopyInitiator::CE:  initiatorName = "CE"; break;
        case CopyInitiator::SM:  initiatorName = "SM"; break;
        case CopyInitiator::TMA: initiatorName = "TMA"; break;
        default:
            ASSERT(false && "Unknown CopyInitiator");
            break;
    }
    return "memcpy " + initiatorName + " CPU(row) " +
           directionStrategy_->getDirectionArrow() + " GPU(column) bandwidth (GB/s)";
}

std::string HostDeviceBidirTransfer::generateDesc(CopyInitiator /*initiator*/, Direction direction) {
    std::string measured = (direction == Direction::HostToDevice)
                           ? "host to device" : "device to host";
    std::string interference = (direction == Direction::HostToDevice)
                               ? "device to host" : "host to device";
    std::string result = "\tA " + measured + " copy is measured while a " + interference +
                         " copy is run simultaneously.\n\tOnly the " + measured +
                         " copy bandwidth is reported.";
    return result;
}

std::string HostDeviceBidirTransfer::buildOutputLabel() const {
    std::string initiatorName;
    switch (initiator_) {
        case CopyInitiator::CE: initiatorName = "CE"; break;
        case CopyInitiator::SM: initiatorName = "SM"; break;
        default:
            ASSERT(false && "Unknown CopyInitiator");
            break;
    }
    return "memcpy " + initiatorName + " CPU(row) <-> GPU(column) bandwidth (GB/s)";
}

void HostDeviceBidirTransfer::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator());

    runWithCache(bandwidthValues, [&]() {
        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            if (initiator_ == CopyInitiator::CE) {
                // Double the size of the interference copy to ensure it interferes correctly
                HostBuffer host1(size, deviceId), host2(size * 2, deviceId);
                DeviceBuffer dev1(size, deviceId), dev2(size * 2, deviceId);

                std::vector<const MemcpyBuffer*> srcBuffers, dstBuffers;
                if (direction_ == Direction::HostToDevice) {
                    srcBuffers = {&host1, &dev2};
                    dstBuffers = {&dev1, &host2};
                } else {
                    srcBuffers = {&dev1, &host2};
                    dstBuffers = {&host1, &dev2};
                }
                bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcBuffers, dstBuffers);
            } else {
                HostBuffer hostBuffer(size, deviceId);
                DeviceBuffer deviceBuffer(size, deviceId);
                bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(hostBuffer, deviceBuffer);
            }
        }
    });

    recordStability(bandwidthValues, memcpyInstance);
    output->addTestcaseResults(bandwidthValues, buildOutputLabel());
}

// NVLink SM scheduling support cache (mirrors logic in testcases_sm.cpp)
namespace {
std::vector<bool> nvlinkSchedulingSupportCache;
bool cacheInitialized = false;

void initializeNvlinkSchedulingCache(int deviceCount) {
    if (cacheInitialized) {
        return;
    }

    nvlinkSchedulingSupportCache.resize(deviceCount, false);

#if CUDART_VERSION >= 13000
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, deviceId) == cudaSuccess) {
            nvlinkSchedulingSupportCache[deviceId] = (prop.major >= 10);
        }
    }
#endif
    cacheInitialized = true;
}

inline bool devicePairSupportsNvlinkScheduling(int deviceId1, int deviceId2) {
    return nvlinkSchedulingSupportCache[deviceId1] &&
           nvlinkSchedulingSupportCache[deviceId2];
}

inline bool allDevicesSupportNvlinkScheduling(int deviceCount) {
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        if (!nvlinkSchedulingSupportCache[deviceId]) {
            return false;
        }
    }
    return true;
}
}  // namespace

// ============================================================================
// DeviceToDeviceTransfer
// ============================================================================

// Shared factory for the access-type strategy. Defined ahead of the transfer
// constructors below so their initializer lists can call it directly.
std::unique_ptr<DeviceToDeviceAccessStrategy> createDeviceToDeviceAccessStrategy(
        AccessType accessType) {
    switch (accessType) {
        case AccessType::Read:  return std::make_unique<ReadAccessStrategy>();
        case AccessType::Write: return std::make_unique<WriteAccessStrategy>();
    }
    ASSERT(false && "Unknown AccessType in createDeviceToDeviceAccessStrategy");
    return nullptr;
}

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
static bool runLocalizedConcurrentCeD2D(const std::string &key, int srcDeviceId, int dstDeviceId,
                                        unsigned long long size, unsigned long long loopCount,
                                        const DeviceToDeviceAccessStrategy &accessStrategy,
                                        bool bidirectional,
                                        std::vector<double> &bandwidthPerDirection);
#endif

DeviceToDeviceTransfer::DeviceToDeviceTransfer(CopyInitiator initiator, AccessType accessType)
    : Testcase(generateKey(initiator, accessType), generateDesc(initiator, accessType))
    , initiator_(initiator)
    , accessType_(accessType)
    , accessStrategy_(createDeviceToDeviceAccessStrategy(accessType)) {}

FilterResult DeviceToDeviceTransfer::filter() {
    auto r = Testcase::filterHasAccessiblePeerPairs();
    if (!r) return r;
    if (initiator_ == CopyInitiator::TMA) {
        auto tma = Testcase::filterSupportsTMA();
        if (!tma) return tma;
    }
    if (initiatorUsesHandleBuffers(initiator_))
        return Testcase::filterSupportsHandleCopy();
    return FilterResult::pass();
}

std::unique_ptr<BufferTypeStrategy> DeviceToDeviceTransfer::createBufferStrategy() const {
    if (initiatorUsesHandleBuffers(initiator_)) {
        return std::make_unique<LEBufferStrategy>();
    }
    return std::make_unique<DeviceBufferStrategy>();
}

MemcpyInitiator* DeviceToDeviceTransfer::createMemcpyInitiator() const {
    switch (initiator_) {
        case CopyInitiator::CE:  return new MemcpyInitiatorCE();
        case CopyInitiator::SM:  return new MemcpyInitiatorSM();
        case CopyInitiator::TMA: return new MemcpyInitiatorTMA();
    }
    return nullptr;
}

std::string DeviceToDeviceTransfer::generateKey(CopyInitiator initiator, AccessType accessType) {
    std::string result = "device_to_device_memcpy_";
    result += (accessType == AccessType::Read) ? "read_" : "write_";
    switch (initiator) {
        case CopyInitiator::CE:  result += "ce"; break;
        case CopyInitiator::SM:  result += "sm"; break;
        case CopyInitiator::TMA: result += "tma"; break;
    }
    return result;
}

// Only CE localizes by default, so the other initiators must not carry this note.
static std::string multiDomainD2DNote() {
#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
    return "\n\tOn architectures that localize, one buffer pair is pinned to each locality domain "
           "and all domains are copied concurrently, so the reported figure is the summed "
           "bandwidth of every domain; otherwise a single non-localized copy is measured.";
#else
    return "";
#endif
}

std::string DeviceToDeviceTransfer::generateDesc(CopyInitiator initiator, AccessType accessType) {
    std::string result = "\tMeasures bandwidth of ";
    switch (initiator) {
        case CopyInitiator::CE:  result += "CE copies (cuMemcpyAsync)"; break;
        case CopyInitiator::SM:  result += "a copy kernel"; break;
        case CopyInitiator::TMA: result += "TMA copies"; break;
    }
    result += " between each pair of accessible peers.\n\t";
    if (accessType == AccessType::Read) {
        result += "Read tests launch a copy from the peer device to the target using the target's context.";
    } else {
        result += "Write tests launch a copy from the target device to the peer using the target's context.";
    }
    if (initiator == CopyInitiator::TMA) {
        result += std::string("\n\t") + handlePeerBufferConventionNote();
    }
    if (initiator == CopyInitiator::CE) {
        result += multiDomainD2DNote();
    }
    return result;
}

std::string DeviceToDeviceTransfer::buildOutputLabel() const {
    std::string initiatorName;
    switch (initiator_) {
        case CopyInitiator::CE:  initiatorName = "CE"; break;
        case CopyInitiator::SM:  initiatorName = "SM"; break;
        case CopyInitiator::TMA: initiatorName = "TMA"; break;
    }
    std::string arrow = (accessType_ == AccessType::Read) ? "->" : "<-";
    return "memcpy " + initiatorName + " GPU(row) " + arrow + " GPU(column) bandwidth (GB/s)";
}

void DeviceToDeviceTransfer::run(unsigned long long size, unsigned long long loopCount) {
    bufferStrategy_ = createBufferStrategy();
    if (!bufferStrategy_->isSupported()) {
        output->setTestcaseStatusAndAddIfNeeded(key, NVB_WAIVED, bufferStrategy_->getLabel() + " buffer not supported");
        return;
    }

    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, testKey());

    MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator(),
                                   accessStrategy_->getContextPreference());

    if (initiator_ == CopyInitiator::SM) {
        initializeNvlinkSchedulingCache(deviceCount);
    }

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }
            if (!peerIsGpuPairEndpoint(srcDeviceId) || !peerIsGpuPairEndpoint(peerDeviceId)) {
                continue;
            }

            if (initiator_ == CopyInitiator::SM) {
                bool enableNvlinkScheduling = !gSettings.disableNvlinkSMScheduling &&
                                             devicePairSupportsNvlinkScheduling(srcDeviceId, peerDeviceId);
                memcpyInstance.setUseNvlinkUtilScheduling(enableNvlinkScheduling);
            }

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
            if (initiator_ == CopyInitiator::CE) {
                std::vector<double> localizedBw;
                if (runLocalizedConcurrentCeD2D(testKey(), srcDeviceId, peerDeviceId, size,
                                                loopCount, *accessStrategy_,
                                                /*bidirectional=*/false, localizedBw)) {
                    bandwidthValues.value(srcDeviceId, peerDeviceId) = localizedBw[0];
                    continue;
                }
            }
#endif

            auto bufferPair = bufferStrategy_->createBufferPair(size, srcDeviceId, peerDeviceId);

            if (!bufferStrategy_->enablePeerAccess(*bufferPair)) {
                continue;
            }

            auto [src, dst] = accessStrategy_->getOrderedBuffers(*bufferPair->src, *bufferPair->peer);
            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(*src, *dst);
        }
    }

    recordStability(bandwidthValues, memcpyInstance);
    output->addTestcaseResults(bandwidthValues, buildOutputLabel());
}

// ============================================================================
// DeviceToDeviceBidirTransfer
// ============================================================================

DeviceToDeviceBidirTransfer::DeviceToDeviceBidirTransfer(CopyInitiator initiator, AccessType accessType)
    : Testcase(generateKey(initiator, accessType), generateDesc(initiator, accessType))
    , initiator_(initiator)
    , accessType_(accessType)
    , accessStrategy_(createDeviceToDeviceAccessStrategy(accessType)) {}

FilterResult DeviceToDeviceBidirTransfer::filter() {
    auto r = Testcase::filterHasAccessiblePeerPairs();
    if (!r) return r;
    if (initiator_ == CopyInitiator::TMA) {
        auto tma = Testcase::filterSupportsTMA();
        if (!tma) return tma;
    }
    if (initiatorUsesHandleBuffers(initiator_))
        return Testcase::filterSupportsHandleCopy();
    return FilterResult::pass();
}

std::unique_ptr<BufferTypeStrategy> DeviceToDeviceBidirTransfer::createBufferStrategy() const {
    if (initiatorUsesHandleBuffers(initiator_)) {
        return std::make_unique<LEBufferStrategy>();
    }
    return std::make_unique<DeviceBufferStrategy>();
}

MemcpyInitiator* DeviceToDeviceBidirTransfer::createMemcpyInitiator() const {
    switch (initiator_) {
        case CopyInitiator::CE:  return new MemcpyInitiatorCE();
        case CopyInitiator::SM:  return new MemcpyInitiatorSM();
        case CopyInitiator::TMA: return new MemcpyInitiatorTMA();
    }
    return nullptr;
}

std::string DeviceToDeviceBidirTransfer::generateKey(CopyInitiator initiator, AccessType accessType) {
    std::string result = "device_to_device_bidirectional_memcpy_";
    result += (accessType == AccessType::Read) ? "read_" : "write_";
    switch (initiator) {
        case CopyInitiator::CE:  result += "ce"; break;
        case CopyInitiator::SM:  result += "sm"; break;
        case CopyInitiator::TMA: result += "tma"; break;
    }
    return result;
}

std::string DeviceToDeviceBidirTransfer::generateDesc(CopyInitiator initiator, AccessType accessType) {
    std::string result;
    if (initiator == CopyInitiator::CE) {
        result = "\tMeasures bandwidth of CE copies (cuMemcpyAsync)\n"
                 "\tbetween each pair of accessible peers.\n"
                 "\tA copy in the opposite direction of the measured copy is run simultaneously but not measured.\n\t";
    } else if (initiator == CopyInitiator::SM) {
        result = "\tMeasures bandwidth of a copy kernel between each pair of accessible peers. Copies are run\n"
                 "\tin both directions between each pair, and the sum is reported.\n\t";
    } else if (initiator == CopyInitiator::TMA) {
        result = "\tMeasures bandwidth of TMA bidirectional copies between each pair of accessible peers.\n\t";
    } else {
        result = "\tMeasures bidirectional bandwidth between each pair of accessible peers.\n\t";
    }
    result += (accessType == AccessType::Read)
        ? "Read tests launch a copy from the peer device to the target using the target's context."
        : "Write tests launch a copy from the target device to the peer using the target's context.";
    if (initiator == CopyInitiator::TMA) {
        result += std::string("\n\t") + handlePeerBufferConventionNote();
    }
    if (initiator == CopyInitiator::CE) {
        result += multiDomainD2DNote();
    }
    return result;
}

std::string DeviceToDeviceBidirTransfer::buildOutputLabel(const std::string& suffix) const {
    std::string initiatorName;
    switch (initiator_) {
        case CopyInitiator::CE:  initiatorName = "CE"; break;
        case CopyInitiator::SM:  initiatorName = "SM"; break;
        case CopyInitiator::TMA: initiatorName = "TMA"; break;
    }
    return "memcpy " + initiatorName + " GPU(row) <-> GPU(column) " + suffix + " bandwidth (GB/s)";
}

void DeviceToDeviceBidirTransfer::run(unsigned long long size, unsigned long long loopCount) {
    bufferStrategy_ = createBufferStrategy();
    if (!bufferStrategy_->isSupported()) {
        output->setTestcaseStatusAndAddIfNeeded(key, NVB_WAIVED, bufferStrategy_->getLabel() + " buffer not supported");
        return;
    }

    std::string suffix = (accessType_ == AccessType::Read) ? "read" : "write";
    PeerValueMatrix<double> bandwidthValues1(deviceCount, deviceCount, key + "_" + suffix + "1");
    PeerValueMatrix<double> bandwidthValues2(deviceCount, deviceCount, key + "_" + suffix + "2");
    PeerValueMatrix<double> bandwidthValuesTotal(deviceCount, deviceCount, key + "_total");

    // For TMA with --force-le-variant (LE buffers):
    // - Read: kernel reads from remote LE, writes to local device memory -> PREFER_DST_CONTEXT
    // - Write: kernel reads from local device memory, writes to remote LE -> PREFER_SRC_CONTEXT
    MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator(),
                                   accessStrategy_->getContextPreference(),
                                   MemcpyOperation::VECTOR_BW);

    if (initiator_ == CopyInitiator::SM) {
        initializeNvlinkSchedulingCache(deviceCount);
    }

    std::vector<std::pair<int, int>> pairs;
    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId != srcDeviceId && peerIsGpuPairEndpoint(srcDeviceId) &&
                peerIsGpuPairEndpoint(peerDeviceId)) {
                pairs.push_back({srcDeviceId, peerDeviceId});
            }
        }
    }

    cachedPairLoop(pairs, bandwidthValues1, bandwidthValues2, bandwidthValuesTotal,
        [&](int srcDeviceId, int peerDeviceId) -> std::vector<double> {
            if (initiator_ == CopyInitiator::SM) {
                bool enableNvlinkScheduling = !gSettings.disableNvlinkSMScheduling &&
                                             devicePairSupportsNvlinkScheduling(srcDeviceId, peerDeviceId);
                memcpyInstance.setUseNvlinkUtilScheduling(enableNvlinkScheduling);
            }

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
            if (initiator_ == CopyInitiator::CE) {
                std::vector<double> localizedBw;
                if (runLocalizedConcurrentCeD2D(key, srcDeviceId, peerDeviceId, size, loopCount,
                                                *accessStrategy_, /*bidirectional=*/true,
                                                localizedBw)) {
                    return localizedBw;
                }
            }
#endif

            // pair1: buffers on (src, peer). pair2 reverses IDs so LEBufferStrategy
            // (LE always on its peer argument) places one LE on each device; for
            // DeviceBufferStrategy both sides stay plain device buffers either way.
            auto pair1 = bufferStrategy_->createBufferPair(size, srcDeviceId, peerDeviceId);
            auto pair2 = bufferStrategy_->createBufferPair(size, peerDeviceId, srcDeviceId);

            if (!bufferStrategy_->enablePeerAccess(*pair1) ||
                !bufferStrategy_->enablePeerAccess(*pair2)) {
                return {0.0, 0.0};
            }

            // With reversed pair2, peer->src is B->A and A->B (LE->device / try_get).
            // Write uses the opposite order so LE is the destination (device->LE / try_put).
            std::vector<const MemcpyBuffer*> srcBuffers;
            std::vector<const MemcpyBuffer*> peerBuffers;
            if (accessType_ == AccessType::Write) {
                srcBuffers = {pair1->src.get(), pair2->src.get()};
                peerBuffers = {pair1->peer.get(), pair2->peer.get()};
            } else {
                srcBuffers = {pair1->peer.get(), pair2->peer.get()};
                peerBuffers = {pair1->src.get(), pair2->src.get()};
            }

            return memcpyInstance.doMemcpyVector(srcBuffers, peerBuffers);
        });

    std::string suffix1 = (accessType_ == AccessType::Read) ? "Read1" : "Write1";
    std::string suffix2 = (accessType_ == AccessType::Read) ? "Read2" : "Write2";
    recordStability(bandwidthValuesTotal, memcpyInstance);
    output->addTestcaseResults(bandwidthValues1, buildOutputLabel(suffix1));
    output->addTestcaseResults(bandwidthValues2, buildOutputLabel(suffix2));
    output->addTestcaseResults(bandwidthValuesTotal, buildOutputLabel("Total"));
}

// ============================================================================
// AllToHostTransfer
// ============================================================================

AllToHostTransfer::AllToHostTransfer(CopyInitiator initiator)
    : Testcase(generateKey(initiator), generateDesc(initiator))
    , initiator_(initiator) {}

MemcpyInitiator* AllToHostTransfer::createMemcpyInitiator() const {
    switch (initiator_) {
        case CopyInitiator::CE:  return new MemcpyInitiatorCE();
        case CopyInitiator::SM:  return new MemcpyInitiatorSM();
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for all-to-host transfers");
            return nullptr;
    }
    return nullptr;
}

std::string AllToHostTransfer::generateKey(CopyInitiator initiator) {
    std::string result = "all_to_host_memcpy_";
    switch (initiator) {
        case CopyInitiator::CE:  result += "ce"; break;
        case CopyInitiator::SM:  result += "sm"; break;
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for all-to-host transfers");
            break;
    }
    return result;
}

std::string AllToHostTransfer::generateDesc(CopyInitiator initiator) {
    std::string initiatorDesc = (initiator == CopyInitiator::CE)
        ? "cuMemcpyAsync" : "a copy kernel";
    return "\tMeasures bandwidth of " + initiatorDesc +
           " between a single device and the host while simultaneously\n"
           "\trunning copies from all other devices to the host.";
}

std::string AllToHostTransfer::buildOutputLabel() const {
    std::string initiatorName = (initiator_ == CopyInitiator::CE) ? "CE" : "SM";
    return "memcpy " + initiatorName + " CPU(row) <- GPU(column) bandwidth (GB/s)";
}

void AllToHostTransfer::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator(),
                                   PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    allHostHelper(size, memcpyInstance, bandwidthValues, false);

    output->addTestcaseResults(bandwidthValues, buildOutputLabel());
}

// ============================================================================
// HostToAllTransfer
// ============================================================================

HostToAllTransfer::HostToAllTransfer(CopyInitiator initiator)
    : Testcase(generateKey(initiator), generateDesc(initiator))
    , initiator_(initiator) {}

MemcpyInitiator* HostToAllTransfer::createMemcpyInitiator() const {
    switch (initiator_) {
        case CopyInitiator::CE:  return new MemcpyInitiatorCE();
        case CopyInitiator::SM:  return new MemcpyInitiatorSM();
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for host-to-all transfers");
            return nullptr;
    }
    return nullptr;
}

std::string HostToAllTransfer::generateKey(CopyInitiator initiator) {
    std::string result = "host_to_all_memcpy_";
    switch (initiator) {
        case CopyInitiator::CE:  result += "ce"; break;
        case CopyInitiator::SM:  result += "sm"; break;
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for host-to-all transfers");
            break;
    }
    return result;
}

std::string HostToAllTransfer::generateDesc(CopyInitiator initiator) {
    std::string initiatorDesc = (initiator == CopyInitiator::CE)
        ? "cuMemcpyAsync" : "a copy kernel";
    return "\tMeasures bandwidth of " + initiatorDesc +
           " between the host to a single device while simultaneously\n"
           "\trunning copies from the host to all other devices.";
}

std::string HostToAllTransfer::buildOutputLabel() const {
    std::string initiatorName = (initiator_ == CopyInitiator::CE) ? "CE" : "SM";
    return "memcpy " + initiatorName + " CPU(row) -> GPU(column) bandwidth (GB/s)";
}

void HostToAllTransfer::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator(),
                                   PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    allHostHelper(size, memcpyInstance, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues, buildOutputLabel());
}

// ============================================================================
// AllToHostBidirTransfer
// ============================================================================

AllToHostBidirTransfer::AllToHostBidirTransfer(CopyInitiator initiator)
    : Testcase(generateKey(initiator), generateDesc(initiator))
    , initiator_(initiator) {}

MemcpyInitiator* AllToHostBidirTransfer::createMemcpyInitiator() const {
    switch (initiator_) {
        case CopyInitiator::CE:  return new MemcpyInitiatorCE();
        case CopyInitiator::SM:  return new MemcpyInitiatorSMSplitWarp();
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for all-to-host bidirectional transfers");
            return nullptr;
    }
    return nullptr;
}

std::string AllToHostBidirTransfer::generateKey(CopyInitiator initiator) {
    std::string result = "all_to_host_bidirectional_memcpy_";
    switch (initiator) {
        case CopyInitiator::CE:  result += "ce"; break;
        case CopyInitiator::SM:  result += "sm"; break;
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for all-to-host bidirectional transfers");
            break;
    }
    return result;
}

std::string AllToHostBidirTransfer::generateDesc(CopyInitiator initiator) {
    std::string initiatorDesc = (initiator == CopyInitiator::CE)
        ? "copy" : "bandwidth of a copy kernel";
    std::string suffix = (initiator == CopyInitiator::SM)
        ? " using copy kernels" : "";
    return "\tA device to host " + initiatorDesc + " is measured while a host to device copy is run simultaneously.\n"
           "\tOnly the device to host copy bandwidth is reported.\n"
           "\tAll other devices generate simultaneous host to device and device to host interfering traffic" + suffix + ".";
}

std::string AllToHostBidirTransfer::buildOutputLabel() const {
    std::string initiatorName = (initiator_ == CopyInitiator::CE) ? "CE" : "SM";
    return "memcpy " + initiatorName + " CPU(row) <-> GPU(column) bandwidth (GB/s)";
}

void AllToHostBidirTransfer::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);

    if (initiator_ == CopyInitiator::CE) {
        MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator());
        runWithCache(bandwidthValues, [&]() {
            allHostBidirHelper(size, memcpyInstance, bandwidthValues, false);
        });
    } else {
        // SM uses explicit context preference and bandwidth value selection
        MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator(),
            PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

        runWithCache(bandwidthValues, [&]() {
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                std::vector<const MemcpyBuffer*> srcBuffers;
                std::vector<const MemcpyBuffer*> dstBuffers;

                srcBuffers.push_back(new DeviceBuffer(size, deviceId));
                dstBuffers.push_back(new HostBuffer(size, deviceId));

                for (int interferenceDeviceId = 0; interferenceDeviceId < deviceCount; interferenceDeviceId++) {
                    if (interferenceDeviceId == deviceId) continue;

                    srcBuffers.push_back(new DeviceBuffer(size, interferenceDeviceId));
                    dstBuffers.push_back(new HostBuffer(size, interferenceDeviceId));
                }

                bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcBuffers, dstBuffers);

                for (auto node : srcBuffers) delete node;
                for (auto node : dstBuffers) delete node;
            }
            // Must be inside the executor: runWithCache snapshots the matrix
            // for the complementary test once this returns.
            recordStability(bandwidthValues, memcpyInstance);
        });
    }

    output->addTestcaseResults(bandwidthValues, buildOutputLabel());
}

// ============================================================================
// HostToAllBidirTransfer
// ============================================================================

HostToAllBidirTransfer::HostToAllBidirTransfer(CopyInitiator initiator)
    : Testcase(generateKey(initiator), generateDesc(initiator))
    , initiator_(initiator) {}

MemcpyInitiator* HostToAllBidirTransfer::createMemcpyInitiator() const {
    switch (initiator_) {
        case CopyInitiator::CE:  return new MemcpyInitiatorCE();
        case CopyInitiator::SM:  return new MemcpyInitiatorSMSplitWarp();
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for host-to-all bidirectional transfers");
            return nullptr;
    }
    return nullptr;
}

std::string HostToAllBidirTransfer::generateKey(CopyInitiator initiator) {
    std::string result = "host_to_all_bidirectional_memcpy_";
    switch (initiator) {
        case CopyInitiator::CE:  result += "ce"; break;
        case CopyInitiator::SM:  result += "sm"; break;
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for host-to-all bidirectional transfers");
            break;
    }
    return result;
}

std::string HostToAllBidirTransfer::generateDesc(CopyInitiator initiator) {
    if (initiator == CopyInitiator::CE) {
        return "\tA host to device copy is measured while a device to host copy is run simultaneously.\n"
               "\tOnly the host to device copy bandwidth is reported.\n"
               "\tAll other devices generate simultaneous host to device and device to host interfering traffic.";
    } else {
        return "\tA host to device bandwidth of a copy kernel is measured while a device to host copy is run simultaneously.\n"
               "\tOnly the host to device copy bandwidth is reported.\n"
               "\tAll other devices generate simultaneous host to device and device to host interfering traffic using copy kernels.";
    }
}

std::string HostToAllBidirTransfer::buildOutputLabel() const {
    std::string initiatorName = (initiator_ == CopyInitiator::CE) ? "CE" : "SM";
    return "memcpy " + initiatorName + " CPU(row) <-> GPU(column) bandwidth (GB/s)";
}

void HostToAllBidirTransfer::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);

    if (initiator_ == CopyInitiator::CE) {
        MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator());
        runWithCache(bandwidthValues, [&]() {
            allHostBidirHelper(size, memcpyInstance, bandwidthValues, true);
        });
    } else {
        MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator(),
            PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

        runWithCache(bandwidthValues, [&]() {
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                std::vector<const MemcpyBuffer*> srcBuffers;
                std::vector<const MemcpyBuffer*> dstBuffers;

                srcBuffers.push_back(new HostBuffer(size, deviceId));
                dstBuffers.push_back(new DeviceBuffer(size, deviceId));

                for (int interferenceDeviceId = 0; interferenceDeviceId < deviceCount; interferenceDeviceId++) {
                    if (interferenceDeviceId == deviceId) continue;

                    srcBuffers.push_back(new DeviceBuffer(size, interferenceDeviceId));
                    dstBuffers.push_back(new HostBuffer(size, interferenceDeviceId));
                }

                bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcBuffers, dstBuffers);

                for (auto node : srcBuffers) delete node;
                for (auto node : dstBuffers) delete node;
            }
            // Must be inside the executor: runWithCache snapshots the matrix
            // for the complementary test once this returns.
            recordStability(bandwidthValues, memcpyInstance);
        });
    }

    output->addTestcaseResults(bandwidthValues, buildOutputLabel());
}

// ============================================================================
// AllToOneTransfer
// ============================================================================

AllToOneTransfer::AllToOneTransfer(CopyInitiator initiator, AccessType accessType)
    : Testcase(generateKey(initiator, accessType), generateDesc(initiator, accessType))
    , initiator_(initiator)
    , accessType_(accessType)
    , accessStrategy_(createDeviceToDeviceAccessStrategy(accessType)) {}

FilterResult AllToOneTransfer::filter() {
    return Testcase::filterHasAccessiblePeerPairs();
}

MemcpyInitiator* AllToOneTransfer::createMemcpyInitiator() const {
    switch (initiator_) {
        case CopyInitiator::CE:  return new MemcpyInitiatorCE();
        case CopyInitiator::SM:  return new MemcpyInitiatorSM();
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for all-to-one transfers");
            return nullptr;
    }
    return nullptr;
}

std::string AllToOneTransfer::generateKey(CopyInitiator initiator, AccessType accessType) {
    std::string result = "all_to_one_";
    result += (accessType == AccessType::Write) ? "write_" : "read_";
    switch (initiator) {
        case CopyInitiator::CE:  result += "ce"; break;
        case CopyInitiator::SM:  result += "sm"; break;
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for all-to-one transfers");
            break;
    }
    return result;
}

std::string AllToOneTransfer::generateDesc(CopyInitiator /*initiator*/, AccessType accessType) {
    if (accessType == AccessType::Write) {
        return "\tMeasures the total bandwidth of copies from all accessible peers to a single device, for each\n"
               "\tdevice. Bandwidth is reported as the total inbound bandwidth for each device.\n"
               "\tWrite tests launch a copy from the target device to the peer using the target's context.";
    }
    return "\tMeasures the total bandwidth of copies from all accessible peers to a single device, for each\n"
           "\tdevice. Bandwidth is reported as the total outbound bandwidth for each device.\n"
           "\tRead tests launch a copy from the peer device to the target using the target's context.";
}

std::string AllToOneTransfer::buildOutputLabel() const {
    std::string initiatorName = (initiator_ == CopyInitiator::CE) ? "CE" : "SM";
    std::string arrow = (accessType_ == AccessType::Write) ? "<-" : "->";
    return "memcpy " + initiatorName + " GPU(row) " + arrow + " GPU(column) bandwidth (GB/s)";
}

void AllToOneTransfer::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);

    MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator(),
                                   accessStrategy_->getContextPreference(),
                                   MemcpyOperation::TOTAL_BW);

    if (initiator_ == CopyInitiator::SM) {
        initializeNvlinkSchedulingCache(deviceCount);
        bool enableNvlinkScheduling = !gSettings.disableNvlinkSMScheduling &&
                                      allDevicesSupportNvlinkScheduling(deviceCount);
        memcpyInstance.setUseNvlinkUtilScheduling(enableNvlinkScheduling);
    }

    allToOneHelper(size, memcpyInstance, bandwidthValues);

    output->addTestcaseResults(bandwidthValues, buildOutputLabel());
}

// ============================================================================
// OneToAllTransfer
// ============================================================================

OneToAllTransfer::OneToAllTransfer(CopyInitiator initiator, AccessType accessType)
    : Testcase(generateKey(initiator, accessType), generateDesc(accessType))
    , initiator_(initiator)
    , accessType_(accessType)
    , accessStrategy_(createDeviceToDeviceAccessStrategy(accessType)) {}

FilterResult OneToAllTransfer::filter() {
    return Testcase::filterHasAccessiblePeerPairs();
}

MemcpyInitiator* OneToAllTransfer::createMemcpyInitiator() const {
    switch (initiator_) {
        case CopyInitiator::CE:  return new MemcpyInitiatorCE();
        case CopyInitiator::SM:  return new MemcpyInitiatorSMOneToAll();
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for one-to-all transfers");
            return nullptr;
    }
    return nullptr;
}

std::string OneToAllTransfer::generateKey(CopyInitiator initiator, AccessType accessType) {
    std::string result = "one_to_all_";
    result += (accessType == AccessType::Write) ? "write_" : "read_";
    switch (initiator) {
        case CopyInitiator::CE:  result += "ce"; break;
        case CopyInitiator::SM:  result += "sm"; break;
        case CopyInitiator::TMA:
            ASSERT(false && "TMA not supported for one-to-all transfers");
            break;
    }
    return result;
}

std::string OneToAllTransfer::generateDesc(AccessType accessType) {
    if (accessType == AccessType::Write) {
        return "\tMeasures the total bandwidth of copies from a single device to all accessible peers, for each\n"
               "\tdevice. Bandwidth is reported as the total outbound bandwidth for each device.\n"
               "\tWrite tests launch a copy from the target device to the peer using the target's context.";
    }
    return "\tMeasures the total bandwidth of copies from a single device to all accessible peers, for each\n"
           "\tdevice. Bandwidth is reported as the total inbound bandwidth for each device.\n"
           "\tRead tests launch a copy from the peer device to the target using the target's context.";
}

std::string OneToAllTransfer::buildOutputLabel() const {
    std::string initiatorName = (initiator_ == CopyInitiator::CE) ? "CE" : "SM";
    std::string arrow = (accessType_ == AccessType::Write) ? "->" : "<-";
    return "memcpy " + initiatorName + " GPU(row) " + arrow + " GPU(column) bandwidth (GB/s)";
}

void OneToAllTransfer::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);

    MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator(),
                                   accessStrategy_->getContextPreference(),
                                   MemcpyOperation::TOTAL_BW);

    if (initiator_ == CopyInitiator::SM) {
        initializeNvlinkSchedulingCache(deviceCount);
        bool enableNvlinkScheduling = !gSettings.disableNvlinkSMScheduling &&
                                      allDevicesSupportNvlinkScheduling(deviceCount);
        memcpyInstance.setUseNvlinkUtilScheduling(enableNvlinkScheduling);
    }

    oneToAllHelper(size, memcpyInstance, bandwidthValues);

    output->addTestcaseResults(bandwidthValues, buildOutputLabel());
}

// ============================================================================
// DeviceLocalCopyTransfer
// ============================================================================

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
// ----------------------------------------------------------------------------
// Locality-pinned buffer helpers, shared by the local, device-to-device and
// concurrent localized tests further below.
// ----------------------------------------------------------------------------

static void logLocalizedMemoryNode(const char* name, CUdeviceptr ptr, unsigned int expected) {
    unsigned int ordinal = ~0u;
    CUresult r = cuPointerGetAttribute(&ordinal,
        CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL, ptr);
    if (r == CUDA_SUCCESS) {
        VERBOSE << "  " << name << " memoryNodeOrdinal=" << ordinal
                << (ordinal == expected ? "  OK" : "  MISMATCH (expected "
                    + std::to_string(expected) + ")") << "\n";
    }
}

// The node indices used by every localized test: 0..memoryNodeCount-1.
static std::vector<unsigned int> localizedNodeIndices(int memoryNodeCount) {
    std::vector<unsigned int> nodeIndices;
    nodeIndices.reserve(memoryNodeCount);
    for (int i = 0; i < memoryNodeCount; i++)
        nodeIndices.push_back(static_cast<unsigned int>(i));
    return nodeIndices;
}

// Allocate N local buffers on `device`, one pinned to each nodeIndices entry.
static std::vector<std::unique_ptr<LocalizedDeviceBuffer>> allocateLocalizedBuffers(
        unsigned long long size, int device,
        const std::vector<unsigned int>& nodeIndices) {
    std::vector<std::unique_ptr<LocalizedDeviceBuffer>> bufs;
    bufs.reserve(nodeIndices.size());
    for (unsigned int node : nodeIndices)
        bufs.push_back(std::make_unique<LocalizedDeviceBuffer>(size, device, node));
    return bufs;
}

// Allocate N peer buffers on `peerDevice`, one per locality domain, using
// localized VMM.
static void allocateLocalizedPeerBuffers(
        unsigned long long size, int peerDevice,
        const std::vector<unsigned int>& nodeIndices,
        std::vector<std::unique_ptr<MemcpyBuffer>>& peerBufs) {
    peerBufs.clear();
    peerBufs.reserve(nodeIndices.size());
    for (unsigned int node : nodeIndices) {
        peerBufs.push_back(std::make_unique<LocalizedDeviceBuffer>(
            size, peerDevice, node));
    }
}

static const char* localizedCeCopyPathLabel() {
    return "cuMemcpyAsync (CE) + VMM localized alloc";
}
#endif  // NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT

DeviceLocalCopyTransfer::DeviceLocalCopyTransfer(CopyInitiator initiator)
    : Testcase(generateKey(initiator), generateDesc(initiator))
    , initiator_(initiator) {}

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
namespace {
// Deliberately leaked (raw pointer, never deleted): this cache is
// process-lifetime by design, and its destructor's cuGreenCtxDestroy calls
// would otherwise run during static destruction at exit, whose order relative
// to the CUDA driver's own atexit-based teardown is unspecified - freeing here
// risks calling into an already-deinitialized driver. Process exit reclaims
// the underlying resources regardless.
struct GreenCtxCacheSlot {
    unsigned int numDomains = 0;
    LocalizedGreenContexts* ctxs = nullptr;
};
std::vector<GreenCtxCacheSlot> greenCtxCache;

LocalizedGreenContexts& localizedGreenContextsForDevice(int deviceId, unsigned int numDomains) {
    if (static_cast<size_t>(deviceId) >= greenCtxCache.size()) {
        greenCtxCache.resize(deviceId + 1);
    }
    GreenCtxCacheSlot& slot = greenCtxCache[deviceId];
    if (!slot.ctxs) {
        slot.ctxs = new LocalizedGreenContexts(deviceId, numDomains);
        slot.numDomains = numDomains;
    } else {
        // Domain count is a device property; a mismatch means the cache key was
        // incomplete and would return the wrong number of green contexts.
        ASSERT(slot.numDomains == numDomains &&
               "localizedGreenContextsForDevice: numDomains mismatch for device");
    }
    return *slot.ctxs;
}
}  // namespace
#endif

FilterResult DeviceLocalCopyTransfer::filter() {
    if (initiator_ == CopyInitiator::TMA) {
        auto tma = Testcase::filterSupportsTMA();
        if (!tma) return tma;
    }
    return FilterResult::pass();
}

MemcpyInitiator* DeviceLocalCopyTransfer::createMemcpyInitiator() const {
    switch (initiator_) {
        case CopyInitiator::CE:  return new MemcpyInitiatorCE();
        case CopyInitiator::SM:  return new MemcpyInitiatorSM();
        case CopyInitiator::TMA: return new MemcpyInitiatorTMA();
        default:
            ASSERT(false && "unsupported initiator for DeviceLocalCopyTransfer");
            return nullptr;
    }
}

std::string DeviceLocalCopyTransfer::generateKey(CopyInitiator initiator) {
    switch (initiator) {
        case CopyInitiator::CE:  return "device_local_copy";
        case CopyInitiator::SM:  return "device_local_copy_sm";
        case CopyInitiator::TMA: return "device_local_copy_tma";
        default:
            ASSERT(false && "unsupported initiator for DeviceLocalCopyTransfer");
            return "";
    }
}

std::string DeviceLocalCopyTransfer::generateDesc(CopyInitiator initiator) {
    static const std::string multiDomainNote =
#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
        " On architectures that localize, divisible sizes are split evenly across locality domains "
        "and copied concurrently; an indivisible size falls back to domain 0 only. On other "
        "architectures a single non-localized copy is measured.";
#else
        "";
#endif
    // Intra-device memcpy reads size bytes from device memory and writes size
    // bytes back, so reported bandwidth uses 2*size (bytes moved on the memory
    // interface), not the logical copy size.
    static const std::string deviceLocalCopyNote =
        "\n\tReported bandwidth counts device-memory traffic as 2*size (read + write); "
        "each byte crosses the memory interface twice.";
    // Unlike the peer TMA tests, these always stay on raw pointers: a same-device
    // LE destination sends the copy out through the switch and back, which is not
    // local bandwidth.
    static const std::string rawOnlyNote =
        "\n\tAlways uses raw pointers, since a Handle destination owned by the "
        "copying device would loop back through the switch.";
    switch (initiator) {
        case CopyInitiator::CE:
            return "\tMeasures bandwidth of CE copies (cuMemcpyAsync) between device buffers "
                   "local to the GPU."
                   + deviceLocalCopyNote + rawOnlyNote + multiDomainNote;
        case CopyInitiator::SM:
            return "\tMeasures bandwidth of a copy kernel between device buffers local to the GPU."
                   + deviceLocalCopyNote + multiDomainNote;
        case CopyInitiator::TMA:
            return "\tMeasures bandwidth of TMA copies between device buffers local to the GPU."
                   + deviceLocalCopyNote + rawOnlyNote + multiDomainNote + "\n";
        default:
            ASSERT(false && "unsupported initiator for DeviceLocalCopyTransfer");
            return "";
    }
}

std::string DeviceLocalCopyTransfer::buildOutputLabel() const {
    std::string initiatorName;
    switch (initiator_) {
        case CopyInitiator::CE:  initiatorName = "CE"; break;
        case CopyInitiator::SM:  initiatorName = "SM"; break;
        case CopyInitiator::TMA: initiatorName = "TMA"; break;
        default:
            ASSERT(false && "unsupported initiator for DeviceLocalCopyTransfer");
            break;
    }
    return "memcpy " + initiatorName + " local GPU(column) bandwidth (GB/s)";
}

void DeviceLocalCopyTransfer::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator());
    // Intra-device copy: each logical byte is read once and written once in
    // device memory. Scale on the operation so verbose samples and the reported
    // metric both count 2*size of device-memory traffic.
    memcpyInstance.setBandwidthScale(2.0);

    // Raw-pointer direction matches historical local CE/TMA (dst->src) vs SM (src->dst).
    auto measureBandwidth = [&](MemcpyBuffer& deviceBuffer1, MemcpyBuffer& deviceBuffer2) {
        if (initiator_ == CopyInitiator::SM) {
            return memcpyInstance.doMemcpy(deviceBuffer1, deviceBuffer2);
        }
        return memcpyInstance.doMemcpy(deviceBuffer2, deviceBuffer1);
    };

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
        int numDomains = 0;
        if (shouldLocalize(deviceId, numDomains) && size % numDomains == 0) {
            // Split both buffers across all locality domains and drive every
            // locality domain's copy concurrently, reporting one combined
            // bandwidth number.
            MemcpyOperation splitMemcpy(loopCount, createMemcpyInitiator(),
                                         ContextPreference::PREFER_SRC_CONTEXT,
                                         MemcpyOperation::TOTAL_BW);
            splitMemcpy.setBandwidthScale(2.0);
            SplitLocalizedDeviceBuffer srcBuf(size, deviceId, numDomains);
            SplitLocalizedDeviceBuffer dstBuf(size, deviceId, numDomains);
            LocalizedGreenContexts& greenCtxs =
                localizedGreenContextsForDevice(deviceId, numDomains);

            std::vector<std::unique_ptr<LocalizedBufferChunkView>> srcViews, dstViews;
            std::vector<const MemcpyBuffer*> srcBuffers, dstBuffers;
            for (int d = 0; d < numDomains; d++) {
                srcViews.push_back(
                    std::make_unique<LocalizedBufferChunkView>(srcBuf, d, &greenCtxs));
                dstViews.push_back(
                    std::make_unique<LocalizedBufferChunkView>(dstBuf, d, &greenCtxs));
                srcBuffers.push_back(srcViews.back().get());
                dstBuffers.push_back(dstViews.back().get());
            }

            bandwidthValues.value(0, deviceId) =
                splitMemcpy.doMemcpy(srcBuffers, dstBuffers);
            // splitMemcpy is per-device; record before it goes out of scope.
            recordStability(bandwidthValues, splitMemcpy);
            continue;
        }
        if (numDomains >= 2) {
            // size isn't evenly divisible by numDomains: fall back to today's
            // domain-0-only behavior for this data point rather than dropping
            // localization entirely.
            VERBOSE << "[" << key << "] size=" << size << " not divisible by "
                    << numDomains << " domains on device " << deviceId
                    << "; reporting domain-0-only bandwidth\n";
            LocalizedDeviceBuffer deviceBuffer1(size, deviceId, 0);
            LocalizedDeviceBuffer deviceBuffer2(size, deviceId, 0);

            bandwidthValues.value(0, deviceId) =
                measureBandwidth(deviceBuffer1, deviceBuffer2);
            continue;
        }
#endif
        DeviceBuffer deviceBuffer1(size, deviceId);
        DeviceBuffer deviceBuffer2(size, deviceId);

        bandwidthValues.value(0, deviceId) =
            measureBandwidth(deviceBuffer1, deviceBuffer2);
    }

    recordStability(bandwidthValues, memcpyInstance);
    output->addTestcaseResults(bandwidthValues, buildOutputLabel());
}

// ============================================================================
// Latency Tests (SM)
// ============================================================================

void HostDeviceLatencySM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> latencyValues(1, deviceCount, key, gSettings.perfFormatter, LATENCY);
    MemPtrChaseOperation ptrChaseOp(latencyMemAccessCnt);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer dataBuffer(size, deviceId);
        latencyHelper(dataBuffer, false);
        latencyValues.value(0, deviceId) = ptrChaseOp.doPtrChase(deviceId, dataBuffer, launchPtrChasingKernelSM);
    }

    recordStability(latencyValues, ptrChaseOp);
    output->addTestcaseResults(latencyValues, "memory latency SM CPU(row) <-> GPU(column) (ns)");
}

void DeviceToDeviceLatencySM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> latencyValues(deviceCount, deviceCount, key, gSettings.perfFormatter, LATENCY);
    MemPtrChaseOperation ptrChaseOp(latencyMemAccessCnt);

    for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
        if (!peerIsGpuPairEndpoint(peerDeviceId)) {
            continue;
        }

        DeviceBuffer peerBuffer(size, peerDeviceId);
        latencyHelper(peerBuffer, true);

        for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }
            if (!peerIsGpuPairEndpoint(srcDeviceId)) {
                continue;
            }

            // Note: srcBuffer is not used in the pointer chase operation
            // It is simply used here to enable peer access
            DeviceBuffer srcBuffer(size, srcDeviceId);
            if (!srcBuffer.enablePeerAccess(peerBuffer)) {
                continue;
            }
            latencyValues.value(srcDeviceId, peerDeviceId) = ptrChaseOp.doPtrChase(srcDeviceId, peerBuffer, launchPtrChasingKernelSM);
        }
    }
    recordStability(latencyValues, ptrChaseOp);
    output->addTestcaseResults(latencyValues, "Device to Device Latency SM GPU(row) <-> GPU(column) (ns)");
}

// ============================================================================
// Device Local Read/Write Tests (SM)
// ============================================================================

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
// Splits the read-only source buffer across every uGPU locality domain and drives
// all domains concurrently, each from its own domain-local green context,
// reporting the aggregate. Mirrors the split path in DeviceLocalCopyTransfer::run().
//
// Only the source is localized; the destination stays a minimal per-domain scratch
// buffer that the kernel never touches, because MemcpyInitiatorSMReadOnly sizes the
// copy from the source alone.
//
// Returns false when the device has one domain or size does not divide evenly,
// leaving the caller on its non-localized path for that data point.
static bool runLocalizedReadOnlySM(MemcpyOperation &splitMemcpy, const std::string &key,
                                   int deviceId, unsigned long long size,
                                   double &bandwidthOut) {
    int numDomains = 0;
    if (!shouldLocalize(deviceId, numDomains)) return false;
    if (size % numDomains != 0) {
        VERBOSE << "[" << key << "] size=" << size << " not divisible by "
                << numDomains << " domains on device " << deviceId
                << "; reporting non-localized bandwidth\n";
        return false;
    }

    SplitLocalizedDeviceBuffer srcBuf(size, deviceId, numDomains);
    LocalizedGreenContexts &greenCtxs = localizedGreenContextsForDevice(deviceId, numDomains);

    std::vector<std::unique_ptr<LocalizedBufferChunkView>> srcViews;
    std::vector<std::unique_ptr<DeviceBuffer>> scratchBufs;
    std::vector<const MemcpyBuffer *> srcBuffers, dstBuffers;
    for (int d = 0; d < numDomains; d++) {
        srcViews.push_back(std::make_unique<LocalizedBufferChunkView>(srcBuf, d, &greenCtxs));
        scratchBufs.push_back(std::make_unique<DeviceBuffer>(sizeof(unsigned), deviceId));
        srcBuffers.push_back(srcViews.back().get());
        dstBuffers.push_back(scratchBufs.back().get());
    }

    bandwidthOut = splitMemcpy.doMemcpy(srcBuffers, dstBuffers);
    return true;
}
#endif  // NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT

void DeviceLocalReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSMReadOnly());
#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
    // Measured buffer is src, so the kernel runs in the src chunk's green context.
    MemcpyOperation splitMemcpy(loopCount, new MemcpyInitiatorSMReadOnly(),
                                ContextPreference::PREFER_SRC_CONTEXT,
                                MemcpyOperation::TOTAL_BW);
#endif

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
        double localizedBw = 0.0;
        if (runLocalizedReadOnlySM(splitMemcpy, key, deviceId, size, localizedBw)) {
            bandwidthValues.value(0, deviceId) = localizedBw;
            continue;
        }
#endif
        DeviceBuffer srcBuffer(size, deviceId);
        // Read-only test: allocate minimal destination buffer (not actually used)
        DeviceBuffer dstBuffer(sizeof(unsigned), deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcBuffer, dstBuffer);
    }

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
    recordStability(bandwidthValues, splitMemcpy, memcpyInstance);
#else
    recordStability(bandwidthValues, memcpyInstance);
#endif
    output->addTestcaseResults(bandwidthValues, "memcpy SM local GPU(column) read-only bandwidth (GB/s)");
}

void DeviceLocalWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSMWriteOnly());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        // Write-only test: allocate minimal source buffer (not actually used)
        DeviceBuffer srcBuffer(sizeof(unsigned), deviceId);
        DeviceBuffer dstBuffer(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcBuffer, dstBuffer);
    }

    recordStability(bandwidthValues, memcpyInstance);
    output->addTestcaseResults(bandwidthValues, "memcpy SM local GPU(column) write-only bandwidth (GB/s)");
}

// ============================================================================
// Latency Tests (TMA)
// ============================================================================

void DeviceToDeviceLatencyTMA::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> latencyValues(deviceCount, deviceCount, key, gSettings.perfFormatter, LATENCY);
    MemPtrChaseOperation ptrChaseOp(latencyMemAccessCnt);

    for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
        if (!peerIsGpuPairEndpoint(peerDeviceId)) {
            continue;
        }

        DeviceBuffer peerBuffer(size, peerDeviceId);
        latencyHelper<TMALatencyNode>(peerBuffer, true);  // Use 16-byte aligned TMALatencyNode

        for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }
            if (!peerIsGpuPairEndpoint(srcDeviceId)) {
                continue;
            }

            // Note: srcBuffer is not used in the pointer chase operation
            // It is simply used here to enable peer access
            DeviceBuffer srcBuffer(size, srcDeviceId);
            if (!srcBuffer.enablePeerAccess(peerBuffer)) {
                continue;
            }
            latencyValues.value(srcDeviceId, peerDeviceId) = ptrChaseOp.doPtrChase(srcDeviceId, peerBuffer, launchPtrChasingKernelTMA);
        }
    }
    recordStability(latencyValues, ptrChaseOp);
    output->addTestcaseResults(latencyValues, "Device to Device Latency TMA GPU(row) <-> GPU(column) (ns)");
}
// ============================================================================
// uGPU localization tests
// ============================================================================

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT

// One access direction of a localized concurrent CE measurement: one locality-pinned
// buffer pair per locality domain.
struct LocalizedCeDirection {
    std::vector<std::unique_ptr<LocalizedDeviceBuffer>> localBufs;
    std::vector<std::unique_ptr<MemcpyBuffer>> peerBufs;
};

// Callers reverse the two devices to build the opposite direction of a
// bidirectional pair.
static bool buildLocalizedCeDirection(unsigned long long size, int localDev, int peerDev,
                                      const std::vector<unsigned int>& nodeIndices,
                                      LocalizedCeDirection& accessDirection) {
    accessDirection.localBufs = allocateLocalizedBuffers(size, localDev, nodeIndices);
    allocateLocalizedPeerBuffers(size, peerDev, nodeIndices, accessDirection.peerBufs);
    return accessDirection.localBufs[0]->enablePeerAccess(*accessDirection.peerBufs[0]);
}

// Runs one stream per locality domain per access direction, all simultaneously, and
// reports each direction's bytes over the span in which it ran. The driver routes each
// cuMemcpyAsync to the copy engine with affinity for the buffer's locality domain
// (requires the matching driver release).
//
// bandwidthPerDirection is sized to directions.size() on success.
static bool runLocalizedConcurrentCeDirections(
        const std::string& verboseTag, int ctxDev, int peerDev,
        const std::vector<LocalizedCeDirection>& directions,
        unsigned long long size, unsigned long long loopCount,
        const std::vector<unsigned int>& nodeIndices,
        const DeviceToDeviceAccessStrategy& accessStrategy,
        std::vector<double>& bandwidthPerDirection) {
    const int localityDomains = static_cast<int>(nodeIndices.size());
    const int directionCount = static_cast<int>(directions.size());
    if (localityDomains == 0 || directionCount == 0) return false;
    const int streamCount = directionCount * localityDomains;

    if (gSettings.verbose) {
        VERBOSE << "[" << verboseTag << " src=" << ctxDev << " dst=" << peerDev << "] "
                << "buffer memory node placement (raw):\n";
        for (int d = 0; d < directionCount; d++) {
            for (int i = 0; i < localityDomains; i++) {
                std::string idx = (directionCount > 1 ? "dir" + std::to_string(d) : std::string()) +
                                  std::to_string(i);
                logLocalizedMemoryNode(("localBuf" + idx).c_str(),
                                       directions[d].localBufs[i]->getBuffer(), nodeIndices[i]);
                if (directions[d].peerBufs[i]->getDevicePtr() != 0)
                    logLocalizedMemoryNode(("peerBuf" + idx).c_str(),
                                           directions[d].peerBufs[i]->getDevicePtr(),
                                           nodeIndices[i]);
            }
        }
    }

    CUcontext ctxDevPrimaryCtx = nullptr;
    CUcontext peerPrimaryCtx   = nullptr;
    CU_ASSERT(cuDevicePrimaryCtxRetain(&ctxDevPrimaryCtx, static_cast<CUdevice>(ctxDev)));
    CU_ASSERT(cuDevicePrimaryCtxRetain(&peerPrimaryCtx, static_cast<CUdevice>(peerDev)));

    CUcontext callerCtx = nullptr;
    CU_ASSERT(cuCtxGetCurrent(&callerCtx));

    // A direction is driven by the device holding its local buffers, so a bidirectional
    // pair has each device driving its own direction rather than one driving both.
    std::vector<CUcontext> directionContexts(directionCount);
    directionContexts[0] = ctxDevPrimaryCtx;
    if (directionCount > 1) directionContexts[1] = peerPrimaryCtx;

    std::vector<CUcontext> streamContexts(streamCount);
    for (int i = 0; i < streamCount; i++)
        streamContexts[i] = directionContexts[i / localityDomains];

    CU_ASSERT(cuCtxSetCurrent(ctxDevPrimaryCtx));

    CUresult peerAccess = cuCtxEnablePeerAccess(peerPrimaryCtx, 0);
    if (peerAccess != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) CU_ASSERT(peerAccess);
    if (directionCount > 1) {
        CU_ASSERT(cuCtxSetCurrent(peerPrimaryCtx));
        peerAccess = cuCtxEnablePeerAccess(ctxDevPrimaryCtx, 0);
        if (peerAccess != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) CU_ASSERT(peerAccess);
        CU_ASSERT(cuCtxSetCurrent(ctxDevPrimaryCtx));
    }

    auto orderedFor = [&](int i) {
        const LocalizedCeDirection& accessDirection = directions[i / localityDomains];
        const int node = i % localityDomains;
        return accessStrategy.getOrderedBuffers(*accessDirection.localBufs[node],
                                                *accessDirection.peerBufs[node]);
    };

    auto submitCopyOn = [&](int i, CUstream stream) {
        auto [memcpySrc, memcpyDst] = orderedFor(i);
        submitLocalizedCeCopy(*memcpySrc, *memcpyDst, size, stream);
    };

    if (gSettings.verbose) {
        VERBOSE << "[" << verboseTag << " src=" << ctxDev << " dst=" << peerDev
                << "] copy path: " << localizedCeCopyPathLabel()
                << ", " << streamCount << " concurrent streams\n";
    }

    // Directions write into disjoint buffers, so every stream is checked independently.
    if (!gSettings.skipVerification) {
        for (int i = 0; i < streamCount; i++) {
            CU_ASSERT(cuCtxSetCurrent(streamContexts[i]));
            auto [memcpySrc, memcpyDst] = orderedFor(i);
            prepareLocalizedConcurrentCeCopy(*memcpySrc, *memcpyDst, size);
        }
        CU_ASSERT(cuCtxSetCurrent(ctxDevPrimaryCtx));
    }

    // One group per direction, each on its own device.
    std::vector<double> perDirectionBw;
    runConcurrentLocalizedStreams(streamCount, size, loopCount, "localized_concurrent",
                                  submitCopyOn, nullptr, &streamContexts,
                                  localityDomains, &perDirectionBw);

    if (!gSettings.skipVerification) {
        for (int i = 0; i < streamCount; i++) {
            CU_ASSERT(cuCtxSetCurrent(streamContexts[i]));
            auto [memcpySrc, memcpyDst] = orderedFor(i);
            verifyLocalizedConcurrentCeCopy(*memcpySrc, *memcpyDst, size);
        }
    }

    cuDevicePrimaryCtxRelease(static_cast<CUdevice>(peerDev));
    cuDevicePrimaryCtxRelease(static_cast<CUdevice>(ctxDev));
    CU_ASSERT(cuCtxSetCurrent(callerCtx));

    if (static_cast<int>(perDirectionBw.size()) != directionCount) return false;
    bandwidthPerDirection = perDirectionBw;
    return true;
}

// Entry point for the ordinary CE device-to-device tests. A multi-domain device
// reaches its full peer bandwidth only when every domain drives the link at once,
// so where the hardware allows it the pair is measured that way rather than from a
// single domain. Bidirectional callers get one bandwidth per direction back;
// unidirectional callers get a single entry.
//
// Both endpoints must report the same domain count: the pair is measured with one
// buffer per domain on each side, so a mismatch would ask the peer for a domain it
// does not have. Returns false whenever the pair cannot be localized -- one domain,
// mismatched counts, or no usable peer access -- leaving the caller on its
// non-localized path for that data point.
static bool runLocalizedConcurrentCeD2D(const std::string &key, int srcDeviceId, int dstDeviceId,
                                        unsigned long long size, unsigned long long loopCount,
                                        const DeviceToDeviceAccessStrategy &accessStrategy,
                                        bool bidirectional,
                                        std::vector<double> &bandwidthPerDirection) {
    int srcDomains = 0, dstDomains = 0;
    if (!shouldLocalize(srcDeviceId, srcDomains) ||
        !shouldLocalize(dstDeviceId, dstDomains) ||
        srcDomains != dstDomains) {
        return false;
    }

    const std::vector<unsigned int> nodeIndices = localizedNodeIndices(srcDomains);

    std::vector<LocalizedCeDirection> directions(bidirectional ? 2 : 1);
    if (!buildLocalizedCeDirection(size, srcDeviceId, dstDeviceId, nodeIndices,
                                   directions[0]))
        return false;
    if (bidirectional &&
        !buildLocalizedCeDirection(size, dstDeviceId, srcDeviceId, nodeIndices,
                                   directions[1]))
        return false;

    return runLocalizedConcurrentCeDirections(key, srcDeviceId, dstDeviceId, directions, size,
                                              loopCount, nodeIndices, accessStrategy,
                                              bandwidthPerDirection);
}

#endif  // NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
