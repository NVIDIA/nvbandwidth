/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>
#include <cuda_runtime_api.h>

#include "common.h"
#include "output.h"
#include "testcase.h"
#include "test_settings.h"
#include "memcpy.h"

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
    if (initiator_ == CopyInitiator::CE) return FilterResult::pass();
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
    std::string initiatorName = (initiator_ == CopyInitiator::CE) ? "CE" : "SM";
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

DeviceToDeviceTransfer::DeviceToDeviceTransfer(CopyInitiator initiator, AccessType accessType)
    : Testcase(generateKey(initiator, accessType), generateDesc(initiator, accessType))
    , initiator_(initiator)
    , accessType_(accessType)
    , accessStrategy_(createAccessStrategy(accessType))
    , bufferStrategy_(createBufferStrategy()) {}

FilterResult DeviceToDeviceTransfer::filter() {
    auto r = Testcase::filterHasAccessiblePeerPairs();
    if (!r) return r;
    if (initiator_ == CopyInitiator::TMA) {
        return Testcase::filterSupportsTMA();
    }
    return FilterResult::pass();
}

std::unique_ptr<BufferTypeStrategy> DeviceToDeviceTransfer::createBufferStrategy() const {
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

std::unique_ptr<DeviceToDeviceAccessStrategy> DeviceToDeviceTransfer::createAccessStrategy(AccessType accessType) {
    switch (accessType) {
        case AccessType::Read:  return std::make_unique<ReadAccessStrategy>();
        case AccessType::Write: return std::make_unique<WriteAccessStrategy>();
    }
    return std::make_unique<WriteAccessStrategy>();
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

std::string DeviceToDeviceTransfer::generateDesc(CopyInitiator initiator, AccessType accessType) {
    std::string result = "\tMeasures bandwidth of ";
    switch (initiator) {
        case CopyInitiator::CE:  result += "cuMemcpyAsync"; break;
        case CopyInitiator::SM:  result += "a copy kernel"; break;
        case CopyInitiator::TMA: result += "TMA copies"; break;
    }
    result += " between each pair of accessible peers.\n\t";
    if (accessType == AccessType::Read) {
        result += "Read tests launch a copy from the peer device to the target using the target's context.";
    } else {
        result += "Write tests launch a copy from the target device to the peer using the target's context.";
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
    // Check buffer type support (always true for DeviceBuffer, but extensible)
    if (!bufferStrategy_->isSupported()) {
        std::cout << "Skipping " << testKey() << ": "
                  << bufferStrategy_->getLabel() << " buffer not supported" << std::endl;
        return;
    }

    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator(), accessStrategy_->getContextPreference());

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
    , accessStrategy_(createAccessStrategy(accessType))
    , bufferStrategy_(createBufferStrategy()) {}

FilterResult DeviceToDeviceBidirTransfer::filter() {
    auto r = Testcase::filterHasAccessiblePeerPairs();
    if (!r) return r;
    if (initiator_ == CopyInitiator::TMA) {
        return Testcase::filterSupportsTMA();
    }
    return FilterResult::pass();
}

std::unique_ptr<BufferTypeStrategy> DeviceToDeviceBidirTransfer::createBufferStrategy() const {
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

std::unique_ptr<DeviceToDeviceAccessStrategy> DeviceToDeviceBidirTransfer::createAccessStrategy(AccessType accessType) {
    switch (accessType) {
        case AccessType::Read:  return std::make_unique<ReadAccessStrategy>();
        case AccessType::Write: return std::make_unique<WriteAccessStrategy>();
    }
    return std::make_unique<WriteAccessStrategy>();
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
        result = "\tMeasures bandwidth of cuMemcpyAsync between each pair of accessible peers.\n"
                 "\tA copy in the opposite direction of the measured copy is run simultaneously but not measured.\n\t";
    } else if (initiator == CopyInitiator::SM) {
        result = "\tMeasures bandwidth of a copy kernel between each pair of accessible peers. Copies are run\n"
                 "\tin both directions between each pair, and the sum is reported.\n\t";
    } else {
        result = "\tMeasures bandwidth of TMA bidirectional copies between each pair of accessible peers.\n\t";
    }
    result += (accessType == AccessType::Read)
        ? "Read tests launch a copy from the peer device to the target using the target's context."
        : "Write tests launch a copy from the target device to the peer using the target's context.";
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
    // Check buffer type support (always true for DeviceBuffer, but extensible)
    if (!bufferStrategy_->isSupported()) {
        std::cout << "Skipping " << testKey() << ": "
                  << bufferStrategy_->getLabel() << " buffer not supported" << std::endl;
        return;
    }

    std::string suffix = (accessType_ == AccessType::Read) ? "read" : "write";
    PeerValueMatrix<double> bandwidthValues1(deviceCount, deviceCount, key + "_" + suffix + "1");
    PeerValueMatrix<double> bandwidthValues2(deviceCount, deviceCount, key + "_" + suffix + "2");
    PeerValueMatrix<double> bandwidthValuesTotal(deviceCount, deviceCount, key + "_total");

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

            auto pair1 = bufferStrategy_->createBufferPair(size, srcDeviceId, peerDeviceId);
            auto pair2 = bufferStrategy_->createBufferPair(size, srcDeviceId, peerDeviceId);

            if (!bufferStrategy_->enablePeerAccess(*pair1)) {
                return {0.0, 0.0};
            }

            // pair1->src = src1, pair1->peer = peer1
            // pair2->src = src2, pair2->peer = peer2
            std::vector<const MemcpyBuffer*> srcBuffers = {pair1->peer.get(), pair2->src.get()};
            std::vector<const MemcpyBuffer*> peerBuffers = {pair1->src.get(), pair2->peer.get()};

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
    , accessStrategy_(createAccessStrategy(accessType)) {}

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

std::unique_ptr<DeviceToDeviceAccessStrategy> AllToOneTransfer::createAccessStrategy(AccessType accessType) {
    switch (accessType) {
        case AccessType::Read:  return std::make_unique<ReadAccessStrategy>();
        case AccessType::Write: return std::make_unique<WriteAccessStrategy>();
    }
    return std::make_unique<WriteAccessStrategy>();
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
    , accessStrategy_(createAccessStrategy(accessType)) {}

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

std::unique_ptr<DeviceToDeviceAccessStrategy> OneToAllTransfer::createAccessStrategy(AccessType accessType) {
    switch (accessType) {
        case AccessType::Read:  return std::make_unique<ReadAccessStrategy>();
        case AccessType::Write: return std::make_unique<WriteAccessStrategy>();
    }
    return std::make_unique<WriteAccessStrategy>();
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

DeviceLocalCopyTransfer::DeviceLocalCopyTransfer(CopyInitiator initiator)
    : Testcase(generateKey(initiator), generateDesc(initiator))
    , initiator_(initiator) {}

FilterResult DeviceLocalCopyTransfer::filter() {
    if (initiator_ == CopyInitiator::TMA) {
        return Testcase::filterSupportsTMA();
    }
    return FilterResult::pass();
}

MemcpyInitiator* DeviceLocalCopyTransfer::createMemcpyInitiator() const {
    switch (initiator_) {
        case CopyInitiator::CE:  return new MemcpyInitiatorCE();
        case CopyInitiator::SM:  return new MemcpyInitiatorSM();
        case CopyInitiator::TMA: return new MemcpyInitiatorTMA();
    }
    return nullptr;
}

std::string DeviceLocalCopyTransfer::generateKey(CopyInitiator initiator) {
    switch (initiator) {
        case CopyInitiator::CE:  return "device_local_copy";
        case CopyInitiator::SM:  return "device_local_copy_sm";
        case CopyInitiator::TMA: return "device_local_copy_tma";
    }
    return "";
}

std::string DeviceLocalCopyTransfer::generateDesc(CopyInitiator initiator) {
    switch (initiator) {
        case CopyInitiator::CE:
            return "\tMeasures bandwidth of cuMemcpyAsync between device buffers local to the GPU.";
        case CopyInitiator::SM:
            return "\tMeasures bandwidth of a copy kernel between device buffers local to the GPU.";
        case CopyInitiator::TMA:
            return "\tMeasures bandwidth of TMA copies between device buffers local to the GPU.\n";
    }
    return "";
}

std::string DeviceLocalCopyTransfer::buildOutputLabel() const {
    std::string initiatorName;
    switch (initiator_) {
        case CopyInitiator::CE:  initiatorName = "CE"; break;
        case CopyInitiator::SM:  initiatorName = "SM"; break;
        case CopyInitiator::TMA: initiatorName = "TMA"; break;
    }
    return "memcpy " + initiatorName + " local GPU(column) bandwidth (GB/s)";
}

void DeviceLocalCopyTransfer::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        DeviceBuffer deviceBuffer1(size, deviceId);
        DeviceBuffer deviceBuffer2(size, deviceId);

        if (initiator_ == CopyInitiator::SM) {
            bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(deviceBuffer1, deviceBuffer2);
        } else {
            bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(deviceBuffer2, deviceBuffer1);
        }
    }

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
            if (!srcBuffer.enablePeerAcess(peerBuffer)) {
                continue;
            }
            latencyValues.value(srcDeviceId, peerDeviceId) = ptrChaseOp.doPtrChase(srcDeviceId, peerBuffer, launchPtrChasingKernelSM);
        }
    }
    output->addTestcaseResults(latencyValues, "Device to Device Latency SM GPU(row) <-> GPU(column) (ns)");
}

// ============================================================================
// Device Local Read/Write Tests (SM)
// ============================================================================

void DeviceLocalReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSMReadOnly());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        DeviceBuffer srcBuffer(size, deviceId);
        // Read-only test: allocate minimal destination buffer (not actually used)
        DeviceBuffer dstBuffer(sizeof(unsigned), deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcBuffer, dstBuffer);
    }

    recordStability(bandwidthValues, memcpyInstance);
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
            if (!srcBuffer.enablePeerAcess(peerBuffer)) {
                continue;
            }
            latencyValues.value(srcDeviceId, peerDeviceId) = ptrChaseOp.doPtrChase(srcDeviceId, peerBuffer, launchPtrChasingKernelTMA);
        }
    }
    output->addTestcaseResults(latencyValues, "Device to Device Latency TMA GPU(row) <-> GPU(column) (ns)");
}
