/*
* Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#include <limits>
#include <mutex>
#include <vector>

#include "rtx_asset_replacer.h"
#include "rtx_fork_hooks.h"
#include "rtx_scene_manager.h"
#include "rtx_opacity_micromap_manager.h"
#include "dxvk_device.h"
#include "dxvk_context.h"
#include "dxvk_buffer.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_preserved_object_picking.h"
#include "rtx_terrain_baker.h"
#include "rtx_texture_manager.h"
#include "rtx_texture.h"
#include "rtx_xess.h"

#include <assert.h>

#include "../d3d9/d3d9_state.h"
#include "vulkan/vulkan_core.h"

#include "rtx_game_capturer.h"
#include "rtx_matrix_helpers.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_lights_data.h"
#include "rtx_light_utils.h"

#include "../util/util_global_time.h"
#include "../util/util_struct_hash.h"

#include "rtx/pass/particles/particle_system_common.h"

namespace {
  // helper function to ensure generating spatialMapHash for external draws is done the same way in multiple places.
  XXH64_hash_t spatialMapHashForExternalDrawMesh(remixapi_MeshHandle mesh) {
    const uintptr_t meshId = reinterpret_cast<uintptr_t>(mesh);
    return XXH3_64bits(&meshId, sizeof(meshId));
  }
} // namespace

namespace dxvk {

  // Compute a hash that can be used to check if an external draw is identical to the previous frame's draw.
  XXH64_hash_t ExternalDrawState::computeExternalDrawIdentityHash() const {
    struct ExternalDrawIdentityHashData {
      uintptr_t meshId;
      XXH64_hash_t materialHash;
      XXH64_hash_t boneHash;
      CameraType::Enum cameraType;
      uint32_t categoriesRaw;
      XXH64_hash_t particleDescHash;
      XXH64_hash_t gpuInstancingHash;
      TexGenMode texgenMode;
      uint8_t usesVertexShader;
      uint8_t usesPixelShader;
      uint8_t zWriteEnable;
      uint8_t zEnable;
      uint8_t skyAutoDetected;
      uint8_t _pad0;
      uint8_t _pad1;
      Matrix4 objectToWorld;
      Matrix4 textureTransform;
    };

    ExternalDrawIdentityHashData data{};

    const DrawCallTransforms& transforms = drawCall.getTransformData();
    data.meshId = reinterpret_cast<uintptr_t>(mesh);
    data.materialHash = drawCall.getMaterialData().getHash();
    data.boneHash = drawCall.getSkinningState().boneHash;
    data.cameraType = cameraType;
    data.categoriesRaw = categories.raw();

    if (optionalParticleDesc.has_value()) {
      data.particleDescHash = optionalParticleDesc->calcHash();
    }

    if (!gpuInstancingTransforms.empty()) {
      data.gpuInstancingHash = XXH3_64bits(
          gpuInstancingTransforms.data(),
          gpuInstancingTransforms.size() * sizeof(Matrix4));
    }

    data.texgenMode = transforms.texgenMode;
    data.usesVertexShader = drawCall.usesVertexShader ? 1u : 0u;
    data.usesPixelShader = drawCall.usesPixelShader ? 1u : 0u;
    data.zWriteEnable = drawCall.zWriteEnable ? 1u : 0u;
    data.zEnable = drawCall.zEnable ? 1u : 0u;
    data.skyAutoDetected = drawCall.skyAutoDetected ? 1u : 0u;
    data.objectToWorld = transforms.objectToWorld;
    data.textureTransform = transforms.textureTransform;

    return hashStructByMemory<ExternalDrawIdentityHashData,
        &ExternalDrawIdentityHashData::meshId,
        &ExternalDrawIdentityHashData::materialHash,
        &ExternalDrawIdentityHashData::boneHash,
        &ExternalDrawIdentityHashData::cameraType,
        &ExternalDrawIdentityHashData::categoriesRaw,
        &ExternalDrawIdentityHashData::particleDescHash,
        &ExternalDrawIdentityHashData::gpuInstancingHash,
        &ExternalDrawIdentityHashData::texgenMode,
        &ExternalDrawIdentityHashData::usesVertexShader,
        &ExternalDrawIdentityHashData::usesPixelShader,
        &ExternalDrawIdentityHashData::zWriteEnable,
        &ExternalDrawIdentityHashData::zEnable,
        &ExternalDrawIdentityHashData::skyAutoDetected,
        &ExternalDrawIdentityHashData::_pad0,
        &ExternalDrawIdentityHashData::_pad1,
        &ExternalDrawIdentityHashData::objectToWorld,
        &ExternalDrawIdentityHashData::textureTransform>(data);
  }

  SceneManager::SceneManager(DxvkDevice* device)
    : CommonDeviceObject(device)
    , m_instanceManager(device, this)
    , m_accelManager(device)
    , m_lightManager(device)
    , m_graphManager()
    , m_rayPortalManager(device, this)
    , m_drawCallCache(device)
    , m_drawCallTracker(device)
    , m_bindlessResourceManager(device)
    , m_pReplacer(new AssetReplacer())
    , m_terrainBaker(new TerrainBaker())
    , m_cameraManager(device)
    , m_uniqueObjectSearchDistance(RtxOptions::uniqueObjectDistance()) {
    InstanceEventHandler instanceEvents(this);
    instanceEvents.onInstanceAddedCallback = [this](RtInstance& instance) { onInstanceAdded(instance); };
    instanceEvents.onInstanceUpdatedCallback = [this](RtInstance& instance, const DrawCallState& drawCall, const MaterialData* material, bool hasTransformChanged, bool hasVerticesChanged, bool isFirstUpdateThisFrame) { onInstanceUpdated(instance, drawCall, material, hasTransformChanged, hasVerticesChanged, isFirstUpdateThisFrame); };
    instanceEvents.onInstanceDestroyedCallback = [this](RtInstance& instance) { onInstanceDestroyed(instance); };
    m_instanceManager.addEventHandler(instanceEvents);
    
    if (env::getEnvVar("DXVK_RTX_CAPTURE_ENABLE_ON_FRAME") != "") {
      m_beginUsdExportFrameNum = stoul(env::getEnvVar("DXVK_RTX_CAPTURE_ENABLE_ON_FRAME"));
    }
  }

  SceneManager::~SceneManager() {
  }

  bool SceneManager::areAllReplacementsLoaded() const {
    return m_pReplacer->areAllReplacementsLoaded();
  }

  std::vector<Mod::State> SceneManager::getReplacementStates() const {
    return m_pReplacer->getReplacementStates();
  }

  void SceneManager::initialize(Rc<DxvkContext> ctx) {
    // Instrumentation manifest.
    //
    // The point of this is discoverability, not diagnostics. Every stream below can be switched on without a
    // rebuild, but only if someone knows it exists -- and that knowledge otherwise lives with whoever wrote
    // it. Printing the inventory into the log means any run, read by anyone, later, carries its own index of
    // what can be turned on and how.
    if (RtxOptions::ForkLogging::manifest()) {
      const auto onOff = [](bool value) { return value ? "on" : "off"; };
      Logger::info(str::format("[Remix instrumentation] ",
        "scatterSubmit=", onOff(RtxOptions::ForkLogging::scatterSubmit()),
        " meshLookups=", onOff(RtxOptions::ForkLogging::meshLookups()),
        " materialLookups=", onOff(RtxOptions::ForkLogging::materialLookups()),
        " heavyAssets=", onOff(RtxOptions::ForkLogging::heavyAssets()),
        " frameSpikes=", onOff(RtxOptions::ForkLogging::frameSpikes()),
        " neeOverflow=", onOff(RtxOptions::ForkLogging::neeOverflow())));
      Logger::info("[Remix instrumentation] set any of the above with rtx.fork.log.<name> = True in rtx.conf "
                   "-- no rebuild needed. Host-side streams (per-material and per-texture identity lines, "
                   "scene summaries, probe quad) are in settings.cfg [Remix] and the launcher's Testing tab.");
    }

    ScopedCpuProfileZone();
    m_pReplacer->initialize(ctx);
  }

  void SceneManager::logStatistics() {
    if (m_opacityMicromapManager.get()) {
      m_opacityMicromapManager->logStatistics();
    }
  }

  Vector3 SceneManager::getSceneUp() {
    return RtxOptions::zUp() ? Vector3(0.f, 0.f, 1.f) : Vector3(0.f, 1.f, 0.f);
  }

  Vector3 SceneManager::getSceneForward() {
    return RtxOptions::zUp() ? Vector3(0.f, 1.f, 0.f) : Vector3(0.f, 0.f, 1.f);
  }

  Vector3 SceneManager::calculateSceneRight() {
    const Vector3 up = SceneManager::getSceneUp();
    const Vector3 forward = SceneManager::getSceneForward();
    return RtxOptions::leftHandedCoordinateSystem() ? cross(up, forward) : cross(forward, up);
  }

  Vector3 SceneManager::worldToSceneOrientedVector(const Vector3& worldVector) {
    return RtxOptions::zUp() ? worldVector : Vector3(worldVector.x, worldVector.z, worldVector.y);
  }

  Vector3 SceneManager::sceneToWorldOrientedVector(const Vector3& sceneVector) {
    // Same transform applies to and from
    return worldToSceneOrientedVector(sceneVector);
  }

  float SceneManager::getTotalMipBias() {
    auto& resourceManager = m_device->getCommon()->getResources();
  
    const bool temporalUpscaling = RtxOptions::isDLSSOrRayReconstructionEnabled() || RtxOptions::isXeSSEnabled() || RtxOptions::isFSREnabled() || RtxOptions::isTAAEnabled();
    
    float totalUpscaleMipBias = 0.0f;
    
    if (temporalUpscaling) {
      if (RtxOptions::isXeSSEnabled()) {
        // XeSS uses the new formula from the XeSS developer guide
        totalUpscaleMipBias = -log2(resourceManager.getUpscaleRatio());
        
        // Add XeSS-specific mip bias when XeSS is active
        DxvkXeSS& xess = m_device->getCommon()->metaXeSS();
        if (xess.isActive()) {
          float xessMipBias = xess.calcRecommendedMipBias();
          totalUpscaleMipBias += xessMipBias;
        }
      } else if (RtxOptions::isFSREnabled()) {
        totalUpscaleMipBias = fork_hooks::fsrUpscalingMipBias(m_device);
      } else {
        // Restore original behavior for DLSS, TAA, and other upscalers
        totalUpscaleMipBias = log2(resourceManager.getUpscaleRatio()) + RtxOptions::upscalingMipBias();
      }
    }
    
    return totalUpscaleMipBias + RtxOptions::nativeMipBias();
  }

  float SceneManager::getCalculatedUpscalingMipBias() {
    auto& resourceManager = m_device->getCommon()->getResources();
    
    const bool temporalUpscaling = RtxOptions::isXeSSEnabled() || RtxOptions::isFSREnabled();
    if (!temporalUpscaling) {
      return 0.0f;
    }
    
    float calculatedUpscalingBias = -log2(resourceManager.getUpscaleRatio());
    return calculatedUpscalingBias;
  }

  void SceneManager::clear(Rc<DxvkContext> ctx, bool needWfi) {
    ScopedCpuProfileZone();

    auto& textureManager = m_device->getCommon()->getTextureManager();

    // Only clear once after the scene disappears, to avoid adding a WFI on every frame through clear().
    if (needWfi) {
      if (ctx.ptr())
        ctx->flushCommandList();
      m_device->waitForIdle();
    }

    // We still need to clear caches even if the scene wasn't rendered
    m_bufferCache.clear();
    m_preCreationSurfaceMaterialMap.clear();
    m_volumeMaterialCache.clear();

    // Clear ReplacementInstances first: their destructors call clear() which
    // accesses prims[] to mark entities for GC and clear back-pointers.
    // Entities must still be alive at this point.
    m_drawCallTracker.clear();

    // Called before instance manager's clear, so that it resets all tracked instances in Opacity Micromap manager at once
    if (m_opacityMicromapManager.get())
      m_opacityMicromapManager->clear();

    // Invalidate AccelManager's bucket cache before InstanceManager::clear() deletes
    // every RtInstance. The cache holds raw RtInstance* in m_cachedBuckets[].instances /
    // .surfaces and m_instanceBucketIndex, and the next frame's mergeInstancesIntoBlas
    // dirty check would dereference those (now-freed) pointers. The per-instance
    // onInstanceDestroyed -> removeInstanceFromBucketCache hook only patches the index
    // map, not the vectors, so a bulk reset must drop the cache wholesale.
    m_accelManager.clear();

    // Instance destruction fires onInstanceDestroyed -> releaseSurfaceMaterial for every
    // game-submitted instance, decrementing texture ref counts and feature counts.
    // The surface material cache must still be valid here so those releases are not
    // silently skipped by the bounds check in releaseSurfaceMaterial.
    m_instanceManager.clear();

    // After all instances are destroyed their retain/release pairs must be balanced.
    // If any count is non-zero here there is a retain/release mismatch.
    if (m_activePOMCount != 0) {
      Logger::err(str::format("[RTX] SceneManager::clear: POM count is ", m_activePOMCount, " after instance destruction (retain/release mismatch)"));
      assert(false && "SceneManager::clear: POM count non-zero after instance destruction");
      m_activePOMCount = 0;
    }
    if (m_sssCount != 0) {
      Logger::err(str::format("[RTX] SceneManager::clear: SSS count is ", m_sssCount, " after instance destruction (retain/release mismatch)"));
      assert(false && "SceneManager::clear: SSS count non-zero after instance destruction");
      m_sssCount = 0;
    }
    if (m_thinOpaqueCount != 0) {
      Logger::err(str::format("[RTX] SceneManager::clear: thin-opaque count is ", m_thinOpaqueCount, " after instance destruction (retain/release mismatch)"));
      assert(false && "SceneManager::clear: thin-opaque count non-zero after instance destruction");
      m_thinOpaqueCount = 0;
    }
    textureManager.assertAllRefCountsZero();

    // Now safe to clear material caches; all ref counts have been released.
    m_surfaceMaterialCache.clear();
    m_surfaceMaterialExtensionCache.clear();
    m_lightManager.clear();
    m_graphManager.clear();
    m_rayPortalManager.clear();
    m_drawCallCache.clear();
    textureManager.clear();
    m_textureCacheGenerationValidForPreserve = textureManager.getTextureCacheGeneration();

    m_previousFrameSceneAvailable = false;
    m_startInMediumMaterialIndex = SURFACE_INDEX_INVALID;
    m_fogStartInMediumMaterialIndex_inCache = kInvalidMaterialCacheIndex;
    m_externalStartInMediumMaterialIndex_inCache = kInvalidMaterialCacheIndex;
    m_startInMediumMaterialIndex_inCache = kInvalidMaterialCacheIndex;
    m_lastResolvedStartInMediumMaterialIndexInCache = kInvalidMaterialCacheIndex;
    m_lastUploadedStartInMediumMaterialIndexInCache = kInvalidMaterialCacheIndex;
  }

  void SceneManager::garbageCollection() {
    ScopedCpuProfileZone();

    // BlasEntry GC: remove entries not touched recently.
    // Only GC entries with no linked instances -- instances still reference the BlasEntry
    // for TLAS build, and destroying it would cause a one-frame visibility gap.
    if (m_device->getCurrentFrameId() > RtxOptions::numFramesToKeepGeometryData()) {
      const size_t oldestFrame = m_device->getCurrentFrameId() - RtxOptions::numFramesToKeepGeometryData();
      auto& entries = m_drawCallCache.getEntries();
      for (auto iter = entries.begin(); iter != entries.end(); ) {
        if (iter->second.frameLastTouched < oldestFrame &&
            iter->second.getLinkedInstances().empty()) {
          iter = entries.erase(iter);
        } else {
          ++iter;
        }
      }
    }

    // ReplacementInstance GC: marks owned instances/lights for GC
    // and clears their back-pointers while they are still alive.
    m_drawCallTracker.garbageCollectReplacementInstances(
        getCamera(), m_isAntiCullingSupported);

    // Instance/light GC: removes entities marked for GC by ReplacementInstance::clear()
    // or marked on creation (ephemeral copies). Back-pointers are already null.
    m_instanceManager.garbageCollection();
    m_accelManager.garbageCollection();
    m_lightManager.garbageCollection(getCamera());
    m_rayPortalManager.garbageCollection();
  }

  void SceneManager::onDestroy() {
    m_accelManager.onDestroy();
    if (m_opacityMicromapManager) {
      m_opacityMicromapManager->onDestroy();
    }
  }

  template<bool isNew>
  SceneManager::ObjectCacheState SceneManager::processGeometryInfo(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, RaytraceGeometry& inOutGeometry) {
    ScopedCpuProfileZone();
    ObjectCacheState result = ObjectCacheState::KBuildBVH;
    const RasterGeometry& input = drawCallState.getGeometryData();

    // Determine the optimal object state for this geometry
    if (!isNew) {
      // This is a geometry we've seen before, that requires updating
      //  'inOutGeometry' has valid historical data
      if (input.hashes[HashComponents::Indices] == inOutGeometry.hashes[HashComponents::Indices]) {
        // Check if the vertex positions have changed, requiring a BVH refit
        if (input.hashes[HashComponents::VertexPosition] == inOutGeometry.hashes[HashComponents::VertexPosition]
         && input.hashes[HashComponents::VertexShader] == inOutGeometry.hashes[HashComponents::VertexShader]
         && drawCallState.getSkinningState().boneHash == inOutGeometry.lastBoneHash) {
          result = ObjectCacheState::kUpdateInstance;
        } else {
          result = ObjectCacheState::kUpdateBVH;
        }
      }
    }

    RaytraceGeometry& output = inOutGeometry;

    output.lastBoneHash = drawCallState.getSkinningState().boneHash;

    // Update draw parameters
    output.cullMode = drawCallState.getCullMode();
    output.frontFace = input.frontFace;

    // Copy the hashes over
    output.hashes = input.hashes;

    if (!input.positionBuffer.defined()) {
      ONCE(Logger::err("processGeometryInfo: no position data on input detected"));
      return ObjectCacheState::kInvalid;
    }

    if (input.vertexCount == 0) {
      ONCE(Logger::err("processGeometryInfo: input data is violating some assumptions"));
      return ObjectCacheState::kInvalid;
    }

    // Set to 1 if inspection of the GeometryData structures contents on CPU is desired
    #define DEBUG_GEOMETRY_MEMORY 0
    constexpr VkMemoryPropertyFlags memoryProperty = DEBUG_GEOMETRY_MEMORY ? (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Assume we won't need this, and update the value if required
    output.previousPositionBuffer = RaytraceBuffer();

    // When the SmoothNormals category is set and the input has no normals, force the interleaved
    // vertex layout to include space for normals. The smooth normals compute pass will fill them in later.
    const bool needsSmoothNormals = drawCallState.getCategoryFlags().test(InstanceCategories::SmoothNormals);
    const bool forceNormals = needsSmoothNormals && !input.normalBuffer.defined();

    // When smooth normals state changes (added or removed), promote to kUpdateBVH so the vertex
    // data is re-interleaved and the smooth normals dispatch runs (or original normals are restored).
    if (needsSmoothNormals != output.smoothNormalsApplied && result == ObjectCacheState::kUpdateInstance) {
      result = ObjectCacheState::kUpdateBVH;
    }
    if (!needsSmoothNormals) {
      output.smoothNormalsApplied = false;
    }

    // If forceNormals is true, we can't use the fast "already interleaved" path since
    // we need to change the layout to include normal space.
    const size_t vertexStride = (input.isVertexDataInterleaved() && input.areFormatsGpuFriendly() && !forceNormals)
      ? input.positionBuffer.stride()
      : RtxGeometryUtils::computeOptimalVertexStride(input, forceNormals);

    switch (result) {
      case ObjectCacheState::KBuildBVH: {
        // Set up the ideal vertex params, if input vertices are interleaved, it's safe to assume the positionBuffer stride is the vertex stride
        output.vertexCount = input.vertexCount;

        const size_t vertexBufferSize = output.vertexCount * vertexStride;

        // Set up the ideal index params
        output.indexCount = input.isTopologyRaytraceReady() ? input.indexCount : RtxGeometryUtils::getOptimalTriangleListSize(input);
        const VkIndexType indexBufferType = input.isTopologyRaytraceReady() ? input.indexBuffer.indexType() : RtxGeometryUtils::getOptimalIndexFormat(output.vertexCount);
        const size_t indexStride = (indexBufferType == VK_INDEX_TYPE_UINT16) ? 2 : 4;

        // Make sure we're not stomping something else...
        assert(output.indexCacheBuffer == nullptr && output.historyBuffer[0] == nullptr);

        // Create a index buffer and vertex buffer we can use for raytracing.
        DxvkBufferCreateInfo info;
        info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
        info.access = VK_ACCESS_TRANSFER_WRITE_BIT;

        info.size = align(output.indexCount * indexStride, CACHE_LINE_SIZE);
        output.indexCacheBuffer = m_device->createBuffer(info, memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Index Cache Buffer");

        if (!RtxGeometryUtils::cacheIndexDataOnGPU(ctx, input, output)) {
          ONCE(Logger::err("processGeometryInfo: failed to cache index data on GPU"));
          return ObjectCacheState::kInvalid;
        }

        output.indexBuffer = RaytraceBuffer(DxvkBufferSlice(output.indexCacheBuffer), 0, indexStride, indexBufferType);

        info.size = align(vertexBufferSize, CACHE_LINE_SIZE);
        output.historyBuffer[0] = m_device->createBuffer(info, memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Geometry Buffer");

        RtxGeometryUtils::cacheVertexDataOnGPU(ctx, input, output, forceNormals);

        break;
      }
      case ObjectCacheState::kUpdateBVH: {
        bool invalidateHistory = false;

        // Stride changed, so we must recreate the previous buffer and use identical data
        if (output.historyBuffer[0]->info().size != align(vertexStride * input.vertexCount, CACHE_LINE_SIZE)) {
          auto desc = output.historyBuffer[0]->info();
          desc.size = align(vertexStride * input.vertexCount, CACHE_LINE_SIZE);
          output.historyBuffer[0] = m_device->createBuffer(desc, memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Geometry Buffer");

          // Invalidate the current buffer
          output.historyBuffer[1] = nullptr;

          // Mark this object for realignment
          invalidateHistory = true;
        }

        // Use the previous updates vertex data for previous position lookup
        std::swap(output.historyBuffer[0], output.historyBuffer[1]);

        if (output.historyBuffer[0].ptr() == nullptr) {
          // First frame this object has been dynamic need to allocate a 2nd frame of data to preserve history.
          output.historyBuffer[0] = m_device->createBuffer(output.historyBuffer[1]->info(), memoryProperty, DxvkMemoryStats::Category::RTXAccelerationStructure, "Geometry Buffer");
        } 

        RtxGeometryUtils::cacheVertexDataOnGPU(ctx, input, output, forceNormals);

        // Sometimes, we need to invalidate history, do that here by copying the current buffer to the previous..
        if (invalidateHistory) {
          ctx->copyBuffer(output.historyBuffer[1], 0, output.historyBuffer[0], 0, output.historyBuffer[1]->info().size);
        }

        // Assign the previous buffer using the last slice (copy most params from the position, just change buffer)
        output.previousPositionBuffer = RaytraceBuffer(DxvkBufferSlice(output.historyBuffer[1], 0, output.positionBuffer.length()), output.positionBuffer.offsetFromSlice(), output.positionBuffer.stride(), output.positionBuffer.vertexFormat());
        break;
      }
      default:
        break;
    }

    // Update color buffer in BVH with DrawCallState
    // The user can disable/enable color buffer for specific materials, so we manually sync the DrawCallState and BVH here to keep the color buffer in BVH updated.
    // Note, we don't setup kUpdateBVH because it's too waste to update all buffers if only the color buffer needs to be updated.
    if (output.color0Buffer.defined() && !drawCallState.getGeometryData().color0Buffer.defined()) {
      // Remove the color buffer in BVH if the color buffer from drawcall is removed by ignoreBakedLighting
      output.color0Buffer = RaytraceBuffer();
    } else if (!output.color0Buffer.defined() && drawCallState.getGeometryData().color0Buffer.defined()) {
      // Write the color buffer back to BVH if the color buffer is enabled again
      const DxvkBufferSlice slice = DxvkBufferSlice(output.historyBuffer[0]);
      const auto& colorBuffer = drawCallState.getGeometryData().color0Buffer;
      output.color0Buffer = RaytraceBuffer(slice, colorBuffer.offsetFromSlice(), colorBuffer.stride(), colorBuffer.vertexFormat());
    }

    // Update buffers in the cache
    updateBufferCache(output);

    return result;
  }


  void SceneManager::onFrameEnd(Rc<DxvkContext> ctx, bool raytracedThisFrame) {
    ScopedCpuProfileZone();

    // Commit this frame's texture registrations for preserve next frame. Must run before
    // manageTextureVram(), which may clear the cache and bump the generation so the
    // following frame takes the dynamic path for every draw call.
    m_textureCacheGenerationValidForPreserve =
        m_device->getCommon()->getTextureManager().getTextureCacheGeneration();

    manageTextureVram();

    // Release geometry belonging to meshes destroyed a few frames ago, now that nothing still in flight can
    // reference it. Deferred rather than freed on destroy because ray tracing reads a geometry's index buffer
    // and the previous frame's TLAS may still reach it.
    m_pReplacer->releaseRetiredExternalMeshes(m_device->getCurrentFrameId());

    if (m_enqueueDelayedClear || m_pReplacer->checkForChanges(ctx)) {
      clear(ctx, true);
      m_enqueueDelayedClear = false;
    }

    m_cameraManager.onFrameEnd();
    m_instanceManager.onFrameEnd();
    m_previousFrameSceneAvailable = raytracedThisFrame && RtxOptions::enablePreviousTLAS();

    m_bufferCache.clear();
    if (raytracedThisFrame) {
      std::lock_guard lock { m_drawCallMeta.mutex };
      const uint8_t curTick = m_drawCallMeta.ticker;
      const uint8_t nextTick = (m_drawCallMeta.ticker + 1) % m_drawCallMeta.MaxTicks;

      m_drawCallMeta.ready[curTick] = true;

      m_drawCallMeta.infos[nextTick].clear();
      m_drawCallMeta.ready[nextTick] = false;
      m_drawCallMeta.ticker = nextTick;
    }

    m_terrainBaker->onFrameEnd(ctx);

    if (m_opacityMicromapManager) {
      m_opacityMicromapManager->onFrameEnd();
    }
    
    m_startInMediumMaterialIndex = SURFACE_INDEX_INVALID;
    m_fogStartInMediumMaterialIndex_inCache = UINT32_MAX;
    m_startInMediumMaterialIndex_inCache = UINT32_MAX;

    if (m_uniqueObjectSearchDistance != RtxOptions::uniqueObjectDistance()) {
      m_uniqueObjectSearchDistance = RtxOptions::uniqueObjectDistance();
      m_drawCallTracker.rebuildSpatialMaps(m_uniqueObjectSearchDistance * 2.f);
    }

    // Not currently safe to cache these across frames (due to texture indices and rtx options potentially changing)
    m_preCreationSurfaceMaterialMap.clear();


    // execute graph updates after all garbage collection is complete (to avoid updating graphs that will just be deleted)
    // RtxOptions will still be pending, so any changes to them will apply next frame.
    if (raytracedThisFrame){
      m_graphManager.update(ctx);
    }

    // Clear replacement material hashes before the next frame.  These are used by components, so must clear after graphManager updates.
    clearFrameReplacementMaterialHashes();
    
    // Clear mesh hashes before the next frame.  These are used by components, so must clear after graphManager updates.
    clearFrameMeshHashes();
    
    // Reset the fog state to get it re-discovered on the next frame
    ImGUI::SetFogStates(m_fogStates, m_fog.getHash());
    m_fog = FogState();
    m_fogStates.clear();
    
    // Any drawcall translation invalidation has been consumed by this point. Clear the flag before new dirty options are processed.
    RtxOptionManager::clearDrawcallTranslationInvalid();
  }

  std::unordered_set<XXH64_hash_t> uniqueHashes;


  void SceneManager::submitDrawState(Rc<DxvkContext> ctx, const DrawCallState& input, const MaterialData* overrideMaterialData) {
    ScopedCpuProfileZone();
    if (m_bufferCache.getTotalCount() >= kBufferCacheLimit && m_bufferCache.getActiveCount() >= kBufferCacheLimit) {
      ONCE(Logger::info("[RTX-Compatibility-Info] This application is pushing more unique buffers than is currently supported - some objects may not raytrace."));
      return;
    }

    if (input.getFogState().mode != D3DFOG_NONE) {
      XXH64_hash_t fogHash = input.getFogState().getHash();
      if (m_fogStates.find(fogHash) == m_fogStates.end()) {
        // Only do anything if we haven't seen this fog before.
        m_fogStates[fogHash] = input.getFogState();

        MaterialData* pFogReplacement = m_pReplacer->getReplacementMaterial(fogHash);
        if (pFogReplacement) {
          // Track this replacement material hash for hash checking
          trackReplacementMaterialHash(fogHash);
          // Fog has been replaced by a translucent material to start the camera in,
          // meaning that it was being used to indicate 'underwater' or something similar.
          if (pFogReplacement->getType() != MaterialDataType::Translucent) {
            Logger::warn(str::format("Fog replacement materials must be translucent.  Ignoring material for ", std::hex, m_fog.getHash()));
          } else {
            uint32_t id = UINT32_MAX;
            createSurfaceMaterial(*pFogReplacement, input, &id);
            assert(id != UINT32_MAX);
            m_fogStartInMediumMaterialIndex_inCache = id;
          }
        } else if (m_fog.mode == D3DFOG_NONE) {
          // render the first unreplaced fog.
          m_fog = input.getFogState();
        }
      }
    }


    const XXH64_hash_t activeReplacementHash = input.getHash(RtxOptions::geometryAssetHashRule());
    
    // Track this mesh hash for mesh hash checking
    trackMeshHash(activeReplacementHash);

    std::vector<AssetReplacement>* pReplacements = m_pReplacer->getReplacementsForMesh(activeReplacementHash);

    // TODO (REMIX-656): Remove this once we can transition content to new hash
    if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash0) == rules::LegacyAssetHash0) {
      if (!pReplacements) {
        const XXH64_hash_t legacyHash = input.getHashLegacy(rules::LegacyAssetHash0);
        trackMeshHash(legacyHash);
        pReplacements = m_pReplacer->getReplacementsForMesh(legacyHash);
        if (RtxOptions::logLegacyHashReplacementMatches() && pReplacements && uniqueHashes.find(legacyHash) == uniqueHashes.end()) {
          uniqueHashes.insert(legacyHash);
          Logger::info(str::format("[Legacy-Hash-Replacement] Found a mesh referenced from legacyHash0: ", std::hex, legacyHash, ", new hash: ", std::hex, activeReplacementHash));
        }
      }
    }

    if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash1) == rules::LegacyAssetHash1) {
      if (!pReplacements) {
        const XXH64_hash_t legacyHash = input.getHashLegacy(rules::LegacyAssetHash1);
        trackMeshHash(legacyHash);
        pReplacements = m_pReplacer->getReplacementsForMesh(legacyHash);
        if (RtxOptions::logLegacyHashReplacementMatches() && pReplacements && uniqueHashes.find(legacyHash) == uniqueHashes.end()) {
          uniqueHashes.insert(legacyHash);
          Logger::info(str::format("[Legacy-Hash-Replacement] Found a mesh referenced from legacyHash1: ", std::hex, legacyHash, ", new hash: ", std::hex, activeReplacementHash));
        }
      }
    }

    ReplacementInstance* replacementInstance = m_drawCallTracker.findOrCreateReplacementInstance(
        input, m_rayPortalManager, overrideMaterialData);

    const uint32_t currentFrameId = m_device->getCurrentFrameId();
    const bool secondSubmissionThisFrame = (replacementInstance->frameLastSeen == currentFrameId);

    // Preserve path: L1 means this draw still matches the same ReplacementInstance by identity; replacer reload is
    // expected to clear the draw-call tracker (see SceneManager::clear), so we do not re-verify mesh prims or
    // activeReplacements pointers every frame. If a path exists where replacements bind without a clear, use dynamic
    // (drawReplacements) for that transition -- drawReplacements already reconciles activeReplacements and prims.
    //
    // Static path reuses each prim's BlasEntry::modifiedGeometryData as-is. If another draw earlier this frame
    // already entered DrawCallCache::get and re-bound a sibling-topology BlasEntry to its own data (kUpdateBVH),
    // the cached buffers no longer correspond to this draw -- fall back to dynamic so DrawCallCache::get's
    // "frameLastTouched skip" allocates a fresh BlasEntry and processSceneObject re-links the instance.
    auto blasAlreadyTouchedByOtherDraw = [replacementInstance, currentFrameId]() -> bool {
      for (const auto& prim : replacementInstance->prims) {
        RtInstance* inst = prim.getInstance();
        if (inst == nullptr) {
          continue;
        }
        BlasEntry* pBlas = inst->getBlas();
        if (pBlas == nullptr) {
          continue;
        }
        if (pBlas->frameLastTouched == currentFrameId) {
          return true;
        }
      }
      return false;
    };

    // Per-frame override materials (terrain bake, etc.) can introduce particle systems
    // without a prior dynamic update on this RI.
    const bool overrideMaterialHasParticles = overrideMaterialData != nullptr
        && overrideMaterialData->getParticleSystemDesc() != nullptr;

    // The RI's prims must already be wired up for this exact replacements vector. drawReplacements
    // re-initializes prims when activeReplacements changes (e.g. async replacement load completes
    // after the RI was created without replacements, or hot-reload changes the replacement set).
    // The preserve path has no equivalent reinitialization, so fall back to dynamic for that transition.
    const bool activeReplacementsMatch =
        replacementInstance->activeReplacements == pReplacements;

    // Terrain draws share a per-frame override OpaqueMaterialData built from the
    // TerrainBaker cascade set. Cascade images don't carry a stable identity hash,
    // so MaterialData::getHash() doesn't change when the cascade set grows or shrinks
    // (e.g. when the normal cascade is first added on top of an albedo-only set on
    // an earlier frame). The preserve path would then reuse the surfaceMaterialIndex
    // computed before the new cascade textures existed, leaving the secondary
    // texture slots invalid. Force the dynamic path on any frame in which a cascade
    // image was created or resized; subsequent stable frames take the preserve path
    // with the freshly built RtSurfaceMaterial.
    const bool terrainCascadesJustChanged =
        input.getCategoryFlags().test(InstanceCategories::Terrain) &&
        m_terrainBaker->cascadeCompositionChangedThisFrame();

    const bool cachedTexturesValidForPreserve =
        m_device->getCommon()->getTextureManager().getTextureCacheGeneration() ==
        m_textureCacheGenerationValidForPreserve;

    const XXH64_hash_t legacyMaterialIdentityHash = input.getMaterialData().computeIdentityHash();
    const bool legacyMaterialIdentityHashMatch =
        replacementInstance->legacyMaterialIdentityHash == legacyMaterialIdentityHash;

    const bool usePreservePath =
        RtxOptions::enablePreservePath() &&
        replacementInstance->dirtyFlags.isClear() &&
        !RtxOptionManager::isDrawcallTranslationInvalid() &&
        !secondSubmissionThisFrame &&
        !input.getCategoryFlags().test(InstanceCategories::ParticleEmitter) &&
        !RtxOptions::shouldConvertToLight(input.getMaterialData().getHash()) &&
        !blasAlreadyTouchedByOtherDraw() &&
        !overrideMaterialHasParticles&&
        activeReplacementsMatch &&
        legacyMaterialIdentityHashMatch &&
        !terrainCascadesJustChanged &&
        cachedTexturesValidForPreserve;


    if (usePreservePath) {
      preserveReplacementInstance(ctx, input, pReplacements, replacementInstance);
    } else {
      MaterialData renderMaterialData = determineMaterialData(overrideMaterialData, input);
      if (!activeReplacementsMatch) {
        replacementInstance->clear();
      }

      // Recompute dynamic-feature bits on each dynamic update.
      replacementInstance->dirtyFlags.clr(ReplacementInstance::kDynamicFeatureMask);

      // Create / process: full geometry cache and instance update.
      if (pReplacements != nullptr) {
        drawReplacements(ctx, &input, pReplacements, renderMaterialData, replacementInstance);
      } else {
        RtInstance* existingInstance = (replacementInstance->prims.size() > 0)
            ? replacementInstance->prims[0].getInstance() : nullptr;

        RtInstance* instance = processDrawCallState(ctx, input, renderMaterialData,
            *replacementInstance, existingInstance, nullptr);
        if (instance != nullptr) {
          if (replacementInstance->root.getUntyped() == nullptr) {
            replacementInstance->setup(PrimInstance(instance, PrimInstance::Type::Instance), 1, nullptr);
          }
          if (replacementInstance->prims[0].getUntyped() != instance) {
            instance->getPrimInstanceOwner().setReplacementInstance(replacementInstance, 0, instance,
                PrimInstance::Type::Instance);
          }
        }
      }
      replacementInstance->legacyMaterialIdentityHash = legacyMaterialIdentityHash;
    }

    replacementInstance->frameLastSeen = currentFrameId;
    replacementInstance->categoryFlags = input.getCategoryFlags().raw();
    replacementInstance->isSkinned = input.getSkinningState().numBones > 0;

    // Cache this submission's texture-coordinate projection so that next frame's
    // computeDirtyFlags can detect drift. Writing here (after either path has run)
    // mirrors how objectToWorld is updated downstream of the dirty-flag check.
    replacementInstance->textureTransform = input.getTransformData().textureTransform;
    replacementInstance->texgenMode = input.getTransformData().texgenMode;

    // For standalone draw calls, store the object-space bounding box for anti-culling.
    // For replacement draw calls, the aggregate AABB is computed inside drawReplacements.
    if (pReplacements == nullptr) {
      const auto& geoBBox = input.getGeometryData().boundingBox;
      if (geoBBox.isValid()) {
        replacementInstance->geometryBoundingBox = geoBBox;
        replacementInstance->objectToWorld = input.getTransformData().objectToWorld;
      }
    }
  }

  MaterialData SceneManager::determineMaterialData(const MaterialData* overrideMaterialData, const DrawCallState& input) {
    ScopedCpuProfileZone();
    // First see if we have an explicit override
    if (overrideMaterialData != nullptr) {
      return *overrideMaterialData;
    } 

    // test if any direct material replacements exist
    MaterialData* pReplacementMaterial = m_pReplacer->getReplacementMaterial(input.getMaterialData().getHash());
    if (pReplacementMaterial != nullptr) {
      // Make a copy - dont modify the replacement data.
      MaterialData renderMaterialData = *pReplacementMaterial;
      // merge in the input material from game
      renderMaterialData.mergeLegacyMaterial(input.getMaterialData());
      return renderMaterialData;
    }

    // Check if a Ray Portal override is needed
    size_t rayPortalTextureIndex;
    if (RtxOptions::getRayPortalTextureIndex(input.getMaterialData().getHash(), rayPortalTextureIndex)) {
      assert(rayPortalTextureIndex < maxRayPortalCount);
      assert(rayPortalTextureIndex < (std::numeric_limits<uint8_t>::max)());

      MaterialData renderMaterialData = input.getMaterialData().as<RayPortalMaterialData>();
      renderMaterialData.getRayPortalMaterialData().setRayPortalIndex(rayPortalTextureIndex);
      return renderMaterialData;
    }

    // Standard legacy material conversion
    return input.getMaterialData().as<OpaqueMaterialData>();
  }

  void SceneManager::createEffectLight(Rc<DxvkContext> ctx, const DrawCallState& input, const RtInstance* instance) {
    const float effectLightIntensity = RtxOptions::effectLightIntensity();
    if (effectLightIntensity <= 0.f)
      return;

    const RasterGeometry& geometryData = input.getGeometryData();

    const GeometryBufferData bufferData(geometryData);
    
    if (!bufferData.indexData && geometryData.indexCount > 0 || !bufferData.positionData)
      return;

    // Find centroid of point cloud.
    Vector3 centroid = Vector3();
    uint32_t counter = 0;
    if (geometryData.indexCount > 0) {
      for (uint32_t i = 0; i < geometryData.indexCount; i++) {
        const uint16_t index = bufferData.getIndex(i);
        centroid += bufferData.getPosition(index);
        ++counter;
      }
    } else {
      for (uint32_t i = 0; i < geometryData.vertexCount; i++) {
        centroid += bufferData.getPosition(i);
        ++counter;
      }
    }
    centroid /= (float) counter;
    
    const Vector4 renderingPos = input.getTransformData().objectToView * Vector4(centroid.x, centroid.y, centroid.z, 1.0f);
    // Note: False used in getViewToWorld since the renderingPos of the object is defined with respect to the game's object to view
    // matrix, not our freecam's, and as such we want to convert it back to world space using the matching matrix.
    const Vector4 worldPos{ getCamera().getViewToWorld(false) * Vector4d{ renderingPos } };

    RtLightShaping shaping{};

    float lightRadius = std::max(RtxOptions::effectLightRadius(), 1e-3f);
    const Vector3 lightPosition { worldPos.x, worldPos.y, worldPos.z };
    Vector3 lightRadiance;
    if (RtxOptions::effectLightPlasmaBall()) {
      // Todo: Make these options more configurable via config options.
      const double timeMilliseconds = static_cast<double>(GlobalTime::get().absoluteTimeMs());
      const double animationPhase = sin(timeMilliseconds * 0.006) * 0.5 + 0.5;
      lightRadiance = lerp(Vector3(1.f, 0.921f, 0.738f), Vector3(1.f, 0.521f, 0.238f), animationPhase);
    } else {
      const D3DCOLORVALUE originalColor = input.getMaterialData().getLegacyMaterial().Diffuse;
      lightRadiance = Vector3(originalColor.r, originalColor.g, originalColor.b) * RtxOptions::effectLightColor();
    }
    const float surfaceArea = 4.f * kPi * lightRadius * lightRadius;
    const float radianceFactor = 1e5f * effectLightIntensity / surfaceArea;
    lightRadiance *= radianceFactor;

    RtLight rtLight(RtSphereLight(lightPosition, lightRadiance, lightRadius, shaping));
    rtLight.isDynamic = true;

    m_lightManager.addLight(rtLight, input, RtLightAntiCullingType::MeshReplacement);
  }

  std::optional<DrawCallState> SceneManager::buildReplacementMeshDrawCallState(
      const DrawCallState& input,
      const AssetReplacement& replacement) {
    if (replacement.includeOriginal) {
      auto newDrawCallState = std::make_optional<DrawCallState>(input);
      newDrawCallState->modifyCategoryFlags() = replacement.categories.applyCategoryFlags(newDrawCallState->getCategoryFlags());
      return newDrawCallState;
    }

    if (replacement.type == AssetReplacement::eMesh) {
      auto newDrawCallState = std::make_optional<DrawCallState>(input);

      DrawCallTransforms& transforms = newDrawCallState->modifyTransformData();
      transforms.objectToWorld = transforms.objectToWorld * replacement.replacementToObject;
      transforms.objectToView = transforms.objectToView * replacement.replacementToObject;
      if (replacement.instancesToObject && !replacement.instancesToObject->empty()) {
        transforms.instancesToObject = replacement.instancesToObject;
      } else {
        transforms.instancesToObject = nullptr;
      }
      // Mesh replacements don't support these.
      transforms.textureTransform = Matrix4();
      transforms.texgenMode = TexGenMode::None;

      newDrawCallState->overrideGeometryData(&replacement.geometry->data);
      newDrawCallState->modifyCategoryFlags() = replacement.categories.applyCategoryFlags(newDrawCallState->getCategoryFlags());
      return newDrawCallState;
    }

    return std::nullopt;
  }

  void SceneManager::drawReplacements(Rc<DxvkContext> ctx, const DrawCallState* input, const std::vector<AssetReplacement>* pReplacements, MaterialData& renderMaterialData, ReplacementInstance* replacementInstance) {
    ScopedCpuProfileZone();
    if (pReplacements == nullptr) {
      assert(false && "pReplacements should never be nullptr here");
      ONCE(Logger::err("pReplacements should never be nullptr in SceneManager::drawReplacements"));
      return;
    }
    // The game's own material, kept aside before the loop starts overwriting renderMaterialData.
    //
    // Needed as a stable base for the merge below. renderMaterialData is a reference that the loop
    // reassigns per replacement, so merging against it would fold each entry into the previous entry's
    // result rather than into the game's material.
    const MaterialData hostMaterialData = renderMaterialData;

    // If the index contains an RtInstance, get a pointer to it.
    auto getExistingInstance = [replacementInstance](size_t idx) -> RtInstance* {
      if (replacementInstance->prims.size() <= idx) {
        return nullptr;
      }
      return replacementInstance->prims[idx].getInstance();
    };

    for (size_t i = 0; i < pReplacements->size(); i++) {
      auto& replacement = (*pReplacements)[i];
      RtInstance* instance = nullptr;

      std::optional<DrawCallState> newDrawCallState = SceneManager::buildReplacementMeshDrawCallState(*input, replacement);
      if (newDrawCallState.has_value()) {
        // Note: Material Data replaced if a replacement is specified in the Mesh Replacement.
        // Only meaningful when geometry is replaced (eMesh); the includeOriginal branch keeps the
        // game's original material data.
        if (!replacement.includeOriginal && replacement.type == AssetReplacement::eMesh && replacement.materialData != nullptr) {
          // Merge the game's material underneath rather than assigning the replacement's wholesale.
          //
          // Most entries in a real pack are partial: an `over` that authors roughness, or a normal map, and
          // nothing else. Assigning such an entry directly hands the surface a material with no albedo
          // texture, which renders black -- measured at 9 of 40 replacements in one census office frame.
          // merge() takes each field the USD did not author from the argument, so the pack keeps what it
          // specified and the rest comes back from the game.
          //
          // This is the same correction already made on the material path in
          // fork_hooks::externalDrawMaterialReplacement, which documents having hit exactly this.
          // Both material types carry a merge(); gating on Opaque alone left translucent replacements
          // taking the replacement's material wholesale, which is the case this exists to avoid. The pack
          // authors glass and foliage against AperturePBR_Translucent, so those entries were still landing
          // on a material with no albedo and rendering blank.
          MaterialData merged = *replacement.materialData;
          if (merged.getType() == hostMaterialData.getType()) {
            if (merged.getType() == MaterialDataType::Opaque) {
              merged.getOpaqueMaterialData().merge(hostMaterialData.getOpaqueMaterialData());
            } else if (merged.getType() == MaterialDataType::Translucent) {
              merged.getTranslucentMaterialData().merge(hostMaterialData.getTranslucentMaterialData());
            }
          }
          renderMaterialData = merged;
        }

        const RtxParticleSystemDesc* pParticleSystemDesc = replacement.particleSystem.has_value() ? &replacement.particleSystem.value() : nullptr;
        instance = processDrawCallState(ctx, *newDrawCallState, renderMaterialData, *replacementInstance, getExistingInstance(i), pParticleSystemDesc);
      }

      if (instance != nullptr) {
        if (replacementInstance->root.getUntyped() == nullptr) {
          // This is the first time this replacementInstance is used, and the first mesh drawn
          //  as part of this replacementInstance, so invoke setup and set the root.
          replacementInstance->setup(PrimInstance(instance, PrimInstance::Type::Instance), pReplacements->size(), pReplacements);
          instance->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, instance, PrimInstance::Type::Instance);
        } else if (replacementInstance->prims[i].getUntyped() != instance) {
          instance->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, instance, PrimInstance::Type::Instance);
        }
      }
    }

    processReplacementLights(input, pReplacements, replacementInstance);
    processReplacementGraphs(ctx, input, pReplacements, replacementInstance);

    replacementInstance->recalculateBoundingBox(
        input->getTransformData().objectToWorld,
        &input->getGeometryData().boundingBox);
  }
  
  void SceneManager::processReplacementLights(
      const DrawCallState* input, const std::vector<AssetReplacement>* pReplacements,
      ReplacementInstance* replacementInstance) {
    ScopedCpuProfileZone();
    for (size_t i = 0; i < pReplacements->size(); i++) {
      auto&& replacement = (*pReplacements)[i];
      if (replacement.type == AssetReplacement::eLight) {
        if (replacementInstance->root.getUntyped() == nullptr) {
          Logger::err(str::format(
              "Light prims anchored to a mesh replacement must also include actual meshes.  mesh hash: ",
              std::hex, input->getHash(RtxOptions::geometryAssetHashRule())
          ));
          break;
        }
        if (replacement.lightData.has_value()) {
          RtLight objectSpaceLight = replacement.lightData->toRtLight();

          // Transform to world space for the actual light creation
          RtLight localLight = objectSpaceLight;
          localLight.applyTransform(input->getTransformData().objectToWorld);

          RtLight* existingLight = (replacementInstance->prims.size() > i)
              ? replacementInstance->prims[i].getLight() : nullptr;
          if (existingLight != nullptr) {
            if (existingLight->getPrimInstanceOwner().getReplacementInstance() != replacementInstance) {
              ONCE(assert(false && "light in a replacementInstance believes it is owned by a different replacementInstance."));
            }
            m_lightManager.updateExternallyTrackedLight(existingLight, localLight);
          } else {
            RtLight* newLight = m_lightManager.createExternallyTrackedLight(localLight);
            newLight->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, newLight, PrimInstance::Type::Light);
          }
        }
      }
    }
  }

  void SceneManager::processReplacementGraphs(
      Rc<DxvkContext> ctx, const DrawCallState* input, const std::vector<AssetReplacement>* pReplacements,
      ReplacementInstance* replacementInstance) {
    ScopedCpuProfileZone();
    for (size_t i = 0; i < pReplacements->size(); i++) {
      auto&& replacement = (*pReplacements)[i];
      if (replacement.type == AssetReplacement::eGraph) {
        bool hasGraph = (replacementInstance->prims.size() > i) &&
                        (replacementInstance->prims[i].getGraph() != nullptr);
        if (!hasGraph) {
          if (!replacement.graphState.has_value()) {
            Logger::err(str::format(
                "Graph prims missing graph state in mesh replacement.  mesh hash: ",
                std::hex, input->getHash(RtxOptions::geometryAssetHashRule())
            ));
            break;
          }
          GraphInstance* graphInstance = m_graphManager.addInstance(ctx, replacement.graphState.value());
          if (graphInstance) {
            graphInstance->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, graphInstance, PrimInstance::Type::Graph);
          }
        }
      }
    }
  }

  void SceneManager::updateBufferCache(RaytraceGeometry& newGeoData) {
    ScopedCpuProfileZone();
    if (newGeoData.indexBuffer.defined()) {
      newGeoData.indexBufferIndex = m_bufferCache.track(newGeoData.indexBuffer);
    } else {
      newGeoData.indexBufferIndex = kSurfaceInvalidBufferIndex;
    }

    if (newGeoData.normalBuffer.defined()) {
      newGeoData.normalBufferIndex = m_bufferCache.track(newGeoData.normalBuffer);
    } else {
      newGeoData.normalBufferIndex = kSurfaceInvalidBufferIndex;
    }

    if (newGeoData.color0Buffer.defined()) {
      newGeoData.color0BufferIndex = m_bufferCache.track(newGeoData.color0Buffer);
    } else {
      newGeoData.color0BufferIndex = kSurfaceInvalidBufferIndex;
    }

    if (newGeoData.texcoordBuffer.defined()) {
      newGeoData.texcoordBufferIndex = m_bufferCache.track(newGeoData.texcoordBuffer);
    } else {
      newGeoData.texcoordBufferIndex = kSurfaceInvalidBufferIndex;
    }

    if (newGeoData.positionBuffer.defined()) {
      newGeoData.positionBufferIndex = m_bufferCache.track(newGeoData.positionBuffer);
    } else {
      newGeoData.positionBufferIndex = kSurfaceInvalidBufferIndex;
    }

    if (newGeoData.previousPositionBuffer.defined()) {
      newGeoData.previousPositionBufferIndex = m_bufferCache.track(newGeoData.previousPositionBuffer);
    } else {
      newGeoData.previousPositionBufferIndex = kSurfaceInvalidBufferIndex;
    }
  }

  void SceneManager::preserveInstance(
      RtInstance& instance,
      const DrawCallState* pInput) {
    ScopedCpuProfileZone();
    BlasEntry* pBlas = instance.getBlas();
    if (pBlas == nullptr) {
      return;
    }

    instance.setFrameLastUpdated(m_device->getCurrentFrameId());

    // Preserve path: keep RtInstance surface/material/transform/mask state from the last dynamic update.
    // Only refresh per-frame buffer-cache indices (and BLAS touch / texture lifetime) so GPU addresses stay valid.

    // Preserve / anti-culled draw: positions are unchanged this frame, so there is no meaningful
    // previousPosition data. Clear it to match processGeometryInfo's kUpdateInstance behavior
    // (it always clears, then only kUpdateBVH re-points it at historyBuffer[1]); without this,
    // a stale buffer left over from the last kUpdateBVH frame would feed into motion vectors.
    pBlas->modifiedGeometryData.previousPositionBuffer = RaytraceBuffer();

    // The last dynamic update may have left prevObjectToWorld != objectToWorld and
    // isStatic == false (e.g. after a transform-changing move()). Re-sync on the first
    // preserve frame after that.
    if (!instance.surface.isStatic) {
      instance.surface.prevObjectToWorld = instance.surface.objectToWorld;
      instance.surface.isStatic = true;
    }

    // The last dynamic update may have left hasMaterialChanged == true.
    instance.surface.hasMaterialChanged = false;

    // Buffer indices are per-frame (m_bufferCache is cleared in onFrameEnd),
    // so re-register geometry buffers and copy fresh indices to the surface.
    updateBufferCache(pBlas->modifiedGeometryData);
    m_instanceManager.processInstanceBuffers(*pBlas, instance);

    pBlas->frameLastTouched = m_device->getCurrentFrameId();

    const uint32_t surfaceMatIdx = instance.surface.surfaceMaterialIndex;

    // Ray Portal refresh on the preserve path. RayPortalManager::clear() wipes m_rayPortalInfos
    // every frame in endFrame, so processRayPortalData must repopulate the slot for any portal
    // instance still present. The cached RtSurfaceMaterial supplies the RayPortalIndex and the
    // resource-cache key processRayPortalData needs.
    if (instance.getMaterialType() == MaterialDataType::RayPortal &&
        surfaceMatIdx < m_surfaceMaterialCache.getTotalCount()) {
      const RtSurfaceMaterial& surfaceMaterial = m_surfaceMaterialCache.getObjectTable()[surfaceMatIdx];
      if (surfaceMaterial.getType() == RtSurfaceMaterialType::RayPortal) {
        m_rayPortalManager.processRayPortalData(instance, surfaceMaterial);
      }
    }

    // m_billboards is cleared every frame in InstanceManager::onFrameEnd, so re-run the
    // billboard step to repopulate it and refresh m_firstBillboard / m_billboardCount for
    // this frame's unordered TLAS. pInput is null on the anti-culling preserve path;
    // those instances are off-screen and don't need portal-space billboards.
    if (pInput != nullptr && m_cameraManager.isCameraValid(CameraType::Main)) {
      m_instanceManager.refreshBillboardsForCurrentFrame(
          instance,
          pInput->cameraType,
          m_cameraManager.getMainCamera().getDirection(false));
    }
  }

  void SceneManager::syncPreservedReplacementMeshesState(
      const DrawCallState& input,
      const std::vector<AssetReplacement>* pReplacements,
      ReplacementInstance* replacementInstance) {
    if (pReplacements == nullptr) {
      return;
    }
    // Push this frame's per-replacement DrawCallState into each prim's BlasEntry::input.
    // Preserve path: RtInstance state is unchanged from the last dynamic update; only BLAS
    // input tracks remix state. The construction matches drawReplacements() (both share
    // buildReplacementMeshDrawCallState) so dynamic and preserve paths feed identical
    // DrawCallStates into the BlasEntry.
    for (size_t i = 0; i < pReplacements->size(); i++) {
      if (replacementInstance->prims.size() <= i) {
        break;
      }
      RtInstance* instance = replacementInstance->prims[i].getInstance();
      if (instance == nullptr) {
        continue;
      }
      BlasEntry* pBlas = instance->getBlas();
      if (pBlas == nullptr) {
        continue;
      }
      std::optional<DrawCallState> newDrawCallState =
          SceneManager::buildReplacementMeshDrawCallState(input, (*pReplacements)[i]);
      if (newDrawCallState.has_value()) {
        pBlas->input = *newDrawCallState;
      }
    }
  }

  void SceneManager::preserveReplacementInstance(
      Rc<DxvkContext> ctx,
      const DrawCallState& input,
      const std::vector<AssetReplacement>* pReplacements,
      ReplacementInstance* replacementInstance) {
    ScopedCpuProfileZone();
    // Refresh BlasEntry::input with this frame's draw state BEFORE dispatching preserveInstance.
    // refreshBillboardsForCurrentFrame -> createBeams / createBillboards consult
    // pBlas->input.getGeometryData() and call mapPtr() on its RasterBuffer slices to read
    // CPU-side positions/texcoords. If pBlas->input is left over from a previous frame, those
    // slices may reference vertex memory that has been recycled (notably with useSharedHeap=off
    // in the bridge), giving zero-fill positions and tripping the normalize() assert in
    // createBeams. Refreshing here keeps the dynamic and preserve paths feeding the same
    // frame's geometry into billboard / beam creation.
    if (pReplacements != nullptr) {
      replacementInstance->activeReplacements = pReplacements;
      syncPreservedReplacementMeshesState(input, pReplacements, replacementInstance);
    } else if (replacementInstance->prims.size() > 0) {
      RtInstance* inst = replacementInstance->prims[0].getInstance();
      if (inst != nullptr && inst->getBlas() != nullptr) {
        inst->getBlas()->input = input;
      }
    }

    // Re-register per-frame buffer cache indices (same order as dynamic path: buffers resolved first).
    // No MaterialData is threaded through: SceneManager::preserveInstance reads the cached
    // RtSurfaceMaterial via surfaceMaterialIndex (Ray Portals included), and InstanceManager
    // event handlers contract for a null material on the preserve path.
    preserveInstancesWithObjectPicking(
      replacementInstance->prims,
      input.drawCallID,
      [&](RtInstance& instance) {
        instance.surface.isPreservePath = true;
        preserveInstance(instance, &input);
        m_instanceManager.preserveInstance(instance, input, nullptr);
      },
      [&] {
        trackObjectPickingMeta(input, input.drawCallID);
      });

    replacementInstance->recalculateBoundingBox(
        input.getTransformData().objectToWorld, &input.getGeometryData().boundingBox);
  }

  SceneManager::ObjectCacheState SceneManager::onSceneObjectAdded(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas) {
    // This is a new object.
    ObjectCacheState result = processGeometryInfo<true>(ctx, drawCallState, pBlas->modifiedGeometryData);
    
    assert(result == ObjectCacheState::KBuildBVH);

    pBlas->frameLastUpdated = m_device->getCurrentFrameId();
    m_instanceManager.notifySceneChanged();

    return result;
  }
  
  SceneManager::ObjectCacheState SceneManager::onSceneObjectUpdated(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas) {
    if (pBlas->frameLastTouched == m_device->getCurrentFrameId()) {
      pBlas->cacheMaterial(drawCallState.getMaterialData());
      return SceneManager::ObjectCacheState::kUpdateInstance;
    }

    // TODO: If mesh is static, no need to do any of the below, just use the existing modifiedGeometryData and set result to kInstanceUpdate.
    ObjectCacheState result = processGeometryInfo<false>(ctx, drawCallState, pBlas->modifiedGeometryData);

    // We dont expect to hit the rebuild path here - since this would indicate an index buffer or other topological change, and that *should* trigger a new scene object (since the hash would change)
    assert(result != ObjectCacheState::KBuildBVH);

    if (result == ObjectCacheState::kUpdateBVH) {
      pBlas->frameLastUpdated = m_device->getCurrentFrameId();
      m_instanceManager.notifySceneChanged();
    }
    
    pBlas->clearMaterialCache();
    pBlas->input = drawCallState; // cache the draw state for the next time.
    return result;
  }
  
  void SceneManager::onInstanceAdded(RtInstance& instance) {
    BlasEntry* pBlas = instance.getBlas();
    if (pBlas != nullptr) {
      pBlas->linkInstance(&instance);
    }
  }

  void SceneManager::onInstanceUpdated(RtInstance& instance, const DrawCallState& drawCall, const MaterialData* material, const bool hasTransformChanged, const bool hasVerticesChanged, const bool isFirstUpdateThisFrame) {
    // The preserve path passes a null material and reuses RtSurfaceMaterial via
    // RtInstance::surface.surfaceMaterialIndex; the dynamic path always provides a material.
    if (instance.surface.isPreservePath) {
      assert(material == nullptr);
      return;
    }
    assert(material != nullptr);
    auto capturer = m_device->getCommon()->capturer();
    if (hasTransformChanged) {
      capturer->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::XformUpdate);
    }

    if (hasVerticesChanged) {
      capturer->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::PositionsUpdate);
      capturer->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::NormalsUpdate);
    }
    
    // Create and bind the RT material
    uint32_t newMatIdx = kInvalidMaterialCacheIndex;
    const RtSurfaceMaterial& surfaceMaterial = createSurfaceMaterial(*material, drawCall, &newMatIdx);

    if(isFirstUpdateThisFrame) {
      const uint32_t oldMatIdx = instance.surface.surfaceMaterialIndex;
      m_instanceManager.bindMaterial(instance, surfaceMaterial);
      if (newMatIdx != oldMatIdx) {
        retainSurfaceMaterial(newMatIdx);
        releaseSurfaceMaterial(oldMatIdx);
      }
    }

    // Update portal
    if (surfaceMaterial.getType() == RtSurfaceMaterialType::RayPortal) {
      m_rayPortalManager.processRayPortalData(instance, surfaceMaterial);
    }
  }

  void SceneManager::retainSurfaceMaterial(uint32_t matIdx) {
    // Retain is always called with newMatIdx from createSurfaceMaterial, which must return
    // a valid in-bounds index. An out-of-bounds index here indicates a logic error upstream.
    if (matIdx >= m_surfaceMaterialCache.getTotalCount()) {
      Logger::err(str::format("[RTX] retainSurfaceMaterial: matIdx ", matIdx,
        " out of bounds (cache size ", m_surfaceMaterialCache.getTotalCount(),
        "; createSurfaceMaterial returned invalid index"));
      assert(false && "retainSurfaceMaterial: matIdx out of bounds");
      return;
    }
    auto& textureManager = m_device->getCommon()->getTextureManager();
    const RtSurfaceMaterial& mat = m_surfaceMaterialCache.getObjectTable()[matIdx];
    mat.forEachTextureIndex([&](uint32_t texIdx) {
      textureManager.retainTexture(texIdx);
    });
    if (mat.getType() == RtSurfaceMaterialType::Opaque) {
      const RtOpaqueSurfaceMaterial& opaque = mat.getOpaqueSurfaceMaterial();
      if (opaque.hasValidDisplacement()) {
        ++m_activePOMCount;
      }
      const uint32_t subsurfaceIdx = opaque.getSubsurfaceMaterialIndex();
      if (subsurfaceIdx != SURFACE_INDEX_INVALID &&
          subsurfaceIdx < m_surfaceMaterialExtensionCache.getTotalCount()) {
        const RtSurfaceMaterial& extMat = m_surfaceMaterialExtensionCache.getObjectTable()[subsurfaceIdx];
        extMat.forEachTextureIndex([&](uint32_t texIdx) {
          textureManager.retainTexture(texIdx);
        });
        if (extMat.getType() == RtSurfaceMaterialType::Subsurface) {
          const float radiusScale = extMat.getSubsurfaceMaterial().getSubsurfaceRadiusScale();
          if (radiusScale > 0.0f)      ++m_sssCount;
          else if (radiusScale < 0.0f) ++m_thinOpaqueCount;
        }
      }
    }
  }

  void SceneManager::releaseSurfaceMaterial(uint32_t matIdx) {
    // Out-of-bounds covers instances that were never bound (surfaceMaterialIndex ==
    // kSurfaceInvalidSurfaceMaterialIndex). Renderer-created instances skip
    // onInstanceDestroyedCallback entirely so they never reach here.
    if (matIdx >= m_surfaceMaterialCache.getTotalCount()) {
      return;
    }
    auto& textureManager = m_device->getCommon()->getTextureManager();
    const RtSurfaceMaterial& mat = m_surfaceMaterialCache.getObjectTable()[matIdx];
    mat.forEachTextureIndex([&](uint32_t texIdx) {
      textureManager.releaseTexture(texIdx);
    });
    if (mat.getType() == RtSurfaceMaterialType::Opaque) {
      const RtOpaqueSurfaceMaterial& opaque = mat.getOpaqueSurfaceMaterial();
      if (opaque.hasValidDisplacement()) {
        if (m_activePOMCount == 0) {
          Logger::err("[RTX] releaseSurfaceMaterial: POM count underflow (mismatched retain/release)");
          assert(false && "releaseSurfaceMaterial: POM count underflow");
        } else {
          --m_activePOMCount;
        }
      }
      const uint32_t subsurfaceIdx = opaque.getSubsurfaceMaterialIndex();
      if (subsurfaceIdx != SURFACE_INDEX_INVALID &&
          subsurfaceIdx < m_surfaceMaterialExtensionCache.getTotalCount()) {
        const RtSurfaceMaterial& extMat = m_surfaceMaterialExtensionCache.getObjectTable()[subsurfaceIdx];
        extMat.forEachTextureIndex([&](uint32_t texIdx) {
          textureManager.releaseTexture(texIdx);
        });
        if (extMat.getType() == RtSurfaceMaterialType::Subsurface) {
          const float radiusScale = extMat.getSubsurfaceMaterial().getSubsurfaceRadiusScale();
          if (radiusScale > 0.0f) {
            if (m_sssCount == 0) {
              Logger::err("[RTX] releaseSurfaceMaterial: SSS count underflow (mismatched retain/release)");
              assert(false && "releaseSurfaceMaterial: SSS count underflow");
            } else {
              --m_sssCount;
            }
          } else if (radiusScale < 0.0f) {
            if (m_thinOpaqueCount == 0) {
              Logger::err("[RTX] releaseSurfaceMaterial: thin-opaque count underflow (mismatched retain/release)");
              assert(false && "releaseSurfaceMaterial: thin-opaque count underflow");
            } else {
              --m_thinOpaqueCount;
            }
          }
        }
      }
    }
  }

  void SceneManager::onInstanceDestroyed(RtInstance& instance) {
    releaseSurfaceMaterial(instance.surface.surfaceMaterialIndex);

    // Evict from the AccelManager bucket cache to prevent stale pointer ABA issues.
    m_accelManager.removeInstanceFromBucketCache(&instance);

    BlasEntry* pBlas = instance.getBlas();
    if (pBlas != nullptr) {
      pBlas->unlinkInstance(&instance);
    }
  }

  // Helper to populate the texture cache with this resource (and patch sampler if required for texture)
  // If 'inout_samplerFeedbackStamp' is non-null and still INVALID, it is seeded from the
  // first present texture; that stamp is then reused so subsequent textures in the same
  // material are grouped together for sampler-feedback streaming.
  void SceneManager::trackTexture(const TextureRef& inputTexture,
                                  uint32_t& textureIndex,
                                  bool hasTexcoords,
                                  bool async,
                                  uint16_t* inout_samplerFeedbackStamp) {
    if (!hasTexcoords) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Trying to bind a texture to a mesh without UVs.  Was this intended?")));
      return;
    }

    if (inout_samplerFeedbackStamp != nullptr && *inout_samplerFeedbackStamp == SAMPLER_FEEDBACK_INVALID && inputTexture.getManagedTexture() != nullptr) {
      *inout_samplerFeedbackStamp = inputTexture.getManagedTexture()->m_samplerFeedbackStamp;
    }

    const uint16_t stamp = inout_samplerFeedbackStamp != nullptr ? *inout_samplerFeedbackStamp : SAMPLER_FEEDBACK_INVALID;
    auto& textureManager = m_device->getCommon()->getTextureManager();
    textureManager.addTexture(inputTexture, stamp, async, textureIndex);
  }

  RtInstance* SceneManager::processDrawCallState(const Rc<DxvkContext>& ctx, const DrawCallState& drawCallState, const MaterialData& renderMaterialData, ReplacementInstance& replacementInstance, RtInstance* existingInstance, const RtxParticleSystemDesc* pParticleSystemDesc) {
    ScopedCpuProfileZone();

    if (renderMaterialData.getIgnored()) {
      return nullptr;
    }

    // Fork: record which exterior cells this frame's terrain covers, so world-anchored scatter can be
    // scoped to the space actually being rendered. See trackTerrainCell.
    //
    // Here rather than in submitDrawState, which is where it belongs by reading but not by execution:
    // external (API) draws never pass through submitDrawState, they call this function directly. A host
    // that submits its whole world through the API therefore contributed nothing at all, and the gate
    // rejected every group because no cell was ever live. This function is the one both paths share.
    trackTerrainCell(drawCallState);

    ObjectCacheState result = ObjectCacheState::kInvalid;
    BlasEntry* pBlas = nullptr;
    if (m_drawCallCache.get(drawCallState, &pBlas) == DrawCallCache::CacheState::kExisted) {
      result = onSceneObjectUpdated(ctx, drawCallState, pBlas);
    } else {
      result = onSceneObjectAdded(ctx, drawCallState, pBlas);
    }
    
    assert(pBlas != nullptr);
    assert(result != ObjectCacheState::kInvalid);

    // Update the input state, so we always have a reference to the original draw call state
    pBlas->frameLastTouched = m_device->getCurrentFrameId();

    // Generate smooth normals for geometry that is flagged via the SmoothNormals texture category.
    // This is useful for older D3D9 games where geometry may lack smooth normals, especially
    // when using the VertexShader Capture mechanism. The smooth normals are computed on the GPU
    // from the triangle mesh (area-weighted) and written into the normal buffer.
    // Only dispatch on BVH build/update - for static geometry, positions don't change so
    // the normals computed on the first pass remain valid for subsequent frames.
    if (drawCallState.getCategoryFlags().test(InstanceCategories::SmoothNormals) &&
        (result == ObjectCacheState::KBuildBVH || result == ObjectCacheState::kUpdateBVH)) {
      m_device->getCommon()->metaGeometryUtils().dispatchSmoothNormals(ctx, drawCallState.getGeometryData(), pBlas->modifiedGeometryData);
      pBlas->modifiedGeometryData.smoothNormalsApplied = true;
      pBlas->frameLastUpdated = pBlas->frameLastTouched;
      m_instanceManager.notifySceneChanged();
    }

    if (drawCallState.getSkinningState().numBones > 0 &&
        drawCallState.getGeometryData().numBonesPerVertex > 0 &&
        (result == ObjectCacheState::KBuildBVH || result == ObjectCacheState::kUpdateBVH)) {
      m_device->getCommon()->metaGeometryUtils().dispatchSkinning(drawCallState, pBlas->modifiedGeometryData);
      pBlas->frameLastUpdated = pBlas->frameLastTouched;
      m_instanceManager.notifySceneChanged();
    }

    // Note: The material data can be modified in instance manager
    RtInstance* instance = m_instanceManager.processSceneObject(m_cameraManager, m_rayPortalManager, *pBlas, drawCallState, renderMaterialData, existingInstance);

    // Check if a light should be created for this Material
    if (instance && RtxOptions::shouldConvertToLight(drawCallState.getMaterialData().getHash())) {
      createEffectLight(ctx, drawCallState, instance);
    }

    if (instance) {
      trackObjectPickingMeta(drawCallState, instance->surface.objectPickingValue);
    }

    // Priority ordering for particle system descriptors is: Mesh, Material, Texture.  This matches the implementation in toolkit.
    // By this point, pParticleSystemDesc will contain the information from a mesh replacement (if one exists), so we just handle
    // materials replacements, and texture taggin categories below.
    RtxParticleSystemDesc globalParticleDesc; // Storage for global desc if needed
    if (!pParticleSystemDesc) {
      pParticleSystemDesc = renderMaterialData.getParticleSystemDesc();
    }
    if (!pParticleSystemDesc && drawCallState.getCategoryFlags().test(InstanceCategories::ParticleEmitter)) {
      globalParticleDesc = RtxParticleSystemManager::createGlobalParticleSystemDesc();
      pParticleSystemDesc = &globalParticleDesc;
    }
    if (instance && pParticleSystemDesc) {
      RtxParticleSystemManager& particleSystem = device()->getCommon()->metaParticleSystem();
      particleSystem.spawnParticles(ctx.ptr(), *pParticleSystemDesc, instance->getVectorIdx(), drawCallState, renderMaterialData);

      if (pParticleSystemDesc->hideEmitter) {
        instance->setHidden(true);
      }

      replacementInstance.dirtyFlags.set(ReplacementInstance::DirtyFlag::ParticleSystem);
    }

    return instance; 
  }

  void SceneManager::trackObjectPickingMeta(
      const DrawCallState& drawCallState,
      ObjectPickingValue objectPickingValue) {
    const bool objectPickingActive = m_device->getCommon()->getResources().getRaytracingOutput()
      .m_primaryObjectPicking.isValid();
    if (!objectPickingActive || !g_allowMappingLegacyHashToObjectPickingValue) {
      return;
    }

    auto meta = DrawCallMetaInfo {};
    XXH64_hash_t h = drawCallState.getMaterialData().getColorTexture().getImageHash();
    if (h != kEmptyHash) {
      meta.legacyTextureHash = h;
    }
    h = drawCallState.getMaterialData().getColorTexture2().getImageHash();
    if (h != kEmptyHash) {
      meta.legacyTextureHash2 = h;
    }

    std::lock_guard lock { m_drawCallMeta.mutex };
    auto [iter, isNew] = m_drawCallMeta.infos[m_drawCallMeta.ticker].emplace(objectPickingValue, meta);
    ONCE_IF_FALSE(isNew, Logger::warn(
      "Found multiple draw calls with the same \'objectPickingValue\'. "
      "Ignoring further MetaInfo-s, some objects might be not be available through object picking"));
  }

  const RtSurfaceMaterial& SceneManager::createSurfaceMaterial(const MaterialData& renderMaterialData,
                                                               const DrawCallState& drawCallState,
                                                               uint32_t* out_indexInCache) {
    ScopedCpuProfileZone();
    const bool hasTexcoords = drawCallState.hasTextureCoordinates();
    const auto renderMaterialDataType = renderMaterialData.getType();

    // We're going to use this to create a modified sampler for replacement textures.
    // Legacy and replacement materials should follow same filtering but due to lack of override capability per texture
    // legacy textures use original sampler to stay true to the original intent while replacements use more advanced filtering
    // for better quality by default.
    const Rc<DxvkSampler>& samplerOverride = renderMaterialData.getSamplerOverride();
    Rc<DxvkSampler> sampler = samplerOverride;
    // If the original sampler if valid and there isnt an override sampler
    // go ahead with patching and maybe merging the sampler states
    if (samplerOverride == nullptr && drawCallState.getMaterialData().getSampler().ptr() != nullptr) {
      DxvkSamplerCreateInfo samplerInfo = drawCallState.getMaterialData().getSampler()->info(); // Use sampler create info struct as convenience
      renderMaterialData.populateSamplerInfo(samplerInfo);

      sampler = patchSampler(samplerInfo.magFilter,
                             samplerInfo.addressModeU, samplerInfo.addressModeV, samplerInfo.addressModeW,
                             samplerInfo.borderColor);
    }
    if (drawCallState.isEye()) {
      // force eye whites and iris to not repeat
      sampler = patchSampler(
        VK_FILTER_LINEAR,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        {}
      );
    }
    uint32_t samplerIndex = trackSampler(sampler);
    uint32_t samplerIndex2 = UINT32_MAX;
    if (renderMaterialDataType == MaterialDataType::RayPortal) {
      samplerIndex2 = trackSampler(drawCallState.getMaterialData().getSampler2());
    }

    XXH64_hash_t preCreationHash = renderMaterialData.getHash();
    preCreationHash = XXH64(&samplerIndex, sizeof(samplerIndex), preCreationHash);
    preCreationHash = XXH64(&samplerIndex2, sizeof(samplerIndex2), preCreationHash);
    preCreationHash = XXH64(&hasTexcoords, sizeof(hasTexcoords), preCreationHash);
    preCreationHash = XXH64(&drawCallState.isUsingRaytracedRenderTarget, sizeof(drawCallState.isUsingRaytracedRenderTarget), preCreationHash);
    const bool isHairCard = drawCallState.testCategoryFlags(InstanceCategories::HairCards);
    preCreationHash = XXH64(&isHairCard, sizeof(isHairCard), preCreationHash);

    // For Opaque materials, fold in a bitmask of which texture slots are populated. MaterialData::getHash()
    // sums TextureRef::getImageHash() across slots, but render-target-backed TextureRefs (notably
    // TerrainBaker cascades) have a zero image hash, so two materials that differ only in cascade
    // composition (e.g. "albedo only" vs "albedo + normal + roughness") collide on getHash() and the
    // per-frame cache would serve the earlier "albedo only" entry to later draws that have valid
    // secondary cascades.
    if (renderMaterialDataType == MaterialDataType::Opaque) {
      const auto& opaqueMaterialData = renderMaterialData.getOpaqueMaterialData();
      uint32_t texturePresenceMask = 0;
      texturePresenceMask |= opaqueMaterialData.getAlbedoOpacityTexture().isImageEmpty()             ? 0u : (1u << 0);
      texturePresenceMask |= opaqueMaterialData.getSecondaryTexture().isImageEmpty()                 ? 0u : (1u << 1);
      texturePresenceMask |= opaqueMaterialData.getNormalTexture().isImageEmpty()                    ? 0u : (1u << 2);
      texturePresenceMask |= opaqueMaterialData.getTangentTexture().isImageEmpty()                   ? 0u : (1u << 3);
      texturePresenceMask |= opaqueMaterialData.getHeightTexture().isImageEmpty()                    ? 0u : (1u << 4);
      texturePresenceMask |= opaqueMaterialData.getRoughnessTexture().isImageEmpty()                 ? 0u : (1u << 5);
      texturePresenceMask |= opaqueMaterialData.getMetallicTexture().isImageEmpty()                  ? 0u : (1u << 6);
      texturePresenceMask |= opaqueMaterialData.getEmissiveColorTexture().isImageEmpty()             ? 0u : (1u << 7);
      texturePresenceMask |= opaqueMaterialData.getSubsurfaceTransmittanceTexture().isImageEmpty()   ? 0u : (1u << 8);
      texturePresenceMask |= opaqueMaterialData.getSubsurfaceThicknessTexture().isImageEmpty()       ? 0u : (1u << 9);
      texturePresenceMask |= opaqueMaterialData.getSubsurfaceSingleScatteringAlbedoTexture().isImageEmpty() ? 0u : (1u << 10);
      texturePresenceMask |= opaqueMaterialData.getSubsurfaceRadiusTexture().isImageEmpty()          ? 0u : (1u << 11);
      preCreationHash = XXH64(&texturePresenceMask, sizeof(texturePresenceMask), preCreationHash);

      // Fold in the sRGB-linearization toggle so flipping rtx.linearizeSrgbTextures at runtime invalidates
      // cached opaque materials, forcing them to rebuild with the new albedo/emissive sRGB flags. Without this
      // the preCreationHash cache below would keep serving materials built under the previous setting, so the
      // change would only reach freshly-encountered materials. Enables a live A/B without reloading.
      const uint32_t srgbLinearizeToggle = RtxOptions::linearizeSrgbTextures() ? 1u : 0u;
      preCreationHash = XXH64(&srgbLinearizeToggle, sizeof(srgbLinearizeToggle), preCreationHash);
    }

    auto iter = m_preCreationSurfaceMaterialMap.find(preCreationHash);
    if (iter != m_preCreationSurfaceMaterialMap.end()) {
      if (out_indexInCache) {
        *out_indexInCache = iter->second;
      }
      return m_surfaceMaterialCache.at(iter->second);
    }

    std::optional<RtSurfaceMaterial> surfaceMaterial;

    if (renderMaterialDataType == MaterialDataType::Opaque || drawCallState.isUsingRaytracedRenderTarget) {
      uint32_t albedoOpacityTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t secondaryTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t normalTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t tangentTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t heightTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t roughnessTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t metallicTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t emissiveColorTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t subsurfaceMaterialIndex = SURFACE_INDEX_INVALID;
      uint32_t subsurfaceTransmittanceTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t subsurfaceThicknessTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t subsurfaceSingleScatteringAlbedoTextureIndex = kSurfaceMaterialInvalidTextureIndex;

      float anisotropy;
      float emissiveIntensity;
      Vector4 albedoOpacityConstant;
      float roughnessConstant;
      float metallicConstant;
      Vector3 emissiveColorConstant;
      bool enableEmissive;
      bool thinFilmEnable = false;
      bool alphaIsThinFilmThickness = false;
      float thinFilmThicknessConstant = 0.0f;
      float displaceIn = 0.0f;
      float displaceOut = 0.0f;
      bool isUsingRaytracedRenderTarget = drawCallState.isUsingRaytracedRenderTarget;
      uint16_t samplerFeedbackStamp = SAMPLER_FEEDBACK_INVALID;

      Vector3 subsurfaceTransmittanceColor(0.0f, 0.0f, 0.0f);
      float subsurfaceMeasurementDistance = 0.0f;
      Vector3 subsurfaceSingleScatteringAlbedo(0.0f, 0.0f, 0.0f);
      float subsurfaceVolumetricAnisotropy = 0.0f;

      float subsurfaceRadiusScale = 0.0f;
      float subsurfaceMaxSampleRadius = 0.0f;

      bool ignoreAlphaChannel = false;

      constexpr Vector4 kWhiteModeAlbedo = Vector4(0.7f, 0.7f, 0.7f, 1.0f);

      const auto& opaqueMaterialData = renderMaterialData.getOpaqueMaterialData();

      if (RtxOptions::useWhiteMaterialMode()) {
        albedoOpacityConstant = kWhiteModeAlbedo;
        metallicConstant = 0.f;
        roughnessConstant = 1.f;
      } else {
        trackTexture(opaqueMaterialData.getAlbedoOpacityTexture(), albedoOpacityTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
        trackTexture(opaqueMaterialData.getRoughnessTexture(), roughnessTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
        trackTexture(opaqueMaterialData.getMetallicTexture(), metallicTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
        trackTexture(opaqueMaterialData.getSecondaryTexture(), secondaryTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);

        albedoOpacityConstant.xyz() = opaqueMaterialData.getAlbedoConstant();
        albedoOpacityConstant.w = opaqueMaterialData.getOpacityConstant();
        metallicConstant = opaqueMaterialData.getMetallicConstant();
        roughnessConstant = opaqueMaterialData.getRoughnessConstant();
      }

      trackTexture(opaqueMaterialData.getNormalTexture(), normalTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getTangentTexture(), tangentTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getHeightTexture(), heightTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getEmissiveColorTexture(), emissiveColorTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);

      emissiveIntensity = opaqueMaterialData.getEmissiveIntensity() * RtxOptions::emissiveIntensity();
      emissiveColorConstant = opaqueMaterialData.getEmissiveColorConstant();
      enableEmissive = opaqueMaterialData.getEnableEmission();
      anisotropy = opaqueMaterialData.getAnisotropyConstant();
        
      thinFilmEnable = opaqueMaterialData.getEnableThinFilm();
      alphaIsThinFilmThickness = opaqueMaterialData.getAlphaIsThinFilmThickness();
      thinFilmThicknessConstant = opaqueMaterialData.getThinFilmThicknessConstant();
      displaceIn = opaqueMaterialData.getDisplaceIn();
      displaceOut = opaqueMaterialData.getDisplaceOut();

      ignoreAlphaChannel = opaqueMaterialData.getIgnoreAlphaChannel();

      subsurfaceMeasurementDistance = opaqueMaterialData.getSubsurfaceMeasurementDistance() * RtxOptions::SubsurfaceScattering::surfaceThicknessScale();

      const bool isSubsurfaceScatteringDiffusionProfile = opaqueMaterialData.getSubsurfaceDiffusionProfile();

      if ((RtxOptions::SubsurfaceScattering::enableThinOpaque()       && subsurfaceMeasurementDistance > 0.0f) ||
          (RtxOptions::SubsurfaceScattering::enableDiffusionProfile() && isSubsurfaceScatteringDiffusionProfile)) {

        subsurfaceTransmittanceColor = opaqueMaterialData.getSubsurfaceTransmittanceColor();
        subsurfaceVolumetricAnisotropy = opaqueMaterialData.getSubsurfaceVolumetricAnisotropy();

        if (isSubsurfaceScatteringDiffusionProfile) {
          // NOTE: reuse of the variable!
          subsurfaceSingleScatteringAlbedo = opaqueMaterialData.getSubsurfaceRadius(); 
          subsurfaceMaxSampleRadius = std::max(0.F, opaqueMaterialData.getSubsurfaceMaxSampleRadius());
          subsurfaceRadiusScale = std::max(opaqueMaterialData.getSubsurfaceRadiusScale(), 1e-5f);
          assert(subsurfaceRadiusScale > 0);
        } else /* if thin opaque */ {
          assert(subsurfaceMeasurementDistance > 0);

          subsurfaceSingleScatteringAlbedo = opaqueMaterialData.getSubsurfaceSingleScatteringAlbedo();
          subsurfaceMaxSampleRadius = 0;
          subsurfaceRadiusScale = -1;
          assert(subsurfaceRadiusScale < 0);  // if < 0, then shaders assume that
                                              // this material is not SubsurfaceScatter, but just SingleScatter
                                              // same here, but <0.F
        }

        if (RtxOptions::SubsurfaceScattering::enableTextureMaps()) {
          trackTexture(opaqueMaterialData.getSubsurfaceTransmittanceTexture(), subsurfaceTransmittanceTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);

          if (isSubsurfaceScatteringDiffusionProfile) {
            // NOTE: reuse of 'subsurfaceSingleScatteringAlbedoTextureIndex' variable!
            trackTexture(opaqueMaterialData.getSubsurfaceRadiusTexture(), subsurfaceSingleScatteringAlbedoTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
          } else {
            trackTexture(opaqueMaterialData.getSubsurfaceSingleScatteringAlbedoTexture(), subsurfaceSingleScatteringAlbedoTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
            trackTexture(opaqueMaterialData.getSubsurfaceThicknessTexture(), subsurfaceThicknessTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
          }
        }

        const auto subsurfaceMaterial = RtSubsurfaceMaterial{
          subsurfaceTransmittanceTextureIndex,
          subsurfaceThicknessTextureIndex,
          subsurfaceSingleScatteringAlbedoTextureIndex,
          subsurfaceTransmittanceColor,
          subsurfaceMeasurementDistance,
          subsurfaceSingleScatteringAlbedo,
          subsurfaceVolumetricAnisotropy,
          subsurfaceRadiusScale,
          subsurfaceMaxSampleRadius,
        };
        subsurfaceMaterialIndex = m_surfaceMaterialExtensionCache.track(subsurfaceMaterial);
      }

      // Detect whether the albedo/emissive source textures use an sRGB VkFormat. If so, the sampler hardware
      // linearizes them on read, so the shader must skip its own gammaToLinear() to avoid double linearization.
      // Gated behind linearizeSrgbTextures() (default on) for A/B; when off the flags stay clear and the shader
      // always applies the software conversion (legacy behavior). Uses the resolved image-view format, which is
      // available here whenever the texture is loaded (an unloaded texture reports isImageEmpty(), which also
      // feeds the material cache key, so the material is rebuilt with the correct flag once the texture resolves).
      const bool srgbLinearizeEnabled = RtxOptions::linearizeSrgbTextures();
      auto textureUsesSrgbFormat = [srgbLinearizeEnabled](const TextureRef& tex) -> bool {
        if (!srgbLinearizeEnabled) {
          return false;
        }
        const DxvkImageView* view = tex.getImageView();
        return view != nullptr && TextureUtils::isSRGB(view->info().format);
      };
      const bool albedoTextureIsSrgb = textureUsesSrgbFormat(opaqueMaterialData.getAlbedoOpacityTexture());
      const bool emissiveTextureIsSrgb = textureUsesSrgbFormat(opaqueMaterialData.getEmissiveColorTexture());

      const RtOpaqueSurfaceMaterial opaqueSurfaceMaterial{
        albedoOpacityTextureIndex, normalTextureIndex,
        tangentTextureIndex, heightTextureIndex, roughnessTextureIndex,
        metallicTextureIndex, emissiveColorTextureIndex,
        anisotropy, emissiveIntensity,
        albedoOpacityConstant,
        roughnessConstant, metallicConstant,
        emissiveColorConstant, enableEmissive,
        ignoreAlphaChannel, thinFilmEnable, alphaIsThinFilmThickness,
        thinFilmThicknessConstant, samplerIndex, displaceIn, displaceOut,
        subsurfaceMaterialIndex, isUsingRaytracedRenderTarget, isHairCard,
        samplerFeedbackStamp,
        secondaryTextureIndex,
        albedoTextureIsSrgb, emissiveTextureIsSrgb,
        opaqueMaterialData.getSkyLitParticle()
      };

      surfaceMaterial.emplace(opaqueSurfaceMaterial);
    } else if (renderMaterialDataType == MaterialDataType::Translucent) {
      surfaceMaterial.emplace(createTranslucentSurfaceMaterial(renderMaterialData.getTranslucentMaterialData(), samplerIndex, hasTexcoords));
    } else if (renderMaterialDataType == MaterialDataType::RayPortal) {
      const auto& rayPortalMaterialData = renderMaterialData.getRayPortalMaterialData();

      uint32_t maskTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      trackTexture(rayPortalMaterialData.getMaskTexture(), maskTextureIndex, hasTexcoords, false);
      uint32_t maskTextureIndex2 = kSurfaceMaterialInvalidTextureIndex;
      trackTexture(rayPortalMaterialData.getMaskTexture2(), maskTextureIndex2, hasTexcoords, false);

      uint8_t rayPortalIndex = rayPortalMaterialData.getRayPortalIndex();
      float rotationSpeed = rayPortalMaterialData.getRotationSpeed();
      bool enableEmissive = rayPortalMaterialData.getEnableEmission();
      float emissiveIntensity = rayPortalMaterialData.getEmissiveIntensity() * RtxOptions::emissiveIntensity();

      const RtRayPortalSurfaceMaterial rayPortalSurfaceMaterial{
        maskTextureIndex, maskTextureIndex2, rayPortalIndex,
        rotationSpeed, enableEmissive, emissiveIntensity, samplerIndex, samplerIndex2
      };

      surfaceMaterial.emplace(rayPortalSurfaceMaterial);
    }

    assert(surfaceMaterial.has_value());
    assert(surfaceMaterial->validate());

    // Cache this
    const uint32_t index = m_surfaceMaterialCache.track(*surfaceMaterial);
    m_preCreationSurfaceMaterialMap[preCreationHash] = index;
    if (out_indexInCache) {
      *out_indexInCache = index;
    }
    return m_surfaceMaterialCache.at(index);
  }


  RtTranslucentSurfaceMaterial SceneManager::createTranslucentSurfaceMaterial(const TranslucentMaterialData& translucentMaterialData,
                                                                              uint32_t samplerIndex,
                                                                              bool hasTexcoords) {
    uint32_t normalTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    uint32_t transmittanceTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    uint32_t emissiveColorTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    uint16_t samplerFeedbackStamp = SAMPLER_FEEDBACK_INVALID;

    trackTexture(translucentMaterialData.getNormalTexture(), normalTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
    trackTexture(translucentMaterialData.getTransmittanceTexture(), transmittanceTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
    trackTexture(translucentMaterialData.getEmissiveColorTexture(), emissiveColorTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);

    return RtTranslucentSurfaceMaterial{
      normalTextureIndex,
      transmittanceTextureIndex,
      emissiveColorTextureIndex,
      translucentMaterialData.getRefractiveIndex() * std::clamp(TranslucentMaterialOptions::refractiveIndexScale(), 0.0f, 3.0f),
      translucentMaterialData.getTransmittanceMeasurementDistance(),
      translucentMaterialData.getTransmittanceColor(),
      translucentMaterialData.getEnableEmission(),
      translucentMaterialData.getEmissiveIntensity() * RtxOptions::emissiveIntensity(),
      translucentMaterialData.getEmissiveColorConstant(),
      translucentMaterialData.getEnableThinWalled(),
      translucentMaterialData.getThinWallThickness(),
      translucentMaterialData.getEnableDiffuseLayer(),
      samplerIndex,
      samplerFeedbackStamp
    };
  }

  Rc<DxvkSampler> SceneManager::getOrCreateExternalSampler() {
    if (m_externalSampler == nullptr) {
      auto s = DxvkSamplerCreateInfo {};
      {
        s.magFilter = VK_FILTER_LINEAR;
        s.minFilter = VK_FILTER_LINEAR;
        s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        s.mipmapLodBias = 0.f;
        s.mipmapLodMin = 0.f;
        s.mipmapLodMax = 0.f;
        s.useAnisotropy = VK_FALSE;
        s.maxAnisotropy = 1.f;
        s.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        s.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        s.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        s.compareToDepth = VK_FALSE;
        s.compareOp = VK_COMPARE_OP_NEVER;
        s.borderColor = VkClearColorValue {};
        s.usePixelCoord = VK_FALSE;
      }
      m_externalSampler = m_device->createSampler(s);
    }

    return m_externalSampler;
  }

  void SceneManager::setExternalStartInMediumMaterial(const MaterialData& translucentMaterial) {
    assert(translucentMaterial.getType() == MaterialDataType::Translucent);

    const auto samplerIndex = trackSampler(getOrCreateExternalSampler());
    const auto surfaceMaterial = RtSurfaceMaterial(
      createTranslucentSurfaceMaterial(translucentMaterial.getTranslucentMaterialData(), samplerIndex, true));

    m_externalStartInMediumMaterialIndex_inCache = m_surfaceMaterialCache.track(surfaceMaterial);
  }

  void SceneManager::clearExternalStartInMediumMaterial() {
    m_externalStartInMediumMaterialIndex_inCache = UINT32_MAX;
  }

  void SceneManager::setStartInMediumMaterial(const MaterialData& translucentMaterial) {
    assert(translucentMaterial.getType() == MaterialDataType::Translucent);
    std::lock_guard lock { m_startInMediumMaterialMutex };
    m_pendingClearStartInMediumMaterial = false;
    m_pendingStartInMediumMaterial = translucentMaterial;
  }

  void SceneManager::clearStartInMediumMaterial() {
    std::lock_guard lock { m_startInMediumMaterialMutex };
    m_pendingStartInMediumMaterial.reset();
    m_pendingClearStartInMediumMaterial = true;
  }

  std::optional<XXH64_hash_t> SceneManager::findLegacyTextureHashByObjectPickingValue(uint32_t objectPickingValue) {
    std::lock_guard lock { m_drawCallMeta.mutex };

    auto tryFindIn = [](const std::unordered_map<ObjectPickingValue, DrawCallMetaInfo>& table, ObjectPickingValue toFind)
      -> std::optional<XXH64_hash_t> {
      auto found = table.find(toFind);
      if (found != table.end()) {
        const DrawCallMetaInfo& meta = found->second;
        if (meta.legacyTextureHash != kEmptyHash) {
          return meta.legacyTextureHash;
        }
      }
      return std::nullopt;
    };

    const int ticksToCheck[] = {
      m_drawCallMeta.ticker, // current tick
      (m_drawCallMeta.ticker + m_drawCallMeta.MaxTicks - 1) % m_drawCallMeta.MaxTicks, // prev tick
    };
    for (int tick : ticksToCheck) {
      if (m_drawCallMeta.ready[tick]) {
        if (auto h = tryFindIn(m_drawCallMeta.infos[tick], objectPickingValue)) {
          return h;
        }
      }
    }
    return std::nullopt;
  }

  std::vector<ObjectPickingValue> SceneManager::gatherObjectPickingValuesByTextureHash(XXH64_hash_t texHash) {
    std::lock_guard lock { m_drawCallMeta.mutex };
    assert(texHash != kEmptyHash);

    const int ticksToCheck[] = {
      m_drawCallMeta.ticker, // current tick
      (m_drawCallMeta.ticker + m_drawCallMeta.MaxTicks - 1) % m_drawCallMeta.MaxTicks, // prev tick
    };

    auto correspondingValues = std::vector<ObjectPickingValue> {};
    for (int tick : ticksToCheck) {
      if (m_drawCallMeta.ready[tick]) {
        for (const auto& [pickingValue, meta] : m_drawCallMeta.infos[tick]) {
          if (texHash == meta.legacyTextureHash) {
            correspondingValues.push_back(pickingValue);
          } else if (texHash == meta.legacyTextureHash2) {
            correspondingValues.push_back(pickingValue);
          }
        }
        break;
      }
    }
    return correspondingValues;
  }

  SceneManager::SamplerIndex SceneManager::trackSampler(Rc<DxvkSampler> sampler) {
    if (sampler == nullptr) {
      ONCE(Logger::warn("Found a null sampler. Fallback to linear-repeat"));
      sampler = patchSampler(
        VK_FILTER_LINEAR,
        VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VkClearColorValue {});
    }
    return m_samplerCache.track(sampler);
  }

  Rc<DxvkSampler> SceneManager::patchSampler( const VkFilter filterMode,
                                              const VkSamplerAddressMode addressModeU,
                                              const VkSamplerAddressMode addressModeV,
                                              const VkSamplerAddressMode addressModeW,
                                              const VkClearColorValue borderColor) {
    auto& resourceManager = m_device->getCommon()->getResources();
    // Create a sampler to account for DLSS lod bias and any custom filtering overrides the user has set
    return resourceManager.getSampler(
      filterMode,
      VK_SAMPLER_MIPMAP_MODE_LINEAR,
      addressModeU,
      addressModeV,
      addressModeW,
      borderColor,
      getTotalMipBias(),
      RtxOptions::useAnisotropicFiltering());
  }

  bool SceneManager::applyExternalLightReplacement(RtLight& original) {
    ScopedCpuProfileZone();

    // Identity and lookup key are built exactly as addLight(const D3DLIGHT9&) builds them, because they
    // have to name the same ReplacementInstance a capture and the toolkit name.
    const XXH64_hash_t lightAssetHash = original.getInitialHash();
    const Vector3 lightPos = original.getPosition();
    const XXH64_hash_t lightIdHash = XXH64(&lightPos, sizeof(Vector3), lightAssetHash);

    const std::vector<AssetReplacement>* pReplacements = m_pReplacer->getReplacementsForLight(lightAssetHash);

    if (pReplacements == nullptr) {
      // Nothing authored, or enableReplacementLights was just switched off. The second case has to tear
      // the instance down rather than simply stop using it: replacement lights are externally tracked and
      // have no frame-age collection, their lifetime belongs to the instance, so leaving it alone renders
      // the replacements next to the original that is about to be drawn again.
      if (ReplacementInstance* stale = m_drawCallTracker.findReplacementInstanceByIdentity(lightIdHash)) {
        if (stale->activeReplacements != nullptr) {
          stale->clear();
        }
      }
      return false;
    }

    // A sphere has no orientation, so the transform a relative replacement is placed by is a pure
    // translation -- which is also what LightUtils::getLightTransform returns for a D3D9 point light.
    const Matrix4 lightTransform = Matrix4(lightPos);

    // The original's own shape and radiance, for merging into entries that did not specify them. Taken
    // from the converted light rather than reconstructed, so a replacement that only overrides intensity
    // inherits precisely what the host submitted.
    const float originalRadius = (original.getType() == RtLightType::Sphere)
        ? original.getSphereLight().getRadius()
        : 0.0f;
    const Vector3 originalRadiance = original.getRadiance();

    // The original's flicker, recovered as a ratio so the replacements can inherit it. See m_lightFlicker.
    //
    // Morrowind drives flicker and pulse host-side, so a flickering light's radiance already varies every
    // frame by the time it reaches here. Dividing the current radiance by the brightest this same light has
    // been seen at recovers the curve without knowing anything about how it was produced -- flicker against
    // pulse, fast against slow, and each light's own random phase all come through for free, and cannot
    // drift from what the rest of the scene is doing the way a reimplementation would.
    //
    // Magnitude rather than per channel: LightController scales the colour uniformly, so one scalar carries
    // it and a per-channel ratio would only add noise where a channel is near zero.
    const uint32_t currentFrame = m_device->getCurrentFrameId();
    const float currentMagnitude = std::max(originalRadiance.x,
        std::max(originalRadiance.y, originalRadiance.z));

    LightFlickerState& flicker = m_lightFlicker[lightIdHash];
    flicker.peakRadiance = std::max(flicker.peakRadiance, currentMagnitude);
    flicker.frameLastSeen = currentFrame;

    // Clamped to 1 so the ratio can never brighten a replacement above what it was authored at while the
    // peak is still converging, and guarded against a light that is genuinely off -- a zero peak would
    // otherwise divide by zero and a zeroed original is exactly the workflow here, where the vanilla light
    // is set to zero intensity in the toolkit and replaced. Note that zeroing happens in the mod, so the
    // radiance measured above is still the game's own and still flickers.
    const float flickerRatio = flicker.peakRadiance > 0.0f
        ? std::min(1.0f, currentMagnitude / flicker.peakRadiance)
        : 1.0f;
    const Vector3 flickerPeakRadiance = (flickerRatio > 0.0f)
        ? originalRadiance / flickerRatio
        : originalRadiance;

    // Dropped once the light has been out of sight long enough that its phase no longer matters, which
    // bounds the map across a session rather than letting every light of every visited cell accumulate.
    if (m_lightFlicker.size() > kLightFlickerPruneThreshold) {
      for (auto it = m_lightFlicker.begin(); it != m_lightFlicker.end();) {
        it = (currentFrame - it->second.frameLastSeen > kLightFlickerPruneFrames)
            ? m_lightFlicker.erase(it) : std::next(it);
      }
    }

    const ReplacementInstance::LookupKey lightKey {
      lightIdHash, lightAssetHash, kEmptyHash, kEmptyHash, lightPos, lightTransform
    };
    ReplacementInstance* replacementInstance = m_drawCallTracker.findOrCreateReplacementInstance(lightKey);

    // Reinitialise when the prim count stops matching the replacement count, which is how the transition
    // from unreplaced (one prim) to replaced (N prims) is caught once an async load completes.
    if (replacementInstance->root.getUntyped() != nullptr
        && replacementInstance->prims.size() != pReplacements->size()) {
      replacementInstance->clear();
    }

    const bool needsBBoxUpdate = replacementInstance->boundingBoxDirty;
    AxisAlignedBoundingBox litBBox;
    bool instantiatedAny = false;

    for (size_t i = 0; i < pReplacements->size(); i++) {
      const auto& replacement = (*pReplacements)[i];

      // Meshes parented to a light are not supported by the runtime on any path -- see TREX-1091 on the
      // D3D9 equivalent, which asserts here. Skipped rather than asserted, because a pack authored
      // elsewhere may well contain them and an assert would take a release build down over content.
      if (replacement.type != AssetReplacement::eLight || !replacement.lightData.has_value()) {
        continue;
      }

      LightData replacementLight = replacement.lightData.value();

      // Before the AABB is read, as on the D3D9 path: an entry such as the translated original carries
      // Unknown type and a zero position until the submitted light has been merged into it.
      //
      // Merged with the flicker taken back OUT of the radiance, then re-applied to everything below. An
      // entry that specifies no intensity of its own inherits what is passed here, so passing the live
      // radiance and then scaling as well would flicker those twice -- squared, and visibly wrong. Passing
      // the peak keeps the inherited case exactly as it was and lets the single scale below cover both.
      replacementLight.merge(lightPos, originalRadius, flickerPeakRadiance);

      // The flicker itself, which is the whole of what makes a toolkit-added light stop being static.
      replacementLight.scaleIntensity(flickerRatio);

      RtLight rtReplacementLight = replacementLight.toRtLight();

      if (needsBBoxUpdate) {
        const Vector3 pos = rtReplacementLight.getPosition();
        float lightRadius = 0.f;
        if (rtReplacementLight.getType() == RtLightType::Sphere) {
          lightRadius = rtReplacementLight.getSphereLight().getRadius();
        }
        for (uint32_t j = 0; j < 3; j++) {
          litBBox.minPos[j] = std::min(litBBox.minPos[j], pos[j] - lightRadius);
          litBBox.maxPos[j] = std::max(litBBox.maxPos[j], pos[j] + lightRadius);
        }
      }

      if (replacementLight.relativeTransform()) {
        rtReplacementLight.applyTransform(lightTransform);
      }

      RtLight* existingLight = (replacementInstance->prims.size() > i)
          ? replacementInstance->prims[i].getLight() : nullptr;

      if (existingLight != nullptr) {
        m_lightManager.updateExternallyTrackedLight(existingLight, rtReplacementLight);
        instantiatedAny = true;
      } else {
        RtLight* newLight = m_lightManager.createExternallyTrackedLight(rtReplacementLight);
        if (newLight != nullptr) {
          if (replacementInstance->prims.empty()) {
            replacementInstance->setup(PrimInstance(newLight, PrimInstance::Type::Light),
                pReplacements->size(), pReplacements);
          }
          newLight->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, newLight,
              PrimInstance::Type::Light);
          if (replacementInstance->root.getUntyped() == nullptr) {
            replacementInstance->root = PrimInstance(newLight, PrimInstance::Type::Light);
          }
          instantiatedAny = true;
        }
      }
    }

    replacementInstance->frameLastSeen = m_device->getCurrentFrameId();
    replacementInstance->objectToWorld = lightTransform;

    if (needsBBoxUpdate) {
      if (litBBox.isValid()) {
        replacementInstance->lightBoundingBox = litBBox;
      }
      replacementInstance->boundingBoxDirty = false;
    }

    // Only claim the original when something actually stands in for it. A replacement set consisting
    // solely of unsupported entries would otherwise delete the light from the scene and put nothing in
    // its place, which reads as "the toolkit made my lamp disappear".
    return instantiatedAny;
  }

  void SceneManager::addLight(const D3DLIGHT9& light) {
    ScopedCpuProfileZone();
    // Attempt to convert the D3D9 light to RT

    std::optional<LightData> lightData = LightData::tryCreate(light);

    // Note: Skip adding this light if it is somehow malformed such that it could not be created.
    if (!lightData.has_value()) {
      return;
    }

    const RtLight rtLight = lightData->toRtLight();
    const std::vector<AssetReplacement>* pReplacements = m_pReplacer->getReplacementsForLight(rtLight.getInitialHash());

    // Build identity hash from the light's stable hash + position. Used by
    // both the replacement and the toggle-off cleanup paths below; must stay
    // in sync between them so they target the same RI.
    const XXH64_hash_t lightAssetHash = rtLight.getInitialHash();
    const Vector3 lightPos = rtLight.getPosition();
    const XXH64_hash_t lightIdHash = XXH64(&lightPos, sizeof(Vector3), lightAssetHash);

    if (pReplacements) {
      const Matrix4 lightTransform = LightUtils::getLightTransform(light);

      // Build identity hash from the light's stable hash + position
      const XXH64_hash_t lightAssetHash = rtLight.getInitialHash();
      const Vector3 lightPos = rtLight.getPosition();
      XXH64_hash_t lightIdHash = lightAssetHash;
      lightIdHash = XXH64(&lightPos, sizeof(Vector3), lightIdHash);

      const ReplacementInstance::LookupKey lightKey { lightIdHash, lightAssetHash, kEmptyHash, kEmptyHash, lightPos, lightTransform };
      ReplacementInstance* replacementInstance = m_drawCallTracker.findOrCreateReplacementInstance(lightKey);

      // Reinitialize the RI if the prim count doesn't match the replacement count.
      // This handles the transition from unreplaced (1 prim) to replaced (N prims)
      // when replacements finish loading asynchronously.
      if (replacementInstance->root.getUntyped() != nullptr &&
          replacementInstance->prims.size() != pReplacements->size()) {
        replacementInstance->clear();
      }

      // All lights in a light replacement are externally tracked, with their
      // lifecycle managed by the ReplacementInstance. This unifies root and sub-light
      // handling: create on first frame, update on subsequent frames.
      // TODO(TREX-1091) to implement meshes as light replacements, replace the below loop with a call to drawReplacements.
      const bool needsBBoxUpdate = replacementInstance->boundingBoxDirty;
      AxisAlignedBoundingBox litBBox;
      for (size_t i = 0; i < pReplacements->size(); i++) {
        const auto& replacement = (*pReplacements)[i];
        if (replacement.type == AssetReplacement::eLight && replacement.lightData.has_value()) {
          LightData replacementLight = replacement.lightData.value();

          // Merge the d3d9 light into replacements based on overrides.
          // Must happen before AABB extraction: some entries (e.g. the translated
          // original game light) have Unknown lightType and zero position/radius
          // until merged with the d3d9 light.
          replacementLight.merge(light);

          // Convert to runtime light
          RtLight rtReplacementLight = replacementLight.toRtLight();

          if (needsBBoxUpdate) {
            const Vector3 pos = rtReplacementLight.getPosition();
            float lightRadius = 0.f;
            if (rtReplacementLight.getType() == RtLightType::Sphere) {
              lightRadius = rtReplacementLight.getSphereLight().getRadius();
            }
            for (uint32_t j = 0; j < 3; j++) {
              litBBox.minPos[j] = std::min(litBBox.minPos[j], pos[j] - lightRadius);
              litBBox.maxPos[j] = std::max(litBBox.maxPos[j], pos[j] + lightRadius);
            }
          }

          // Transform the replacement light by the legacy light
          if (replacementLight.relativeTransform()) {
            rtReplacementLight.applyTransform(lightTransform);
          }

          RtLight* existingLight = (replacementInstance->prims.size() > i)
              ? replacementInstance->prims[i].getLight() : nullptr;
          if (existingLight != nullptr) {
            m_lightManager.updateExternallyTrackedLight(existingLight, rtReplacementLight);
          } else {
            RtLight* newLight = m_lightManager.createExternallyTrackedLight(rtReplacementLight);
            if (newLight != nullptr) {
              if (replacementInstance->prims.empty()) {
                replacementInstance->setup(PrimInstance(newLight, PrimInstance::Type::Light), pReplacements->size(), pReplacements);
              }
              newLight->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, newLight, PrimInstance::Type::Light);
              if (replacementInstance->root.getUntyped() == nullptr) {
                replacementInstance->root = PrimInstance(newLight, PrimInstance::Type::Light);
              }
            }
          }
        } else {
          assert(false); // We don't support meshes as children of lights yet.
        }
      }

      replacementInstance->frameLastSeen = m_device->getCurrentFrameId();
      replacementInstance->objectToWorld = lightTransform;
      if (needsBBoxUpdate) {
        if (litBBox.isValid()) {
          replacementInstance->lightBoundingBox = litBBox;
        }
        replacementInstance->boundingBoxDirty = false;
      }
    } else {
      // If this light previously had a replacement (e.g. enableReplacementLights
      // was just toggled off), the externally-tracked replacement lights are
      // still alive in LightManager -- they have no frame-age GC; their
      // lifecycle is owned by the RI. Tear down the RI so its prims get marked
      // for GC; otherwise the user sees the replacement lights and the original
      // game light rendering simultaneously until DrawCallTracker collects the
      // RI ~numFramesToKeepInstances frames later.
      if (ReplacementInstance* stale = m_drawCallTracker.findReplacementInstanceByIdentity(lightIdHash)) {
        if (stale->activeReplacements != nullptr) {
          stale->clear();
        }
      }

      // This is a light coming from the game directly, so use the appropriate API for filter rules
      m_lightManager.addGameLight(light.Type, rtLight);
    }
  }

  void SceneManager::prepareSceneData(Rc<RtxContext> ctx, DxvkBarrierSet& execBarriers) {
    ScopedGpuProfileZone(ctx, "Build Scene");

  #ifdef REMIX_DEVELOPMENT
    if (m_device->getCurrentFrameId() == RtxOptions::dumpAllInstancesOnFrame()) {
      // Print all RtInstances for debugging
      printAllRtInstances();
    }
  #endif

    // Needs to happen before garbageCollection to avoid destroying dynamic lights
    m_lightManager.dynamicLightMatching();

    garbageCollection();

    // Re-register buffers, textures, and materials for anti-culled instances.
    // These instances survived GC but the game didn't submit draw calls for them
    // this frame, so their per-frame table indices (buffer cache, material cache)
    // are stale. Without this, they would render with wrong geometry or textures.
    {
      const uint32_t currentFrameId = m_device->getCurrentFrameId();
      for (auto& ri : m_drawCallTracker.getReplacementInstances()) {
        if (ri->frameLastSeen == currentFrameId) {
          continue;
        }
        for (auto& prim : ri->prims) {
          RtInstance* instance = prim.getInstance();
          if (instance != nullptr) {
            preserveInstance(*instance);
          }
        }
      }
    }

    m_graphManager.applySceneOverrides(ctx);

    m_terrainBaker->prepareSceneData(ctx);

    auto& textureManager = m_device->getCommon()->getTextureManager();
    m_bindlessResourceManager.prepareSceneData(ctx, textureManager.getTextureTable(), getBufferTable(), getSamplerTable());

    // If there are no instances, we should do nothing!
    if (m_instanceManager.getActiveCount() == 0) {
      // Clear the ray portal data before the next frame
      m_rayPortalManager.clear();
      return;
    }

    m_rayPortalManager.prepareSceneData(ctx);
    // Note: only main camera needs to be teleportation corrected as only that one is used for ray tracing & denoising
    m_rayPortalManager.fixCameraInBetweenPortals(m_cameraManager.getCamera(CameraType::Main));
    m_rayPortalManager.fixCameraInBetweenPortals(m_cameraManager.getCamera(CameraType::ViewModel));
    m_rayPortalManager.createVirtualCameras(m_cameraManager);
    const bool didTeleport = m_rayPortalManager.detectTeleportationAndCorrectCameraHistory(
      m_cameraManager.getCamera(CameraType::Main),
      m_cameraManager.isCameraValid(CameraType::ViewModel) ? &m_cameraManager.getCamera(CameraType::ViewModel) : nullptr);

    {
      const uint32_t previousStartInMediumMaterialIndexInCache = m_lastResolvedStartInMediumMaterialIndexInCache;
      std::optional<MaterialData> pendingStartInMediumMaterial;
      uint32_t persistentStartInMediumMaterialIndexInCache = kInvalidMaterialCacheIndex;
      bool clearStartInMediumMaterial = false;
      {
        std::lock_guard lock { m_startInMediumMaterialMutex };
        clearStartInMediumMaterial = m_pendingClearStartInMediumMaterial;
        m_pendingClearStartInMediumMaterial = false;
        if (m_pendingStartInMediumMaterial.has_value()) {
          pendingStartInMediumMaterial = std::move(m_pendingStartInMediumMaterial);
          m_pendingStartInMediumMaterial.reset();
        }
      }

      if (clearStartInMediumMaterial) {
        m_persistentStartInMediumMaterial.reset();
      }

      if (pendingStartInMediumMaterial.has_value()) {
        m_persistentStartInMediumMaterial = std::move(pendingStartInMediumMaterial);
      }

      if (m_persistentStartInMediumMaterial.has_value()) {
        assert(m_persistentStartInMediumMaterial->getType() == MaterialDataType::Translucent);
        const auto samplerIndex = trackSampler(getOrCreateExternalSampler());
        const auto surfaceMaterial = RtSurfaceMaterial(
          createTranslucentSurfaceMaterial(m_persistentStartInMediumMaterial->getTranslucentMaterialData(), samplerIndex, true));
        persistentStartInMediumMaterialIndexInCache = m_surfaceMaterialCache.track(surfaceMaterial);
      }

      m_startInMediumMaterialIndex_inCache = m_externalStartInMediumMaterialIndex_inCache != kInvalidMaterialCacheIndex
        ? m_externalStartInMediumMaterialIndex_inCache
        : persistentStartInMediumMaterialIndexInCache != kInvalidMaterialCacheIndex
          ? persistentStartInMediumMaterialIndexInCache
          : m_fogStartInMediumMaterialIndex_inCache;

      if (m_startInMediumMaterialIndex_inCache != kInvalidMaterialCacheIndex &&
          m_startInMediumMaterialIndex_inCache >= m_surfaceMaterialCache.getObjectTable().size()) {
        Logger::debug(str::format(
          "[RTX] Ignoring stale camera medium material cache index ", m_startInMediumMaterialIndex_inCache,
          " on frame ", m_device->getCurrentFrameId(),
          "; surfaceMaterialCacheSize=", m_surfaceMaterialCache.getObjectTable().size()));
        m_startInMediumMaterialIndex_inCache = kInvalidMaterialCacheIndex;
      }

      if (m_startInMediumMaterialIndex_inCache != previousStartInMediumMaterialIndexInCache) {
        Logger::debug(str::format(
          "[RTX] View history invalidated due to camera medium change on frame ", m_device->getCurrentFrameId(),
          ": previousStartInMediumInCache=", previousStartInMediumMaterialIndexInCache,
          ", currentStartInMediumInCache=", m_startInMediumMaterialIndex_inCache,
          ", cleared=", clearStartInMediumMaterial ? "true" : "false",
          ", pendingSet=", pendingStartInMediumMaterial.has_value() ? "true" : "false"));
        m_cameraManager.getMainCamera().invalidateViewHistory(m_device->getCurrentFrameId());
      }
      m_lastResolvedStartInMediumMaterialIndexInCache = m_startInMediumMaterialIndex_inCache;
    }

    if (m_cameraManager.isCameraCutThisFrame()) {
      // Ignore camera cut events on teleportation so we don't flush the caches
      if (!didTeleport) {
        Logger::info(str::format("Camera cut detected on frame ", m_device->getCurrentFrameId()));
        m_enqueueDelayedClear = true;
      }
    }

    // Initialize/remove opacity micromap manager
    if (RtxOptions::getEnableOpacityMicromap()) {
      if (!m_opacityMicromapManager.get() || 
          // Reset the manager on camera cuts
          m_enqueueDelayedClear) {
        if (m_opacityMicromapManager.get())
          m_instanceManager.removeEventHandler(m_opacityMicromapManager.get());

        m_opacityMicromapManager = std::make_unique<OpacityMicromapManager>(m_device);
        m_instanceManager.addEventHandler(m_opacityMicromapManager->getInstanceEventHandler());
        // Seed candidates with instances that were added before the event handler was registered
        m_opacityMicromapManager->seedCandidates(m_instanceManager.getInstanceTable());
        Logger::info("[RTX] Opacity Micromap: enabled");
      }
    } else if (m_opacityMicromapManager.get()) {
      m_accelManager.invalidateOpacityMicromapBindings();
      m_instanceManager.notifySceneChanged();
      m_instanceManager.removeEventHandler(m_opacityMicromapManager.get());
      m_opacityMicromapManager = nullptr;
      Logger::info("[RTX] Opacity Micromap: disabled");
    }

    RtxParticleSystemManager& particles = m_device->getCommon()->metaParticleSystem();
    particles.simulate(ctx.ptr());

    m_instanceManager.findPortalForVirtualInstances(m_cameraManager, m_rayPortalManager);
    m_instanceManager.createViewModelInstances(ctx, m_cameraManager, m_rayPortalManager);
    m_instanceManager.createPlayerModelVirtualInstances(ctx, m_cameraManager, m_rayPortalManager);

    m_accelManager.mergeInstancesIntoBlas(ctx, execBarriers, textureManager.getTextureTable(), m_cameraManager, m_instanceManager, m_opacityMicromapManager.get());

    // Call on the other managers to prepare their GPU data for the current scene
    m_accelManager.prepareSceneData(ctx, execBarriers, m_instanceManager);
    m_lightManager.prepareSceneData(ctx, m_cameraManager);

    // Upload surface material buffer BEFORE the GPU culling dispatch so the
    // compute shader can copy template material entries to per-instance slots.
    // For PointInstancer duplicate entries we skip writeGPUData and advance past
    // the gap - the GPU shader will fill those slots.
    //
    // When the scene is unchanged (fast-skip path in mergeInstancesIntoBlas),
    // the surface order and material data are normally identical to last frame.
    // Baked terrain materials are updated independently of acceleration-structure
    // scene generation, so keep their surface-material upload live.
    const bool startInMediumStateChanged = m_startInMediumMaterialIndex_inCache != m_lastUploadedStartInMediumMaterialIndexInCache;
    const bool updateSurfaceMaterials =
      !m_accelManager.wasSceneUnchangedThisFrame() ||
      TerrainBaker::needsTerrainBaking() ||
      startInMediumStateChanged;
    if (updateSurfaceMaterials) {
      DxvkBufferCreateInfo matInfo;
      matInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
        | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
        | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      matInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
        | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      matInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT;

      if (m_surfaceMaterialCache.getTotalCount() > 0) {
        ScopedGpuProfileZone(ctx, "updateSurfaceMaterials");
        // Note: We duplicate the materials in the buffer so we don't have to do pointer chasing on the GPU
        size_t surfaceMaterialsGPUSize = m_accelManager.getSurfaceCount() * kSurfaceMaterialGPUSize;
        const uint32_t expectedSurfaceMaterialEntries = m_accelManager.getSurfaceCount()
          + (m_startInMediumMaterialIndex_inCache != kInvalidMaterialCacheIndex ? 1u : 0u);
        if (m_startInMediumMaterialIndex_inCache != kInvalidMaterialCacheIndex) {
          surfaceMaterialsGPUSize += kSurfaceMaterialGPUSize;
        }

        matInfo.size = align(surfaceMaterialsGPUSize, kBufferAlignment);
        if (m_surfaceMaterialBuffer == nullptr || matInfo.size > m_surfaceMaterialBuffer->info().size) {
          m_surfaceMaterialBuffer = m_device->createBuffer(matInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Surface Material Buffer");
        }

        std::size_t dataOffset = 0;
        uint32_t surfaceIndex = 0;
        std::vector<unsigned char> surfaceMaterialsGPUData(surfaceMaterialsGPUSize);
        for (auto&& pInstance : m_accelManager.getOrderedInstances()) {
          // For PointInstancer duplicates (entries beyond the template), skip
          // writeGPUData - the GPU culling shader copies the template material.
          const auto& surf = pInstance->surface;
          if (surf.instancesToObject != nullptr &&
              surf.surfaceIndexOfFirstInstance != SIZE_MAX &&
              surfaceIndex > surf.surfaceIndexOfFirstInstance) {
            dataOffset += kSurfaceMaterialGPUSize;
          } else {
            assert(surf.surfaceMaterialIndex < m_surfaceMaterialCache.getObjectTable().size());
            auto&& surfaceMaterial = m_surfaceMaterialCache.getObjectTable()[surf.surfaceMaterialIndex];
            surfaceMaterial.writeGPUData(surfaceMaterialsGPUData.data(), dataOffset, surfaceIndex);
          }
          surfaceIndex++;
        }

        if (m_startInMediumMaterialIndex_inCache != kInvalidMaterialCacheIndex) {
          auto&& surfaceMaterial = m_surfaceMaterialCache.getObjectTable()[m_startInMediumMaterialIndex_inCache];
          surfaceMaterial.writeGPUData(surfaceMaterialsGPUData.data(), dataOffset, surfaceIndex);
          m_startInMediumMaterialIndex = surfaceIndex;
          surfaceIndex++;
        } else {
          m_startInMediumMaterialIndex = SURFACE_INDEX_INVALID;
        }

        assert(surfaceIndex == expectedSurfaceMaterialEntries);
        assert(dataOffset == surfaceMaterialsGPUSize);
        assert(surfaceMaterialsGPUData.size() == surfaceMaterialsGPUSize);
        m_lastUploadedStartInMediumMaterialIndexInCache = m_startInMediumMaterialIndex_inCache;

        ctx->writeToBuffer(m_surfaceMaterialBuffer, 0, surfaceMaterialsGPUData.size(), surfaceMaterialsGPUData.data());
      }
    } else {
      m_startInMediumMaterialIndex = m_startInMediumMaterialIndex_inCache != kInvalidMaterialCacheIndex
        ? m_accelManager.getSurfaceCount()
        : SURFACE_INDEX_INVALID;
    }

    // GPU-driven PointInstancer culling: overwrites visible instance placeholders
    // in m_vkInstanceBuffer with proper transforms and masks, copies per-instance
    // surface and material data from templates. Must run after prepareSceneData
    // (which uploads placeholders) and before buildTlas.
    m_accelManager.dispatchPointInstancerCulling(ctx, m_cameraManager, m_surfaceMaterialBuffer);

    // Build the TLAS
    m_accelManager.buildTlas(ctx);

    // Todo: These updates require a lot of temporary buffer allocations and memcopies, ideally we should memcpy directly into a mapped pointer provided by Vulkan,
    // but we have to create a buffer to pass to DXVK's updateBuffer for now.
    // Skip when scene is unchanged - buffers from last frame are still valid.
    if (!m_accelManager.wasSceneUnchangedThisFrame()) {
      // Allocate the instance buffer and copy its contents from host to device memory
      DxvkBufferCreateInfo info;
      info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT;

      // Surface Material Extension Buffer
      if (m_surfaceMaterialExtensionCache.getTotalCount() > 0) {
        ScopedGpuProfileZone(ctx, "updateSurfaceMaterialExtensions");
        const auto surfaceMaterialExtensionsGPUSize = m_surfaceMaterialExtensionCache.getTotalCount() * kSurfaceMaterialGPUSize;

        info.size = align(surfaceMaterialExtensionsGPUSize, kBufferAlignment);
        info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (m_surfaceMaterialExtensionBuffer == nullptr || info.size > m_surfaceMaterialExtensionBuffer->info().size) {
          m_surfaceMaterialExtensionBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Surface Material Extension Buffer");
        }

        std::size_t dataOffset = 0;
        std::vector<unsigned char> surfaceMaterialExtensionsGPUData(surfaceMaterialExtensionsGPUSize);

        uint32_t surfaceIndex = 0;
        for (auto&& surfaceMaterialExtension : m_surfaceMaterialExtensionCache.getObjectTable()) {
          surfaceMaterialExtension.writeGPUData(surfaceMaterialExtensionsGPUData.data(), dataOffset, surfaceIndex);
          surfaceIndex++;
        }

        assert(dataOffset == surfaceMaterialExtensionsGPUSize);
        assert(surfaceMaterialExtensionsGPUData.size() == surfaceMaterialExtensionsGPUSize);

        ctx->writeToBuffer(m_surfaceMaterialExtensionBuffer, 0, surfaceMaterialExtensionsGPUData.size(), surfaceMaterialExtensionsGPUData.data());
      }

      // Volume Material buffer
      if (m_volumeMaterialCache.getTotalCount() > 0) {
        ScopedGpuProfileZone(ctx, "updateVolumeMaterials");
        const auto volumeMaterialsGPUSize = m_volumeMaterialCache.getTotalCount() * kVolumeMaterialGPUSize;

        info.size = align(volumeMaterialsGPUSize, kBufferAlignment);
        info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (m_volumeMaterialBuffer == nullptr || info.size > m_volumeMaterialBuffer->info().size) {
          m_volumeMaterialBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Volume Material Buffer");
        }

        std::size_t dataOffset = 0;
        std::vector<unsigned char> volumeMaterialsGPUData(volumeMaterialsGPUSize);

        for (auto&& volumeMaterial : m_volumeMaterialCache.getObjectTable()) {
          volumeMaterial.writeGPUData(volumeMaterialsGPUData.data(), dataOffset);
        }

        assert(dataOffset == volumeMaterialsGPUSize);
        assert(volumeMaterialsGPUData.size() == volumeMaterialsGPUSize);

        ctx->writeToBuffer(m_volumeMaterialBuffer, 0, volumeMaterialsGPUData.size(), volumeMaterialsGPUData.data());
      }
    }

    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
      VK_ACCESS_SHADER_READ_BIT);

    // Update stats
    m_device->statCounters().setCtr(DxvkStatCounter::RtxBlasCount, AccelManager::getBlasCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxBufferCount, m_bufferCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxTextureCount, textureManager.getTextureTable().size());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxReplacementTextureCount, textureManager.getActiveReplacementTextures());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxInstanceCount, m_instanceManager.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxSurfaceMaterialCount, m_surfaceMaterialCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxSurfaceMaterialExtensionCount, m_surfaceMaterialExtensionCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxVolumeMaterialCount, m_volumeMaterialCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxLightCount, m_lightManager.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxSamplers, m_samplerCache.getActiveCount());

    auto capturer = m_device->getCommon()->capturer();
    if (m_device->getCurrentFrameId() == m_beginUsdExportFrameNum) {
      capturer->triggerNewCapture();
    }
    capturer->step(ctx, ctx->getCommonObjects()->getLastKnownWindowHandle());

    // Clear the ray portal data before the next frame
    m_rayPortalManager.clear();

    // Check Anti-Culling Support:
    // When the game doesn't set up the View Matrix, we must disable Anti-Culling to prevent visual corruption.
    m_isAntiCullingSupported = (getCamera().getViewToWorld() != Matrix4d());
  }

  static_assert(std::is_same_v< decltype(RtSurface::objectPickingValue), ObjectPickingValue>);

  void SceneManager::submitWorldAnchoredInstancers(Rc<DxvkContext> ctx) {
    ScopedCpuProfileZone();

    if (!RtxOptions::enableWorldAnchoredInstancers()) {
      return;
    }

    // These groups are replacement geometry and have to answer to the replacement switches.
    //
    // Only their own enable was checked, which left them submitting while replacements were off. That is
    // wrong on its own terms -- turning replacements off should remove all replacement geometry, and this
    // is replacement geometry that merely lacks a game draw to hang off -- and it silently ruined every
    // exterior capture.
    //
    // A capture disables replacements first, and GameCapturer refuses to start otherwise, so a capture
    // should contain the game's own meshes for replacements to be authored against. With this path
    // unguarded it recorded the scatter instead: 20 Bald Cypress trees at roughly 5.6 million triangles
    // each, whose parts match the mod's assets exactly -- 2,282,108 triangles for BaldCypress_0/Clovers
    // and 1,094,010 for BaldCypress_0/mesh_005, both present in the capture under their own hashes at the
    // same 20 world positions. The engine submitted 3.4-5.9 million triangles for that view; the capture
    // held 90,296,668, and 56 meshes accounted for 89.9% of them.
    //
    // getEnableReplacementMeshes() rather than the individual option, because the individual one is
    // documented as requiring the global switch to have any effect.
    if (!RtxOptions::getEnableReplacementMeshes()) {
      return;
    }

    std::vector<const WorldAnchoredInstancerGroup*> groups = m_pReplacer->getWorldAnchoredInstancers();
    if (groups.empty()) {
      return;
    }

    const uint32_t frameId = m_device->getCurrentFrameId();
    const RtCamera& camera = getCamera();
    if (!camera.isValid(frameId)) {
      return;
    }

    // See the option's own description for why this exists and why it is not a fix.
    const bool drawUngated = RtxOptions::worldAnchoredInstancersDrawUngated();

    // Identity, because the placements baked into these instancers are already world-space.
    // buildReplacementMeshDrawCallState multiplies this by the group's own replacementToObject, and the
    // groups carry no transform of their own, so the per-instance transforms reach the TLAS unmodified.
    // That is the entire reason the bake resolves world positions up front instead of storing
    // anchor-relative ones like the authored form does.
    const Matrix4 objectToWorld;

    static MaterialData s_defaultWorldAnchoredMaterial(LegacyMaterialData::createDefault());

    uint32_t submitted = 0;
    uint32_t gatedOut = 0;
    uint32_t untagged = 0;
    uint32_t deferred = 0;

    for (const WorldAnchoredInstancerGroup* group : groups) {
      if (group->replacements.empty()) {
        continue;
      }

      // An untagged group cannot be scoped to a space. Drawing it anyway is the interior-bleed bug, so
      // it is refused even in the ungated bring-up mode -- that mode exists to check placement, not to
      // make unscopeable content appear.
      if (group->anchorMeshHash == 0) {
        ++untagged;
        continue;
      }

      // The gate: draw this group when the space it was authored in is the space being rendered.
      //
      // Two independent tests, because they cover different cases and neither covers both. Anything
      // painted onto a static passes the first on its own -- statics come from the same source data in
      // both engines and the host does draw them, so the test is exact and needs no notion of cells.
      // Anything painted onto captured terrain can only pass the second, because the host builds its own
      // terrain and those captured hashes are never drawn.
      //
      // Neither test refuses a location. Paint in an interior and it anchors to that interior's floor,
      // which passes the first test while indoors and fails it outdoors. Paint on exterior ground and the
      // second test admits it only in the cells whose terrain is on screen.
      const bool anchorDrawnThisFrame = isMeshHashUsedThisFrame(group->anchorMeshHash);
      const bool cellActiveThisFrame = group->hasCell
                                       && isTerrainCellActiveThisFrame(group->cellX, group->cellY);

      if (!anchorDrawnThisFrame && !cellActiveThisFrame && !drawUngated) {
        ++gatedOut;

        // Out of scope means "do not draw this", not "destroy this".
        //
        // Skipping the group outright also skipped the frameLastSeen refresh below, which is the only
        // thing keeping its ReplacementInstance out of the collector. So crossing the gate did not merely
        // stop a draw, it condemned the built geometry -- and it did so for every group crossing at once,
        // because the gate is driven by which terrain cells are live.
        //
        // That is what the GPU faults were. Leaving an area put 1773 groups out of scope in one frame,
        // garbageCollectReplacementInstances retired all of them a few frames later, and the driver
        // faulted reading a merged BLAS that had just been destroyed -- Error_DMA_PageFault against
        // "BLAS Merged", 16 MB to 356 MB, in four separate dumps. Coming back into scope built 1153
        // groups in a single frame, which is the same burst inverted, and crashed the same way.
        //
        // Refreshing the stamp keeps the geometry resident and unsubmitted, so a group crossing the gate
        // costs a skipped draw and nothing else. Raising rtx.numFramesToKeepBLAS was tried first and did
        // not help, which fits: a wider window delays a mass destruction without preventing it.
        //
        // Nothing is created here. findReplacementInstanceByIdentity never allocates, so a group that has
        // never been in scope still has no instance and does not get one until it is genuinely drawn --
        // the whole set is not built up front.
        if (ReplacementInstance* outOfScope
            = m_drawCallTracker.findReplacementInstanceByIdentity(group->identityHash)) {
          outOfScope->frameLastSeen = frameId;
        }
        continue;
      }

      DrawCallState worldDrawCall;
      {
        DrawCallTransforms& transforms = worldDrawCall.modifyTransformData();
        transforms.objectToWorld = objectToWorld;
        transforms.worldToView = camera.getWorldToViewf();
        transforms.viewToProjection = camera.getViewToProjectionf();
        transforms.objectToView = transforms.worldToView;  // objectToWorld is identity
      }

      // Keyed on the group's USD path hash rather than on its transform or position, which is what the
      // draw-anchored paths use. Every group here sits at the same identity transform, so a
      // position-derived key would collide across all of them.
      const ReplacementInstance::LookupKey key {
        group->identityHash,
        group->identityHash,  // its own spatial bucket; these never move, so nothing shares one
        kEmptyHash,
        kEmptyHash,
        Vector3 { 0.0f, 0.0f, 0.0f },
        objectToWorld
      };

      ReplacementInstance* replacementInstance = m_drawCallTracker.findOrCreateReplacementInstance(key);
      if (replacementInstance == nullptr) {
        continue;
      }

      const std::vector<AssetReplacement>* pReplacements = &group->replacements;

      // Same preserve-or-rebuild decision the draw-anchored paths make, and for the same reason: the
      // dynamic path rebuilds each group's acceleration structure, and doing that every frame for two
      // thousand groups would be far more expensive than the geometry itself.
      //
      // Nothing about a world-anchored group changes between frames -- fixed transform, fixed geometry,
      // fixed material -- so once built it should preserve indefinitely. The conditions below are the
      // ones that can still legitimately force a rebuild.
      const bool alreadyWired = replacementInstance->root.getUntyped() != nullptr
                                && replacementInstance->activeReplacements == pReplacements;
      const bool secondSubmissionThisFrame = (replacementInstance->frameLastSeen == frameId);
      const bool cachedTexturesValidForPreserve =
          m_device->getCommon()->getTextureManager().getTextureCacheGeneration() ==
          m_textureCacheGenerationValidForPreserve;

      const bool usePreservePath =
          RtxOptions::enablePreservePath() &&
          alreadyWired &&
          replacementInstance->dirtyFlags.isClear() &&
          !RtxOptionManager::isDrawcallTranslationInvalid() &&
          !secondSubmissionThisFrame &&
          cachedTexturesValidForPreserve;

      // Ration first builds across frames, the same way submitExternalDraw does and for the same reason.
      //
      // The refresh above stops the gate from condemning groups, so scope transitions no longer destroy
      // anything. It does nothing for the opposite direction: any event that invalidates the whole set at
      // once still puts every rebuild in one frame. Switching enhanced meshes back on does that, a mod
      // reload does it, and so does startup. 1153 groups building in a single frame is what faulted the
      // driver, and that number came from this path.
      //
      // Only first builds are paced. A group that is merely being re-preserved costs nothing, and a group
      // already wired is not rebuilt here at all.
      //
      // A deferred group is not drawn this frame. There is nothing else it could draw -- unlike an API
      // mesh, which falls back to its own geometry -- so groundcover appears over the following frames
      // instead of all at once. frameLastSeen still has to be refreshed or the collector retires the
      // instance before its turn arrives, which would restart the build on a fresh instance every frame
      // and never finish.
      if (!usePreservePath && !alreadyWired) {
        if (m_worldAnchoredBuildFrame != frameId) {
          m_worldAnchoredBuildFrame = frameId;
          m_worldAnchoredBuildsThisFrame = 0;
        }

        const uint32_t buildBudget = RtxOptions::maxWorldAnchoredInstancerBuildsPerFrame();
        if (buildBudget != 0 && m_worldAnchoredBuildsThisFrame >= buildBudget) {
          replacementInstance->frameLastSeen = frameId;
          ++deferred;
          continue;
        }

        ++m_worldAnchoredBuildsThisFrame;
      }

      if (usePreservePath) {
        preserveReplacementInstance(ctx, worldDrawCall, pReplacements, replacementInstance);
      } else {
        if (replacementInstance->activeReplacements != pReplacements) {
          replacementInstance->clear();
        }
        replacementInstance->dirtyFlags.clr(ReplacementInstance::kDynamicFeatureMask);
        MaterialData renderMaterialData = s_defaultWorldAnchoredMaterial;
        drawReplacements(ctx, &worldDrawCall, pReplacements, renderMaterialData, replacementInstance);
      }

      // drawReplacements does not do this, and without it the draw call tracker collects the instance on
      // a timer and rebuilds it, and never treats it as stable.
      replacementInstance->frameLastSeen = frameId;
      ++submitted;
    }

    // Reported when the figures change, not once and not every frame.
    //
    // ONCE was wrong here: the first frame has no terrain drawn yet, so it permanently recorded "0 drawn,
    // 0 cells live" and could never show either the steady state or what happens on going indoors --
    // which is the only thing this number is for. Per-frame would be 60 lines a second of mostly
    // unchanged values. On-change gives one line per transition, which is exactly the interesting event.
    {
      static uint32_t s_lastSubmitted = ~0u;
      static size_t s_lastCells = ~0ull;
      static uint32_t s_lastDeferred = ~0u;
      const size_t liveCells = m_currentFrameTerrainCells.size();

      if (RtxOptions::ForkLogging::scatterSubmit()
          && (submitted != s_lastSubmitted || liveCells != s_lastCells || deferred != s_lastDeferred)) {
        s_lastSubmitted = submitted;
        s_lastCells = liveCells;
        s_lastDeferred = deferred;
        Logger::info(str::format(
            "[RTX Scatter] world-anchored submit: ", submitted, " groups drawn, ", gatedOut,
            " out of scope, ", untagged, " untagged, ", deferred, " build deferred; ",
            liveCells, " terrain cells live",
            drawUngated ? " [UNGATED BRING-UP MODE -- scoping off, expect interior bleed]" : ""));
      }
    }
  }

  void SceneManager::submitExternalDraw(const Rc<DxvkContext>& ctx, std::unique_ptr<ExternalDrawState> pstate) {
    ScopedCpuProfileZone();

    Rc<DxvkSampler> externalSampler = getOrCreateExternalSampler();

    auto& state = *pstate;

    {
      state.drawCall.modifyMaterialData().samplers[0] = externalSampler;
      state.drawCall.modifyMaterialData().samplers[1] = externalSampler;
    }
    {
      const RtCamera& rtCamera = ctx->getCommonObjects()->getSceneManager().getCameraManager()
        .getCamera(state.cameraType);
      state.drawCall.modifyTransformData().worldToView = rtCamera.getWorldToViewf();
      state.drawCall.modifyTransformData().viewToProjection = rtCamera.getViewToProjectionf();
      state.drawCall.modifyTransformData().objectToView = state.drawCall.getTransformData().worldToView * state.drawCall.getTransformData().objectToWorld;
    }

    if (!state.gpuInstancingTransforms.empty()) {
      state.drawCall.modifyTransformData().instancesToObject =
        std::make_shared<const std::vector<Matrix4>>(std::move(state.gpuInstancingTransforms));
    }

    const XXH64_hash_t meshHash = reinterpret_cast<XXH64_hash_t>(state.mesh);

    // Fetch submeshes once — they drive both the replacement path (needs submeshes[0]
    // as geometry template) and the default iteration path.
    const std::vector<RasterGeometry>& submeshes = m_pReplacer->accessExternalMesh(state.mesh);
    if (submeshes.empty()) {
      Logger::err(str::format("[RTX-Mesh] External mesh has no submeshes: 0x", std::hex, meshHash, std::dec));
      return;
    }

    // Persistence-tracking setup happens before the replacement-lookup early-out
    // so the same ReplacementInstance can be threaded through both paths —
    // drawReplacements() requires a non-null instance and uses it to drive
    // RtInstance reuse across frames for the replacement primitives.
    const XXH64_hash_t identityHash = state.computeExternalDrawIdentityHash();
    const XXH64_hash_t spatialMapHash = spatialMapHashForExternalDrawMesh(state.mesh);
    const Matrix4& xform = state.drawCall.getTransformData().objectToWorld;
    const XXH64_hash_t matHash = state.drawCall.getMaterialData().getHash();
    const Vector3 worldPos = xform[3].xyz();

    const ReplacementInstance::LookupKey externalKey { identityHash, spatialMapHash, matHash, kEmptyHash, worldPos, xform };
    ReplacementInstance* replacementInstance = m_drawCallTracker.findOrCreateReplacementInstance(externalKey);
    replacementInstance->dirtyFlags.clr(ReplacementInstance::kDynamicFeatureMask);

    std::vector<AssetReplacement>* pReplacements = fork_hooks::externalDrawMeshReplacement(*m_pReplacer, meshHash);

    // Ration first-time replacement builds across frames.
    //
    // A mesh being sighted for the first time has to take the dynamic path, which builds its
    // replacement's acceleration structure. Every frame after that reuses it via the preserve path
    // above, so the cost is once per mesh -- but entering a cell sights every mesh in it at once.
    // Ald'ruhn binds 197 distinct replacements, several of them high-poly, and putting all of those
    // builds in one frame stalled the GPU for 19 seconds and lost the device to a driver reset.
    //
    // Deferring means falling through to the ordinary submission below, so the mesh draws its own
    // geometry this frame and tries again next. The scene fills in over a fraction of a second instead
    // of hitching, and nothing disappears while it waits, which is what skipping the draw entirely
    // would have caused.
    //
    // Not synchronised: external draws are submitted from one thread, and the counter only paces work.
    // Being off by one either way costs nothing.
    // "Already built" has to mean built *for this replacement set*, not merely that prims exist.
    //
    // Testing root alone was wrong and made things worse. Deferring falls through to the ordinary
    // submission, which wires prims for the original geometry and sets root -- so on the next frame the
    // mesh looked built, took the replacement path anyway, found activeReplacements mismatched, cleared
    // the instance and rebuilt it. Every mesh cycled wire, clear, rebuild, which flickered, and because
    // nothing was actually deferred past one frame the builds still all landed together.
    const bool replacementAlreadyWired = replacementInstance->root.getUntyped() != nullptr
        && replacementInstance->activeReplacements == pReplacements;
    bool deferReplacement = false;
    if (pReplacements != nullptr && !replacementAlreadyWired) {
      // Counted per distinct mesh, not per instance.
      //
      // The expense being paced is the acceleration structure build, and that is per geometry: the second
      // instance of a mesh reuses the first one's BLAS from the draw call cache. A ReplacementInstance
      // exists per mesh and transform, so a cell holds thousands of them against a couple of hundred
      // distinct meshes -- counting instances therefore throttled cheap work and starved the expensive
      // work, which is why nothing appeared. Instances of a mesh already admitted this frame pass freely.
      const uint32_t budget = RtxOptions::maxExternalReplacementBuildsPerFrame();
      const uint32_t frameId = m_device->getCurrentFrameId();
      if (m_externalReplacementBuildFrame != frameId) {
        m_externalReplacementBuildFrame = frameId;
        m_externalReplacementBuildMeshes.clear();
      }
      if (m_externalReplacementBuildMeshes.find(meshHash) != m_externalReplacementBuildMeshes.end()) {
        // Already admitted this frame; its build is in flight and further instances are cheap.
      } else if (budget != 0 && m_externalReplacementBuildMeshes.size() >= budget) {
        deferReplacement = true;
        ++m_externalReplacementDeferrals;
      } else {
        m_externalReplacementBuildMeshes.insert(meshHash);
      }

      // Report what the pacing is actually doing, because three attempts at it have now been wrong for
      // three different reasons and the failures are indistinguishable on screen: a starved budget and a
      // replacement that never binds both show the original mesh.
      //
      // Cumulative deferrals is the figure that separates them. It should climb while a cell fills in and
      // then stop. Still climbing steadily means meshes are never becoming wired, so they re-enter the
      // budget every frame and starve each other -- which is a bug here, not a budget that is too small.
      if (m_externalReplacementDeferrals - m_externalReplacementDeferralsLogged >= 20000) {
        m_externalReplacementDeferralsLogged = m_externalReplacementDeferrals;
        Logger::info(str::format("[RTX-Replacement] pacing: budget ", budget,
            ", distinct meshes admitted this frame ", m_externalReplacementBuildMeshes.size(),
            ", cumulative deferred draws ", m_externalReplacementDeferrals));
      }
    }

    if (pReplacements != nullptr && !deferReplacement) {
      // Copy the DrawCallState so we don't mutate the caller's state. Point geometryData
      // at submeshes[0] as the replacement geometry template, clear externalMaterial so
      // the USD replacement material takes precedence, and use a neutral default material
      // since the replacement will provide its own.
      DrawCallState replacementDrawCall = state.drawCall;
      RasterGeometry& replacementGeometry = replacementDrawCall.modifyGeometryData();
      replacementGeometry = submeshes[0];
      replacementGeometry.cullMode = state.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
      replacementGeometry.externalMaterial = nullptr;

      // Resolve the host's material the way the ordinary submission below does, from the submesh's own
      // external material handle.
      //
      // determineMaterialData(state.drawCall) was wrong here and is why replaced surfaces stayed white:
      // an API draw carries no useful material on the DrawCallState at this point. The material arrives
      // through submeshes[i].externalMaterial and is resolved in the loop further down, which this branch
      // returns before reaching -- so the base being merged under the replacement was empty, leaving the
      // surface with no albedo, which is exactly the failure the merge exists to prevent.
      //
      // externalDrawMaterialReplacement is applied first so a material-level replacement for this surface
      // still contributes, matching the ordinary path. It may repoint the pointer into mergeStorage, so
      // that storage has to outlive the copy taken from it.
      //
      // drawReplacements only overwrites this when the replacement carries its own materialData, and it
      // assigns wholesale with no merge. A pack entry that authors no material -- or a partial `over` that
      // sets nothing but roughness -- therefore inherited whatever was passed in here, and a blank
      // LegacyMaterialData has no albedo texture, so the replacement rendered black. Invisible in a dark
      // interior, which is exactly how this presented: the original mesh correctly suppressed, high-poly
      // geometry confirmed built and instanced at the right world position, and nothing on screen.
      //
      // The same mistake on the material path is already fixed and documented in
      // fork_hooks::externalDrawMaterialReplacement, which merges the replacement over the host's material
      // for this reason. This is the geometry path's equivalent.
      MaterialData hostMergeStorage;
      const MaterialData* pHostMaterial = m_pReplacer->accessExternalMaterial(submeshes[0].externalMaterial);
      if (pHostMaterial != nullptr) {
        fork_hooks::externalDrawMaterialReplacement(*m_pReplacer, pHostMaterial, hostMergeStorage);
      }
      static MaterialData s_defaultExternalMaterial(LegacyMaterialData::createDefault());
      MaterialData renderMaterialData
          = (pHostMaterial != nullptr) ? *pHostMaterial : s_defaultExternalMaterial;

      // Reuse last frame's work when nothing about this draw changed, instead of rebuilding every frame.
      //
      // drawReplacements is the dynamic path: a full geometry cache pass and instance update, which for a
      // replacement means rebuilding its acceleration structure. Taking it unconditionally is what made a
      // replaced mesh cost a frame or more to look at -- the census office shield is 1,926,001 triangles and
      // was being rebuilt from scratch every frame it stayed in view.
      //
      // The gate mirrors the D3D9 path's usePreservePath, minus the conditions that cannot arise here.
      // Terrain cascades and override-material particle systems belong to legacy draws; a particle emitter
      // submitted through the API takes the particle path and never reaches this branch.
      //
      // Transform drift needs no dirty flag on this path because the transform is part of the external
      // lookup key: an object that moves resolves to a different ReplacementInstance rather than a stale
      // one. alreadyWired keeps the first sighting on the dynamic path, since preserving requires prims
      // that only a dynamic pass creates.
      const uint32_t currentFrameId = m_device->getCurrentFrameId();
      const bool secondSubmissionThisFrame = (replacementInstance->frameLastSeen == currentFrameId);
      const bool activeReplacementsMatch = replacementInstance->activeReplacements == pReplacements;
      const bool alreadyWired = replacementAlreadyWired;
      const bool cachedTexturesValidForPreserve =
          m_device->getCommon()->getTextureManager().getTextureCacheGeneration() ==
          m_textureCacheGenerationValidForPreserve;

      // A sibling draw may already have rebound a shared BlasEntry to its own data this frame, in which
      // case the cached buffers no longer describe this draw and preserving them would render the wrong
      // geometry. Same check the D3D9 path makes, for the same reason.
      auto blasAlreadyTouchedByOtherDraw = [replacementInstance, currentFrameId]() -> bool {
        for (const auto& prim : replacementInstance->prims) {
          RtInstance* inst = prim.getInstance();
          if (inst == nullptr) {
            continue;
          }
          BlasEntry* pBlas = inst->getBlas();
          if (pBlas != nullptr && pBlas->frameLastTouched == currentFrameId) {
            return true;
          }
        }
        return false;
      };

      const bool usePreservePath =
          RtxOptions::enablePreservePath() &&
          alreadyWired &&
          activeReplacementsMatch &&
          replacementInstance->dirtyFlags.isClear() &&
          !RtxOptionManager::isDrawcallTranslationInvalid() &&
          !secondSubmissionThisFrame &&
          !blasAlreadyTouchedByOtherDraw() &&
          cachedTexturesValidForPreserve;

      if (usePreservePath) {
        preserveReplacementInstance(ctx, replacementDrawCall, pReplacements, replacementInstance);
      } else {
        if (!activeReplacementsMatch) {
          replacementInstance->clear();
        }
        replacementInstance->dirtyFlags.clr(ReplacementInstance::kDynamicFeatureMask);
        drawReplacements(ctx, &replacementDrawCall, pReplacements, renderMaterialData, replacementInstance);
      }

      // Record that this replacement was submitted this frame, which drawReplacements does not do and this
      // branch previously returned without doing.
      //
      // Everything downstream that decides whether a replacement survives reads frameLastSeen. The draw
      // call tracker destroys a ReplacementInstance once frameLastSeen falls numFramesToKeepObjects behind
      // the current frame, and treats one as stable only when frameLastSeen has advanced past frameCreated.
      // Left at its initial value both of those go the wrong way: the instance is collected on a timer and
      // rebuilt from scratch, and it is never once considered stable.
      //
      // Small replacements survive that badly enough to flicker. A large one does not survive it at all --
      // the census office shield is 1,926,001 triangles, so its acceleration structure has no chance to
      // finish before the instance backing it is torn down, which is why it was confirmed built, instanced,
      // materialled and correctly placed yet never appeared on screen.
      //
      // The remaining fields mirror what the D3D9 path stores after its own call to drawReplacements. They
      // are the inputs to next frame's dirty-flag comparison, so leaving them stale makes an unchanged
      // draw look changed and forces a rebuild that was not needed.
      replacementInstance->frameLastSeen = m_device->getCurrentFrameId();
      replacementInstance->categoryFlags = state.drawCall.getCategoryFlags().raw();
      replacementInstance->isSkinned = state.drawCall.getSkinningState().numBones > 0;
      replacementInstance->textureTransform = state.drawCall.getTransformData().textureTransform;
      replacementInstance->texgenMode = state.drawCall.getTransformData().texgenMode;

      // Anti-culling needs an object-space extent and the transform it was measured in. The submeshes are
      // the host's own geometry rather than the replacement's, which is what the non-replacement path below
      // uses too, and is the right frame of reference: it is where the game put the object.
      AxisAlignedBoundingBox replacementBBox;
      for (size_t i = 0; i < submeshes.size(); i++) {
        replacementBBox.unionWith(submeshes[i].boundingBox);
      }
      if (replacementBBox.isValid()) {
        replacementInstance->geometryBoundingBox = replacementBBox;
        replacementInstance->objectToWorld = xform;
      }
      return;
    }

    // Falling through to the ordinary submission has to retire a replacement this instance is still
    // wired to, or the replacement stays on screen after it stops being wanted.
    //
    // The teardown that does this lives inside the branch above, so it was only reachable while a
    // replacement was being drawn. Every route out of that branch -- the lookup returning nothing because
    // enhanced meshes were switched off, a hot reload or variant change pointing at a different set --
    // skipped it, and the loop below only rebinds prims 0 through submeshes.size()-1. A replacement
    // almost always has more prims than the host mesh has submeshes, so the surplus was left holding
    // replacement geometry, still submitted, with frameLastSeen refreshed each frame by the instance
    // being matched, and no material pass ever reaching it again.
    //
    // On screen that is a replacement that will not go away, untextured because its material is no longer
    // resolved, flickering against the original mesh now drawn underneath it. It selects for multi-part
    // assets: a single-submesh object's replacement occupies prim 0 and gets overwritten below, which is
    // why simple objects reverted correctly and trees and doors did not. The D3D9 path cannot reach this
    // because its equivalent null case is an else branch that clears first.
    //
    // Conditioned on activeReplacements being non-null rather than on a plain mismatch, because the
    // deferral above also lands here. A first sighting waiting on the build budget has replacements
    // pending but none attached, and clearing it every frame would destroy and rebuild the original
    // geometry's prims for as long as it waited -- the same wire, clear, rebuild churn described above.
    // Non-null means drawReplacements actually ran and there is replacement state to retire. clear()
    // resets the pointer, so this fires once per transition rather than every frame.
    if (replacementInstance->activeReplacements != nullptr
        && replacementInstance->activeReplacements != pReplacements) {
      replacementInstance->clear();
    }

    AxisAlignedBoundingBox geometryBBox;

    for (size_t i = 0; i < submeshes.size(); i++) {
      state.drawCall.overrideGeometryData(&submeshes[i]);
      state.drawCall.overrideCullMode(state.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT);

      // Reconcile the skinning data's bone count with the geometry now that the geometry exists.
      //
      // This is the API equivalent of the last line of DrawCallState::finalizeSkinningData, which keeps
      // the two in step for legacy draws -- and which never runs here, because it is gated on
      // futureSkinningData being valid and an API mesh has no async skinning future.
      //
      // The ordering is the whole problem. remixapi_DrawInstance builds the draw state from
      // InstanceInfoBoneTransformsEXT and sets skinningData.numBonesPerVertex from
      // geometryData.numBonesPerVertex, but at that point no geometry has been attached, so it copies a
      // zero. The submesh assigned on the line above is what actually carries the real count.
      //
      // Rendering was unaffected and so this stayed hidden: the GPU skinning dispatch reads the bone count
      // from geometryData (rtx_geometry_utils.cpp), which was always right. Only consumers that trust
      // SkinningData saw the zero -- the game capturer being the one that found it, silently emitting
      // skinned meshes with no weights or indices at all.
      if (state.drawCall.getSkinningState().numBones > 0) {
        state.drawCall.modifySkinningData().numBonesPerVertex
          = state.drawCall.getGeometryData().numBonesPerVertex;
      }

      XXH64_hash_t textureHash = 0;

      // Storage for a replacement merged over the host's material. Declared here so it outlives every use
      // of `material` below, which may point into it. Same pattern as tmpMaterialData in submitDrawState.
      MaterialData mergedMaterialData;

      const MaterialData* material = m_pReplacer->accessExternalMaterial(submeshes[i].externalMaterial);
      if (material != nullptr) {
        // Identity first, while `material` is still the host's own. The merge below can repoint it at a
        // material whose albedo the USD supplied, and that texture is one the host never submitted -- so
        // reading the identity afterwards loses both category matching and the developer menu's texture
        // tagging for every replaced surface. See externalDrawTextureIdentity in rtx_fork_hooks.h.
        textureHash = fork_hooks::externalDrawTextureIdentity(material);

        fork_hooks::externalDrawMaterialReplacement(*m_pReplacer, material, mergedMaterialData);

        // Published as the albedo texture hash, which is what Remix's D3D9 path publishes:
        // LegacyMaterialData::updateCachedHash is literally colorTextures[0].getImageHash().
        //
        // This value is the material's public name. It is what the capturer writes as mat_<hash> --
        // rtx_game_capturer reads material.getHash(), and already notes it equals the instance's material
        // hash for D3D9 but differs for API submissions -- and therefore what a pack authored against a
        // Morrowind capture must match to be reachable in the toolkit at all. Publishing the host's richer
        // identity instead shared almost no names with such a pack: 2 of 276 material keys in an OpenMW
        // interior capture were addressable by the pack's 1605, against 131 of 244 for MGE-XE. Nearly
        // every material replacement in the pack was invisible and uneditable.
        //
        // Deliberately not the same change as altering the material's identity, which was tried and
        // rejected on measurement -- see the note at the host's createTexturedMaterial call. That would key
        // the runtime's material cache on albedo alone and collapse surfaces differing only in blend or
        // alpha test onto whichever was created first. This renames only, so the cache stays keyed on the
        // full identity, every surface state stays distinct, and in-game rendering does not move.
        // Replacement binding is unaffected either way, since externalDrawMaterialReplacement already
        // falls back to the albedo hash.
        //
        // The cost is capture fidelity: two materials sharing an albedo now occupy one mat_ prim, so
        // editing it in the toolkit edits both. That is not a new limitation -- it is the one D3D9 has
        // always had, and the assumption every pack authored against MGE-XE was built on.
        //
        // Falls back to the material's own hash when there is no albedo to name it by, so a material with
        // no colour texture keeps a distinct name instead of collapsing onto zero.
        state.drawCall.modifyMaterialData().setHashOverride(
            textureHash != 0 ? textureHash : material->getHash());

        fork_hooks::externalDrawTextureCategories(textureHash, state.drawCall);

        // After the categories, because this one is gated on InstanceCategories::Terrain having just been
        // applied. Repoints `material` at the baked terrain material on success, which is why it comes
        // before the materialData binding below.
        fork_hooks::externalDrawTerrainBake(ctx, *this, state.drawCall, material);
      }

      const RtxParticleSystemDesc* pParticles = nullptr;
      if (state.optionalParticleDesc.has_value()) {
        pParticles = &state.optionalParticleDesc.value();
      }

      fork_hooks::externalDrawObjectPicking(*m_device, state.drawCall, textureHash, *this);

      RtInstance* existingInstance = (replacementInstance->prims.size() > i)
          ? replacementInstance->prims[i].getInstance() : nullptr;

      static MaterialData defaultMaterialData(LegacyMaterialData::createDefault());
      auto& materialData = material != nullptr ? *material : defaultMaterialData;

      RtInstance* instance = processDrawCallState(ctx, state.drawCall, materialData, *replacementInstance, existingInstance, pParticles);

      {
        // Reports what the runtime actually made of this draw. See fork_hooks::ExternalDrawReport --
        // none of this is observable from the API side, which is why an API host can watch healthy
        // submit counts produce an empty image.
        fork_hooks::ExternalDrawReport report {};
        report.accepted = instance != nullptr;
        report.meshHash = meshHash;
        report.vertexCount = submeshes[i].vertexCount;
        report.indexCount = submeshes[i].indexCount;
        const Vector3 reportPos = state.drawCall.getTransformData().objectToWorld[3].xyz();
        report.worldPos[0] = reportPos.x;
        report.worldPos[1] = reportPos.y;
        report.worldPos[2] = reportPos.z;
        if (instance != nullptr) {
          report.instanceMask = instance->getVkInstance().mask;
          report.hidden = instance->isHidden();
          report.fullyOpaque = instance->surface.alphaState.isFullyOpaque;
          report.alphaTestType = static_cast<uint32_t>(instance->surface.alphaState.alphaTestType);
        }
        report.tlasSurfaceCount = m_accelManager.getSurfaceCount();
        fork_hooks::noteExternalDraw(report);
      }

      if (instance != nullptr) {
        if (replacementInstance->root.getUntyped() == nullptr) {
          replacementInstance->setup(PrimInstance(instance, PrimInstance::Type::Instance), submeshes.size(), nullptr);
        }
        if (replacementInstance->prims.size() > i &&
            replacementInstance->prims[i].getUntyped() != instance) {
          instance->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, instance,
              PrimInstance::Type::Instance);
        }
      }

      geometryBBox.unionWith(submeshes[i].boundingBox);
    }

    replacementInstance->frameLastSeen = m_device->getCurrentFrameId();

    if (geometryBBox.isValid()) {
      replacementInstance->geometryBoundingBox = geometryBBox;
      replacementInstance->objectToWorld = xform;
    }
  }

  void SceneManager::destroyExternalMesh(remixapi_MeshHandle handle) {
    if (handle) {
      m_drawCallTracker.removeReplacementInstancesWithSpatialMapHash(
          spatialMapHashForExternalDrawMesh(handle));
      m_pReplacer->destroyExternalMesh(handle, m_device->getCurrentFrameId());
    }
  }

  namespace {
    bool ifTrue_andThenSetFalse(std::atomic_bool& atomicBool) {
      bool expected = true;
      if (atomicBool.compare_exchange_strong(expected, false)) {
        return true;
      }
      return false;
    }
  } // unnamed

  void SceneManager::requestTextureVramFree() {
    m_forceFreeTextureMemory.store(true);
  }

  void SceneManager::requestVramCompaction() {
    m_forceFreeUnusedDxvkAllocatorChunks.store(true);
  }

  void SceneManager::manageTextureVram() {
    bool freeUnused = false;
    bool freeTextures = false;
    {
      if (ifTrue_andThenSetFalse(m_forceFreeTextureMemory)) {
        freeTextures = true;
        freeUnused = true;
      }
      if (ifTrue_andThenSetFalse(m_forceFreeUnusedDxvkAllocatorChunks)) {
        freeUnused = true;
      }
    }

    if (freeTextures) {
      m_device->getCommon()->getTextureManager().clear();

      if (m_opacityMicromapManager) {
        m_opacityMicromapManager->clear();
      }
    }

    if (freeUnused) {
      // DXVK doesnt free chunks for us by default (its high water mark) so force release some memory back to the system here.
      m_device->getCommon()->memoryManager().freeUnusedChunks();
    }
  }

  void SceneManager::printAllRtInstances() {
  #ifdef REMIX_DEVELOPMENT
    
    const auto& instances = m_instanceManager.getInstanceTable();
    Logger::info(str::format("=== Printing all RtInstances (", instances.size(), " total) ==="));
    
    for (size_t i = 0; i < instances.size(); ++i) {
      const RtInstance* instance = instances[i];
      if (instance != nullptr) {
        Logger::info(str::format("Instance ", i, ":"));
        instance->printDebugInfo();
      } else {
        Logger::warn(str::format("Instance ", i, ": nullptr"));
      }
    }
    
    Logger::info("=== End RtInstances Print ===");
  #endif
  }

  void SceneManager::trackReplacementMaterialHash(XXH64_hash_t materialHash) {
    if (materialHash != kEmptyHash) {
      m_currentFrameReplacementMaterialHashes[materialHash]++;
    }
  }

  bool SceneManager::isReplacementMaterialHashUsedThisFrame(XXH64_hash_t materialHash) const {
    return m_currentFrameReplacementMaterialHashes.find(materialHash) != m_currentFrameReplacementMaterialHashes.end();
  }

  uint32_t SceneManager::getReplacementMaterialHashUsageCount(XXH64_hash_t materialHash) const {
    auto it = m_currentFrameReplacementMaterialHashes.find(materialHash);
    return (it != m_currentFrameReplacementMaterialHashes.end()) ? it->second : 0;
  }

  void SceneManager::clearFrameReplacementMaterialHashes() {
    m_currentFrameReplacementMaterialHashes.clear();
  }

  // Morrowind's exterior cell size in world units. Matches the value the bake tags groups with; if one
  // changes the other has to.
  static constexpr float kExteriorCellSize = 8192.0f;

  static inline int32_t worldToCell(float v) {
    return static_cast<int32_t>(std::floor(v / kExteriorCellSize));
  }

  static inline uint64_t packCell(int32_t x, int32_t y) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) | static_cast<uint32_t>(y);
  }

  void SceneManager::trackTerrainCell(const DrawCallState& input) {
    if (!input.testCategoryFlags(InstanceCategories::Terrain)) {
      return;
    }

    // Which exterior cells are live is derived from the terrain actually being drawn, rather than from
    // the camera position or a host-supplied flag.
    //
    // That makes it answer the question that matters without being told: an interior draws no exterior
    // terrain, so no exterior cell is live and exterior groundcover cannot appear inside a building --
    // which is the bug this exists to prevent, and it cannot be defeated by an interior that happens to
    // sit at coordinates overlapping the exterior. It equally scopes one town's scatter away from
    // another's, since only the cells under visible ground are live.
    //
    // It is the same principle as gating on the anchor mesh being drawn, generalised to the cell for the
    // case where the exact mesh cannot be matched because the host generates its own terrain.
    const Vector3 origin = input.getTransformData().objectToWorld[3].xyz();
    m_currentFrameTerrainCells.insert(packCell(worldToCell(origin.x), worldToCell(origin.y)));
  }

  bool SceneManager::isTerrainCellActiveThisFrame(int32_t cellX, int32_t cellY) const {
    // Is any exterior terrain being drawn at all? This is the test that scopes interiors, and it is the
    // one that cannot be fooled: an interior draws none, so nothing exterior can appear inside it
    // regardless of what coordinates that interior happens to occupy.
    if (m_currentFrameTerrainCells.empty()) {
      return false;
    }

    // Then: is this cell near the camera?
    //
    // Membership in the drawn-terrain set is deliberately NOT used for the spatial part. Measured, that
    // set runs to 76-103 cells because distant land is drawn across the whole view distance, while the
    // authored scatter spans only 10 -- so every group passed, every frame, and one town's groundcover
    // was being submitted while standing in another. A set that large scopes nothing.
    //
    // Distance from the camera is the honest measure of "near you", and it is bounded by the radius the
    // point instancer culling already uses, so a group admitted here has instances that can actually
    // survive culling. Anything further away would be submitted, built and then culled on the GPU for
    // nothing.
    const Vector3 cameraPos = getCamera().getPosition();
    const int32_t cameraCellX = worldToCell(cameraPos.x);
    const int32_t cameraCellY = worldToCell(cameraPos.y);

    // One cell of slack beyond the camera's own, which is 8192 units -- comfortably past the default
    // culling radius, so this never clips scatter that would have been visible. It also absorbs the case
    // where the camera sits just inside one cell while looking across the boundary into the next.
    return std::abs(cellX - cameraCellX) <= 1 && std::abs(cellY - cameraCellY) <= 1;
  }

  void SceneManager::trackMeshHash(XXH64_hash_t meshHash) {
    if (meshHash != kEmptyHash) {
      m_currentFrameMeshHashes[meshHash]++;
    }
  }

  bool SceneManager::isMeshHashUsedThisFrame(XXH64_hash_t meshHash) const {
    return m_currentFrameMeshHashes.find(meshHash) != m_currentFrameMeshHashes.end();
  }

  uint32_t SceneManager::getMeshHashUsageCount(XXH64_hash_t meshHash) const {
    auto it = m_currentFrameMeshHashes.find(meshHash);
    return (it != m_currentFrameMeshHashes.end()) ? it->second : 0;
  }

  void SceneManager::clearFrameMeshHashes() {
    m_currentFrameMeshHashes.clear();
    m_currentFrameTerrainCells.clear();
  }

}  // namespace dxvk
