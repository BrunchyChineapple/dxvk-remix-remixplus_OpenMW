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
#pragma once

#include <mutex>
#include <optional>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <variant>

#include "../dxvk_buffer.h"
#include "../dxvk_image.h"
#include "../dxvk_staging.h"
#include "../dxvk_bind_mask.h"
#include "../util/util_hashtable.h"

#include "rtx_globals.h"
#include "rtx_retained_buffer_table.h"
#include "rtx_types.h"
#include "rtx_common_object.h"
#include "rtx_camera_manager.h"
#include "rtx_draw_call_cache.h"
#include "rtx_draw_call_tracker.h"
#include "rtx_sparse_unique_cache.h"
#include "rtx_light_manager.h"
#include "rtx_instance_manager.h"
#include "rtx_accel_manager.h"
#include "rtx_ray_portal_manager.h"
#include "rtx_bindless_resource_manager.h"
#include "rtx_objectpicking.h"
#include "rtx_mod_manager.h"
#include "graph/rtx_graph_manager.h"
#include "rtx_particle_system.h"
#include <d3d9types.h>

namespace dxvk 
{
class DxvkContext;
class DxvkDevice;
struct AssetReplacement;
struct AssetReplacer;
class OpacityMicromapManager;
class TerrainBaker;

// The resource cache can be *searched* by other users
class ResourceCache {
public:
  bool find(const RtSurfaceMaterial& surf, uint32_t& outIdx) const { return m_surfaceMaterialCache.find(surf, outIdx); }
  const RtSurfaceMaterial& get(const uint32_t index) const { return m_surfaceMaterialCache.getObjectTable()[index]; }

protected:
  RetainedBufferTable<RaytraceBuffer> m_bufferCache;

  struct SurfaceMaterialHashFn {
    size_t operator() (const RtSurfaceMaterial& mat) const {
      return (size_t)mat.getHash();
    }
  };
  SparseUniqueCache<RtSurfaceMaterial, SurfaceMaterialHashFn> m_surfaceMaterialCache;
  SparseUniqueCache<RtSurfaceMaterial, SurfaceMaterialHashFn> m_surfaceMaterialExtensionCache;
  fast_unordered_cache<uint32_t> m_preCreationSurfaceMaterialMap;

  struct VolumeMaterialHashFn {
    size_t operator() (const RtVolumeMaterial& mat) const {
      return (size_t)mat.getHash();
    }
  };
  SparseUniqueCache<RtVolumeMaterial, VolumeMaterialHashFn> m_volumeMaterialCache;

  struct SamplerHashFn {
    size_t operator() (const Rc<DxvkSampler>& sampler) const {
      return (size_t) sampler->hash();
    }
  };

  struct SamplerKeyEqual {
    bool operator()(const Rc<DxvkSampler>& lhs, const Rc<DxvkSampler>& rhs) const {
      return lhs->info() == rhs->info();
    }
  };

  SparseUniqueCache<Rc<DxvkSampler>, SamplerHashFn, SamplerKeyEqual> m_samplerCache;
};

struct ExternalDrawState {
  DrawCallState drawCall {};
  remixapi_MeshHandle mesh {};
  CameraType::Enum cameraType {};
  CategoryFlags categories {};
  bool doubleSided {};
  std::optional<RtxParticleSystemDesc> optionalParticleDesc {};
  std::vector<Matrix4> gpuInstancingTransforms {};

  // Draw-instance identity for ReplacementInstance lookup. Excludes per-frame camera matrices
  // (worldToView / viewToProjection / objectToView) filled in submitExternalDraw after hashing.
  XXH64_hash_t computeExternalDrawIdentityHash() const;
};

// Forward decls so SceneManager can friend the fork hook that needs access
// to private m_drawCallMeta state. See docs/fork-touchpoints.md.
class SceneManager;
namespace fork_hooks {
  void externalDrawObjectPicking(
    DxvkDevice& device,
    DrawCallState& drawCall,
    XXH64_hash_t textureHash,
    SceneManager& scene);
}

// Scene manager is a super manager, it's the interface between rendering and world state
// along with managing the operation of other caches, scene manager also manages the cache
// directly for "SceneObject"'s - which are "unique meshes/geometry", which map 1-to-1 with
// BLAS entries in raytracing terminology.
class SceneManager : public CommonDeviceObject, public ResourceCache {
  // Fork touchpoint: the external-draw object-picking hook needs access to
  // private m_drawCallMeta. Tracked as an inline tweak in
  // docs/fork-touchpoints.md.
  friend void fork_hooks::externalDrawObjectPicking(
    DxvkDevice&, DrawCallState&, XXH64_hash_t, SceneManager&);

public:
  SceneManager(SceneManager const&) = delete;
  SceneManager& operator=(SceneManager const&) = delete;

  explicit SceneManager(DxvkDevice* device);
  ~SceneManager();

  void initialize(Rc<DxvkContext> ctx);
  void logStatistics();

  void onDestroy();

  void submitDrawState(Rc<DxvkContext> ctx, const DrawCallState& input, const MaterialData* overrideMaterialData);
  void submitExternalDraw(const Rc<DxvkContext>& ctx, std::unique_ptr<ExternalDrawState> state);

  // Submits replacement geometry that places itself, once per frame.
  //
  // Must run after the frame's game draws and before prepareSceneData: the gate asks which meshes were
  // drawn this frame, so those draws have to have been seen, and the instances this creates must exist
  // before the TLAS is built from them.
  void submitWorldAnchoredInstancers(Rc<DxvkContext> ctx);

  void setStartInMediumMaterial(const MaterialData& translucentMaterial);
  void clearStartInMediumMaterial();

  // Remove an externally created mesh and all associated replacement instances.
  // Note: this is only safe to call from the dxvk-cs thread.
  void destroyExternalMesh(remixapi_MeshHandle handle);
  
  void setExternalStartInMediumMaterial(const MaterialData& translucentMaterial);
  void clearExternalStartInMediumMaterial();
  
  bool areAllReplacementsLoaded() const;
  std::vector<Mod::State> getReplacementStates() const;

  RtxGlobals& getGlobals() { return m_globals; }

  Rc<DxvkBuffer> getSurfaceMaterialBuffer() { return m_surfaceMaterialBuffer; }
  Rc<DxvkBuffer> getSurfaceMaterialExtensionBuffer() { return m_surfaceMaterialExtensionBuffer; }
  Rc<DxvkBuffer> getVolumeMaterialBuffer() { return m_volumeMaterialBuffer; }
  Rc<DxvkBuffer> getSurfaceBuffer() const { return m_accelManager.getSurfaceBuffer(); }
  Rc<DxvkBuffer> getSurfaceMappingBuffer() const { return m_accelManager.getSurfaceMappingBuffer(); }
  Rc<DxvkBuffer> getCurrentFramePrimitiveIDPrefixSumBuffer() const { return m_accelManager.getCurrentFramePrimitiveIDPrefixSumBuffer(); }
  Rc<DxvkBuffer> getLastFramePrimitiveIDPrefixSumBuffer() const { return m_accelManager.getLastFramePrimitiveIDPrefixSumBuffer(); }
  Rc<DxvkBuffer> getBillboardsBuffer() const { return m_accelManager.getBillboardsBuffer(); }
  bool isPreviousFrameSceneAvailable() const { return m_previousFrameSceneAvailable && getSurfaceMappingBuffer().ptr() != nullptr; }

  const std::vector<Rc<DxvkSampler>>& getSamplerTable() const { return m_samplerCache.getObjectTable(); }
  const std::vector<RaytraceBuffer>& getBufferTable() const { return m_bufferCache.getObjectTable(); }
  const std::vector<RtInstance*>& getInstanceTable() const { return m_instanceManager.getInstanceTable(); }
  
  const InstanceManager& getInstanceManager() const { return m_instanceManager; }
  const AccelManager& getAccelManager() const { return m_accelManager; }
  const LightManager& getLightManager() const { return m_lightManager; }
  const GraphManager& getGraphManager() const { return m_graphManager; }
  const RayPortalManager& getRayPortalManager() const { return m_rayPortalManager; }
  const BindlessResourceManager& getBindlessResourceManager() const { return m_bindlessResourceManager; }
  OpacityMicromapManager* getOpacityMicromapManager() const { return m_opacityMicromapManager.get(); }
  LightManager& getLightManager() { return m_lightManager; }
  GraphManager& getGraphManager() { return m_graphManager; }
  std::unique_ptr<AssetReplacer>& getAssetReplacer() { return m_pReplacer; }
  TerrainBaker& getTerrainBaker() { return *m_terrainBaker.get(); }

  // Scene utility functions
  static Vector3 getSceneUp();
  static Vector3 getSceneForward();
  static Vector3 calculateSceneRight();

  // Reswizzles input vector to an output that has xy coordinates on scene's horizontal axes and z coordinate to be on the scene's vertical axis
  static Vector3 worldToSceneOrientedVector(const Vector3& worldVector); 

  static Vector3 sceneToWorldOrientedVector(const Vector3& sceneVector);

  void addLight(const D3DLIGHT9& light);

  // Applies a light replacement to an externally submitted light, if one is authored for it.
  //
  // The replacement path was reachable only from addLight(const D3DLIGHT9&), so a light created
  // through remixapi_CreateLight never met getReplacementsForLight at all. For an API host that meant
  // a pack's light_ keys, and any light edited in the toolkit, were visible in a capture and inert in
  // game -- not because the identity was wrong but because nothing ever looked.
  //
  // Returns true when a replacement was instantiated, in which case the caller must not draw the
  // original: its replacement stands in for it, and drawing both shows the vanilla light alongside
  // the lights meant to supersede it.
  bool applyExternalLightReplacement(RtLight& original);

  const CameraManager& getCameraManager() const { return m_cameraManager; }
  CameraManager& getCameraManager() { return m_cameraManager; }
  const RtCamera& getCamera() const { return m_cameraManager.getMainCamera(); }
  RtCamera& getCamera() { return m_cameraManager.getMainCamera(); }

  const FogState& getFogState() const { return m_fog; }
  FogState& getFogState() { return m_fog; }
  const fast_unordered_cache<FogState>& getFogStates() const { return m_fogStates; }

  uint32_t getStartInMediumMaterialIndex() { return m_startInMediumMaterialIndex; }
  
  uint32_t getActivePOMCount() {return m_activePOMCount;}

  float getTotalMipBias();
  float getCalculatedUpscalingMipBias();

  // ISceneManager but not really
  void clear(Rc<DxvkContext> ctx, bool needWfi);
  void garbageCollection();
  void prepareSceneData(Rc<RtxContext> ctx, class DxvkBarrierSet& execBarriers);

  void onFrameEnd(Rc<DxvkContext> ctx, bool raytracedThisFrame);

  // GameCapturer
  void triggerUsdCapture() const;
  bool isGameCapturerIdle() const;

  using SamplerIndex = uint32_t;

  void trackTexture(const TextureRef& inputTexture,
                    uint32_t& textureIndex,
                    bool hasTexcoords,
                    bool async = true,
                    uint16_t* inout_samplerFeedbackStamp = nullptr);
  [[nodiscard]] SamplerIndex trackSampler(Rc<DxvkSampler> sampler);

  std::optional<XXH64_hash_t> findLegacyTextureHashByObjectPickingValue(uint32_t objectPickingValue);
  std::vector<ObjectPickingValue> gatherObjectPickingValuesByTextureHash(XXH64_hash_t texHash);

  // Replacement material hash tracking
  void trackReplacementMaterialHash(XXH64_hash_t materialHash);
  bool isReplacementMaterialHashUsedThisFrame(XXH64_hash_t materialHash) const;
  uint32_t getReplacementMaterialHashUsageCount(XXH64_hash_t materialHash) const;
  void clearFrameReplacementMaterialHashes();

  // Mesh hash tracking
  void trackMeshHash(XXH64_hash_t meshHash);
  bool isMeshHashUsedThisFrame(XXH64_hash_t meshHash) const;

  // Records the exterior cell a terrain draw sits in, and tests whether a cell is one the frame's
  // terrain covers. Together these scope world-anchored scatter to the space being rendered.
  void trackTerrainCell(const DrawCallState& input);
  bool isTerrainCellActiveThisFrame(int32_t cellX, int32_t cellY) const;
  uint32_t getMeshHashUsageCount(XXH64_hash_t meshHash) const;
  void clearFrameMeshHashes();

  Rc<DxvkSampler> patchSampler( const VkFilter filterMode,
                                const VkSamplerAddressMode addressModeU,
                                const VkSamplerAddressMode addressModeV,
                                const VkSamplerAddressMode addressModeW,
                                const VkClearColorValue borderColor);

  void requestTextureVramFree();
  void requestVramCompaction();
  void manageTextureVram();

  bool isThinOpaqueMaterialExist() const { return m_thinOpaqueCount > 0; }
  bool isSssMaterialExist() const { return m_sssCount > 0; }

  bool isAntiCullingSupported() const { return m_isAntiCullingSupported; }

private:
  enum class ObjectCacheState
  {
    kUpdateInstance = 0,
    kUpdateBVH = 1,
    KBuildBVH = 2,
    kInvalid = -1
  };
  // Handles conversion of geometry data coming from a draw call, to the data used by the raytracing backend
  template<bool isNew>
  ObjectCacheState processGeometryInfo(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas);

  // Consumes a draw call state and updates the scene state accordingly
  RtInstance* processDrawCallState(const Rc<DxvkContext>& ctx, 
                                   const DrawCallState& blasInput, 
                                   const MaterialData& materialData,
                                   ReplacementInstance& replacementInstance,
                                   RtInstance* existingInstance = nullptr,
                                   const RtxParticleSystemDesc* pParticleSystemDesc = nullptr);

  const RtSurfaceMaterial& createSurfaceMaterial(const MaterialData& renderMaterialData,
                                                 const DrawCallState& drawCallState,
                                                 uint32_t* out_indexInCache = nullptr);

  // Retain / release all resources associated with the surface material at the given cache index:
  // texture ref counts (via RtxTextureManager) and scene-wide feature counts (POM, SSS, thin-opaque).
  // Called when an instance's bound material changes or the instance is destroyed.
  void retainSurfaceMaterial(uint32_t matIdx);
  void releaseSurfaceMaterial(uint32_t matIdx);

  RtTranslucentSurfaceMaterial createTranslucentSurfaceMaterial(const TranslucentMaterialData& translucentMaterialData,
                                                                uint32_t samplerIndex,
                                                                bool hasTexcoords);
  Rc<DxvkSampler> getOrCreateExternalSampler();

  // Acquire/update stable buffer-cache slots for all geometry buffers in geo.
  // Called on the dynamic path (KBuildBVH / kUpdateBVH / kUpdateInstance) only.
  void updateBufferCache(BlasEntry* pBlas);

  // Propagate the BLAS's current geometry buffer indices/strides/formats to all
  // currently-linked instances.
  void syncLinkedInstances(BlasEntry* pBlas);

  // Release all buffer-cache slots held by geo. Called when a BlasEntry is GC'd.
  void unregisterGeometryBuffers(RaytraceGeometry& geo);

  // Release every BlasEntry's buffer slots and assert the active count reaches zero.
  // Called from clear() to detect registration/release imbalances before wiping the table.
  void verifyAndReleaseBufferCache();

  // Called whenever a new BLAS scene object is added to the cache
  ObjectCacheState onSceneObjectAdded(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas);
  // Called whenever a BLAS scene object is updated
  ObjectCacheState onSceneObjectUpdated(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas);
  // Called whenever a new instance has been added to the database
  void onInstanceAdded(RtInstance& instance);
  // Called whenever instance metadata is updated
  void onInstanceUpdated(RtInstance& instance, const DrawCallState& drawCall, const MaterialData* material, const bool hasTransformChanged, const bool hasVerticesdChanged, const bool isFirstUpdateThisFrame);
  // Called whenever an instance has been removed from the database
  void onInstanceDestroyed(RtInstance& instance);

  void drawReplacements(Rc<DxvkContext> ctx, const DrawCallState* input, const std::vector<AssetReplacement>* pReplacements, MaterialData& renderMaterialData, ReplacementInstance* replacementInstance);

  // Build the per-replacement DrawCallState used by both the dynamic (drawReplacements) and
  // preserve (syncPreservedReplacementMeshesState) paths so they always feed the same input
  // into the BlasEntry / processDrawCallState. Returns std::nullopt for replacement types
  // that don't drive a mesh draw (lights, graphs).
  // Static member so the friendship of SceneManager with DrawCallState carries through.
  static std::optional<DrawCallState> buildReplacementMeshDrawCallState(
      const DrawCallState& input,
      const AssetReplacement& replacement);

  // Minimal per-frame work for the preserve path (unchanged / anti-culled draws).
  // When pInput is set (preserve replacement draw path), also runs the ray-portal refresh
  // via processRayPortalData on the cached RtSurfaceMaterial.
  void preserveInstance(
      RtInstance& instance,
      const DrawCallState* pInput = nullptr);

  // Lights and graph prims for mesh replacements (drawReplacements / dynamic path).
  void processReplacementLights(const DrawCallState* input, const std::vector<AssetReplacement>* pReplacements, ReplacementInstance* replacementInstance);
  void processReplacementGraphs(Rc<DxvkContext> ctx, const DrawCallState* input, const std::vector<AssetReplacement>* pReplacements, ReplacementInstance* replacementInstance);

  // Preserve path: minimal work to keep replacement meshes, lights, and graphs alive (buffer cache, textures, light touch, bbox).
  void preserveReplacementInstance(
      Rc<DxvkContext> ctx,
      const DrawCallState& input,
      const std::vector<AssetReplacement>* pReplacements,
      ReplacementInstance* replacementInstance);

  void trackObjectPickingMeta(const DrawCallState& drawCallState, ObjectPickingValue objectPickingValue);

  // Refreshes BlasEntry::input and per-instance draw state on the preserve path (matches drawReplacements'
  // DrawCallState wiring).
  void syncPreservedReplacementMeshesState(
      const DrawCallState& input,
      const std::vector<AssetReplacement>* pReplacements,
      ReplacementInstance* replacementInstance);

  void createEffectLight(Rc<DxvkContext> ctx, const DrawCallState& input, const RtInstance* instance);

  // Print all RtInstances for debugging
  void printAllRtInstances();
  
  MaterialData determineMaterialData(const MaterialData* overrideMaterialData, const DrawCallState& input);
  
  const uint32_t kInvalidMaterialCacheIndex = UINT32_MAX;
  uint32_t m_beginUsdExportFrameNum = -1;
  bool m_enqueueDelayedClear = false;
  bool m_previousFrameSceneAvailable = false;

  RtxGlobals m_globals;

  // Hash/Cache's
  InstanceManager m_instanceManager;
  AccelManager m_accelManager;
  LightManager m_lightManager;
  GraphManager m_graphManager;
  RayPortalManager m_rayPortalManager;
  BindlessResourceManager m_bindlessResourceManager;
  std::unique_ptr<OpacityMicromapManager> m_opacityMicromapManager;

  DrawCallCache m_drawCallCache;

  CameraManager m_cameraManager;

  std::unique_ptr<AssetReplacer> m_pReplacer;

  std::unique_ptr<TerrainBaker> m_terrainBaker;

  FogState m_fog;
  fast_unordered_cache<FogState> m_fogStates;
  std::mutex m_startInMediumMaterialMutex;
  std::optional<MaterialData> m_pendingStartInMediumMaterial;
  bool m_pendingClearStartInMediumMaterial = false;
  std::optional<MaterialData> m_persistentStartInMediumMaterial;
  uint32_t m_startInMediumMaterialIndex = SURFACE_INDEX_INVALID;
  uint32_t m_fogStartInMediumMaterialIndex_inCache = kInvalidMaterialCacheIndex;
  uint32_t m_externalStartInMediumMaterialIndex_inCache = kInvalidMaterialCacheIndex;
  uint32_t m_startInMediumMaterialIndex_inCache = kInvalidMaterialCacheIndex;
  uint32_t m_lastResolvedStartInMediumMaterialIndexInCache = kInvalidMaterialCacheIndex;
  uint32_t m_lastUploadedStartInMediumMaterialIndexInCache = kInvalidMaterialCacheIndex;

  // TODO: Move the following resources and getters to RtResources class
  Rc<DxvkBuffer> m_surfaceMaterialBuffer;
  Rc<DxvkBuffer> m_surfaceMaterialExtensionBuffer;
  Rc<DxvkBuffer> m_volumeMaterialBuffer;

  uint32_t m_activePOMCount = 0;
  
  float m_uniqueObjectSearchDistance = 1.f;

  struct DrawCallMetaInfo {
    XXH64_hash_t legacyTextureHash { kEmptyHash };
    XXH64_hash_t legacyTextureHash2 { kEmptyHash };
  };
  struct DrawCallMeta {
    constexpr static inline uint8_t MaxTicks = 2;
    std::unordered_map<ObjectPickingValue, DrawCallMetaInfo> infos[MaxTicks] {};
    bool ready[MaxTicks] {};
    uint8_t ticker {};
    dxvk::mutex mutex {};
  } m_drawCallMeta {};

  // TODO: expand to many different
  Rc<DxvkSampler> m_externalSampler = nullptr;

  std::atomic_bool m_forceFreeTextureMemory = false;
  std::atomic_bool m_forceFreeUnusedDxvkAllocatorChunks = false;

  uint32_t m_thinOpaqueCount = 0;
  uint32_t m_sssCount = 0;

  bool m_isAntiCullingSupported = true;

  // Matches RtxTextureManager::m_textureCacheGeneration after a completed frame's texture
  // registrations. Updated once per frame in onFrameEnd (before manageTextureVram). Left
  // stale when manageTextureVram clears the cache so the next frame is all-dynamic.
  uint32_t m_textureCacheGenerationValidForPreserve = 0;

  // Paces first-time replacement builds for API-submitted meshes, so that entering a cell does not put
  // every one of its replacement acceleration structure builds in a single frame. Holds the distinct mesh
  // hashes admitted in the current frame, since the cost is per geometry rather than per instance. See
  // the use site in submitExternalDraw; ~0u forces a reset on the first frame it is consulted.
  uint32_t m_externalReplacementBuildFrame = ~0u;
  fast_unordered_set m_externalReplacementBuildMeshes;
  uint64_t m_externalReplacementDeferrals = 0;
  uint64_t m_externalReplacementDeferralsLogged = 0;

  // The same pacing for world-anchored groups. A plain counter rather than a set of admitted hashes,
  // because each group is visited at most once per frame and builds its own geometry, so there is no
  // second instance of a group to let through. See the use site in submitWorldAnchoredInstancers.
  uint32_t m_worldAnchoredBuildFrame = ~0u;
  uint32_t m_worldAnchoredBuildsThisFrame = 0;

  // Replacement material hash tracking for current frame (hash -> count)
  std::unordered_map<XXH64_hash_t, uint32_t> m_currentFrameReplacementMaterialHashes;

  // Mesh hash tracking for current frame (hash -> count)
  std::unordered_map<XXH64_hash_t, uint32_t> m_currentFrameMeshHashes;
  // Exterior cells covered by terrain drawn this frame, packed x:y. Empty indoors, which is what makes
  // exterior scatter unable to leak into interiors.
  std::unordered_set<uint64_t> m_currentFrameTerrainCells;

  DrawCallTracker m_drawCallTracker;

  // Per replaced light, the brightest radiance it has been seen at, and when it was last seen.
  //
  // This is what lets a toolkit-added light inherit the flicker of the game light it was created from. The
  // host varies a flickering light's radiance every frame, so the ratio of its current radiance to its own
  // peak is the flicker curve itself -- no authoring, no new USD attribute, and no reimplementation of
  // Morrowind's flicker maths, which could only drift from the original.
  //
  // Keyed on the light's *instance* identity (asset hash mixed with position), which gives the grouping for
  // free: four lights placed above one candle all replace the same original and therefore share one phase,
  // while two candles across a room flicker independently because the game gives each vanilla light its own
  // phase. Four independently guttering lights over a single candle would read as four separate flames.
  //
  // A running peak rather than a first sample, because the first frame a light is seen may catch it
  // mid-dip. Morrowind re-rolls its flicker target into 0.25..1.0 and walks continuously, so the peak
  // converges within a second or two, and the ratio is clamped to 1 until it does. Frame-stamped so the map
  // does not grow without bound across a session's cell changes.
  struct LightFlickerState {
    float peakRadiance = 0.0f;
    uint32_t frameLastSeen = 0;
  };
  std::unordered_map<XXH64_hash_t, LightFlickerState> m_lightFlicker;
  // Only swept once the map is big enough to be worth walking, and only entries older than a few seconds
  // are dropped -- a light that comes back into view sooner than that keeps its phase rather than snapping.
  static constexpr size_t kLightFlickerPruneThreshold = 512;
  static constexpr uint32_t kLightFlickerPruneFrames = 600;
};

}  // namespace nvvk
