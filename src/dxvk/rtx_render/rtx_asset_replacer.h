/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#include <memory>

#include "rtx_types.h"
#include "graph/rtx_graph_types.h"
#include "rtx_lights.h"
#include "rtx_mod_manager.h"
#include "rtx_utils.h"
#include "rtx_lights_data.h"

namespace dxvk {
  class DxvkContext;
  class DxvkDevice;
  class DxvkCommandList;

  struct Categorizer {
    CategoryFlags categoryFlags;
    CategoryFlags categoryExists;

    CategoryFlags applyCategoryFlags(const CategoryFlags& input) const {
      return (input.raw() & (~categoryExists.raw())) | categoryFlags.raw();
    }
  };

  struct MeshReplacement {
    RasterGeometry data;
  };

  struct ParticleEmitter {
    RasterGeometry data;
    RtxParticleSystemDesc desc;
  };

  constexpr uint32_t kInvalidPointInstanceIndex = std::numeric_limits<uint32_t>::max();

  struct AssetReplacement {
    enum Type {
      eMesh,
      eLight,
      eGraph,
      eNone,
    };
    Categorizer categories;
    MeshReplacement* geometry = nullptr;
    std::optional<RtxParticleSystemDesc> particleSystem;
    std::optional<LightData> lightData;
    std::optional<RtGraphState> graphState;
    // Note: This is the material to use for this replacement, if any. Set to null if should use
    // the original material instead, similar to how getReplacementMaterial works.
    MaterialData* materialData = nullptr;
    Matrix4 replacementToObject;
    // If this replacement represents multiple instances of an object, then this will contain a
    // list of transforms from the instance's space to Object space
    // (use the drawcall's objectToWorld * instancesToObject[n] to get instance n's world transform).
    // Stored as shared_ptr so downstream consumers (DrawCallTransforms, RtSurface,
    // PointInstancerBatch) can safely reference the data across frame boundaries.
    std::shared_ptr<std::vector<Matrix4>> instancesToObject;
    // If this replacement represents a single instance from a pointInstancer, then this will 
    // contain the index of the instance in the pointInstancer.
    uint32_t pointInstanceIndex = kInvalidPointInstanceIndex; 
    Type type;
    bool includeOriginal = false;

    std::string primPath;
    XXH64_hash_t usdPathHash = kEmptyHash;

    AssetReplacement(const std::string& primPath) :
      type(eNone),
      primPath(primPath),
      usdPathHash(XXH3_64bits(primPath.c_str(), primPath.size()))
    {}
    AssetReplacement(
        const std::string& primPath, 
        MeshReplacement* geometryData,
        MaterialData* materialData,
        Categorizer categoryFlags,
        const Matrix4& replacementToObject) :
      geometry(geometryData),
      materialData(materialData),
      categories(categoryFlags),
      replacementToObject(replacementToObject),
      type(eMesh),
      primPath(primPath),
      usdPathHash(XXH3_64bits(primPath.c_str(), primPath.size()))
    {}
    AssetReplacement(
        const std::string& primPath,
        const LightData& lightData,
        const Matrix4& replacementToObject) :
      lightData(lightData),
      replacementToObject(replacementToObject),
      type(eLight),
      primPath(primPath),
      usdPathHash(XXH3_64bits(primPath.c_str(), primPath.size()))
    {}
  };

  struct SecretReplacement {
    const std::string header;
    const std::string name;
    const std::string description;
    const XXH64_hash_t unlockHash;
    const XXH64_hash_t assetHash;
    const std::string replacementPath;
    const bool bDisplayBeforeUnlocked;
    // Instance tracking necessary to set this to false
    const bool bExclusiveReplacement = true; 
    const size_t variantId;
  };

  typedef fast_unordered_cache<std::vector<SecretReplacement>> SecretReplacements;

  // Replacement geometry placed by its own world transform rather than by a game draw.
  //
  // Every other replacement here is keyed on the hash of a draw the game makes and is only submitted
  // while that draw is being processed. That anchor is what gives it a position and what scopes it to
  // the right place in the world. Scatter painted onto captured terrain has no such anchor available:
  // the host builds its terrain itself, so the captured mesh hashes the brush painted against are
  // never drawn, and nothing anchored to them can be reached at all.
  //
  // So these carry their own world placement -- and, just as importantly, the identity of the space
  // they belong to. A world position on its own is ambiguous: interiors and the exterior share one
  // numeric coordinate space, and the scatter spans many cells, so drawing purely by position puts
  // exterior groundcover inside any interior that overlaps those coordinates and one town's
  // groundcover in the next town.
  //
  // Two independent keys, because they cover different cases and neither covers both:
  //   anchorMeshHash  the captured mesh this group was painted on. Anything painted on a static has a
  //                   hash the host does reproduce, since statics come from the same source data, so
  //                   "was that mesh drawn this frame" is an exact test needing no cell knowledge.
  //   cellX/cellY     the exterior cell the group occupies, for the terrain case where the anchor
  //                   hash can never appear. Only meaningful against a host-published set of active
  //                   cells; it is not a licence to infer the active space from coordinates.
  struct WorldAnchoredInstancerGroup {
    std::vector<AssetReplacement> replacements;
    // Stable identity for ReplacementInstance lookup, hashed from the group's USD path at load.
    //
    // Has to come from the path rather than from the transform or position, which is what the keyed
    // paths derive identity from: every one of these groups sits at the same identity transform, so a
    // position-derived key would collide across all of them and they would fight over one instance.
    XXH64_hash_t identityHash = 0;
    // 0 means the group arrived untagged. Such a group cannot be scoped, and drawing it
    // unconditionally is exactly the interior-bleed bug, so callers must skip rather than guess.
    XXH64_hash_t anchorMeshHash = 0;
    int32_t cellX = 0;
    int32_t cellY = 0;
    bool hasCell = false;
    // USD prim name, carried purely so a misplaced group can be identified in a log.
    std::string debugName;
  };

  // Asset replacements storage class.
  // Contains and owns the replacements, material and geometry objects.
  class AssetReplacements {
  public:
    // Returns a pointer to replacements of type T for a given hash value,
    // or a nullptr if no replacements found.
    template<AssetReplacement::Type T>
    std::vector<AssetReplacement>* get(XXH64_hash_t hash) {
      std::lock_guard<sync::Spinlock> lock(m_spinlock);
      auto& map = T == AssetReplacement::eMesh ? m_meshReplacers : m_lightReplacers;
      auto it = map.find(hash);
      if (it != map.end()) {
        return &it->second;
      }
      return nullptr;
    }

    // Stores replacements of type T for a hash value.
    template<AssetReplacement::Type T>
    void set(XXH64_hash_t hash, std::vector<AssetReplacement>&& v) {
      std::lock_guard<sync::Spinlock> lock(m_spinlock);
      auto& map = T == AssetReplacement::eMesh ? m_meshReplacers : m_lightReplacers;
      map.emplace(hash, std::move(v));
    }

    // Returns a pointer to the stored object of type T for a given hash value.
    // Return false if no object was found.
    template<typename T>
    bool getObject(XXH64_hash_t hash, T*& obj) {
      std::lock_guard<sync::Spinlock> lock(m_spinlock);
      fast_unordered_cache<T>* cache = nullptr;
      if constexpr (std::is_same_v<T, MaterialData>) {
        cache = &m_materials;
      } else if constexpr (std::is_same_v<T, MeshReplacement>) {
        cache = &m_geometries;
      } else if constexpr (std::is_same_v<T, RtGraphTopology>) {
        cache = &m_graphTopologies;
      }
      if (cache == nullptr) {
        return false;
      }
      auto it = cache->find(hash);
      if (it != cache->end()) {
        obj = &it->second;
        return true;
      }
      return false;
    }

    // Stores the object of type T for a hash value.
    template<typename T>
    T& storeObject(XXH64_hash_t hash, T&& obj) {
      std::lock_guard<sync::Spinlock> lock(m_spinlock);
      if constexpr (std::is_same_v<T, MaterialData>) {
        return m_materials.try_emplace(hash, std::move(obj)).first->second;
      } else if constexpr (std::is_same_v<T, MeshReplacement>) {
        return m_geometries.try_emplace(hash, std::move(obj)).first->second;
      } else if constexpr (std::is_same_v<T, RtGraphTopology>) {
        return m_graphTopologies.try_emplace(hash, std::move(obj)).first->second;
      } else if constexpr (std::is_same_v<T, SecretReplacement>) {
        return m_secretReplacements[hash].emplace_back(obj);
      } else {
        static_assert(false, "Invalid type");
      }
    }

    // Removes the object of type T for a hash value.
    template<typename T>
    void removeObject(XXH64_hash_t hash) {
      std::lock_guard<sync::Spinlock> lock(m_spinlock);
      if constexpr (std::is_same_v<T, MaterialData>) {
        m_materials.erase(hash);
      } else if constexpr (std::is_same_v<T, MeshReplacement>) {
        m_geometries.erase(hash);
      } else if constexpr (std::is_same_v<T, RtGraphTopology>) {
        m_graphTopologies.erase(hash);
      } else if constexpr (std::is_same_v<T, SecretReplacement>) {
        m_secretReplacements.erase(hash);
      } else {
        static_assert(false, "Invalid type");
      }
    }

    // Destroys all replacements and stored objects.
    void clear() {
      std::lock_guard<sync::Spinlock> lock(m_spinlock);
      m_meshReplacers.clear();
      m_lightReplacers.clear();
      m_worldAnchoredInstancers.clear();
      m_materials.clear();
      m_geometries.clear();
      m_graphTopologies.clear();
      m_secretReplacements.clear();
    }

    const SecretReplacements& secretReplacements() const {
      return m_secretReplacements;
    }

    // Append-only, unlike the keyed replacements above: nothing looks these up by hash, they are
    // iterated wholesale each frame and gated individually.
    void addWorldAnchoredInstancer(WorldAnchoredInstancerGroup&& group) {
      std::lock_guard<sync::Spinlock> lock(m_spinlock);
      m_worldAnchoredInstancers.emplace_back(std::move(group));
    }

    const std::vector<WorldAnchoredInstancerGroup>& worldAnchoredInstancers() const {
      return m_worldAnchoredInstancers;
    }

  private:
    mutable sync::Spinlock m_spinlock;

    // Replacements ready to be fed to the renderer
    fast_unordered_cache<std::vector<AssetReplacement>> m_meshReplacers;
    fast_unordered_cache<std::vector<AssetReplacement>> m_lightReplacers;

    // Replacements that place themselves, rather than being placed by a game draw
    std::vector<WorldAnchoredInstancerGroup> m_worldAnchoredInstancers;

    // Replacement geometry storage
    fast_unordered_cache<MeshReplacement> m_geometries;

    // Replacement material storage
    fast_unordered_cache<MaterialData> m_materials;

    // Replacement graph storage
    fast_unordered_cache<RtGraphTopology> m_graphTopologies;

    // Secret replacements if any
    SecretReplacements m_secretReplacements;
  };

  struct AssetReplacer {
    std::vector<AssetReplacement>* getReplacementsForMesh(XXH64_hash_t hash);
    std::vector<AssetReplacement>* getReplacementsForLight(XXH64_hash_t hash);
    MaterialData* getReplacementMaterial(XXH64_hash_t hash);

    // Every world-anchored group across all loaded mods, in load order.
    //
    // Pointers rather than copies because the groups own their AssetReplacement vectors and this is
    // walked per frame. They remain valid until replacements are reloaded, which is the same lifetime
    // the keyed accessors above already hand out.
    std::vector<const WorldAnchoredInstancerGroup*> getWorldAnchoredInstancers();

    // process the replacement USD and create all the m_replacements entries.
    void initialize(const Rc<DxvkContext>& context);

    // returns true if the state of replacements has changed.
    bool checkForChanges(const Rc<DxvkContext>& context);


    // Returns true if all replacement mods are in the loaded state, false otherwise.
    bool areAllReplacementsLoaded() const;
    // Gets the states of all the current replacement mods.
    std::vector<Mod::State> getReplacementStates() const;

    const bool hasNewSecretReplacementInfo() const {
      return m_bSecretReplacementsUpdated;
    }

    const SecretReplacements& getSecretReplacementInfo() {
      assert(m_bSecretReplacementsUpdated);
      m_bSecretReplacementsUpdated = false;
      return m_secretReplacements;
    }

    void markVariantStatus(const XXH64_hash_t assetHash,
                           const size_t variantId,
                           const bool bEnabled) {
      m_variantInfos[assetHash].selectedVariant =
        (bEnabled) ? variantId : VariantInfo::kDefaultVariant;
    }

    void makeMaterialWithTexturePreload(DxvkContext& ctx, remixapi_MaterialHandle handle, MaterialData&& data);
    [[nodiscard]] const MaterialData* accessExternalMaterial(remixapi_MaterialHandle handle) const;
    void destroyExternalMaterial(remixapi_MaterialHandle handle);

    void registerExternalMesh(remixapi_MeshHandle handle, std::vector<RasterGeometry>&& submeshes);
    [[nodiscard]] const std::vector<RasterGeometry>& accessExternalMesh(remixapi_MeshHandle handle) const;
    void destroyExternalMesh(remixapi_MeshHandle handle);

  private:
    void updateSecretReplacements();

    bool m_bSecretReplacementsUpdated = false;

    struct VariantInfo {
      static constexpr size_t kDefaultVariant = 0;
      size_t numVariants = 0;
      size_t selectedVariant = kDefaultVariant;
    };

    fast_unordered_cache<VariantInfo> m_variantInfos;
    SecretReplacements m_secretReplacements;

    ModManager m_modManager;

    std::unordered_map<remixapi_MaterialHandle, std::optional<MaterialData>> m_extMaterials {};
    std::unordered_map<remixapi_MeshHandle, std::unique_ptr<std::vector<RasterGeometry>>> m_extMeshes {};
  };
} // namespace dxvk

