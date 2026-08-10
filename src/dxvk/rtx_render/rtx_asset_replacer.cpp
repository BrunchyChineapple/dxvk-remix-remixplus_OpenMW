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

#include "rtx_asset_replacer.h"

#include "dxvk_device.h"
#include "dxvk_context.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_utils.h"
#include "rtx_asset_data_manager.h"

namespace dxvk {

std::vector<AssetReplacement>* AssetReplacer::getReplacementsForMesh(XXH64_hash_t hash) {
  if (!RtxOptions::getEnableReplacementMeshes())
    return nullptr;

  auto variantInfo = m_variantInfos.find(hash);

  if (variantInfo != m_variantInfos.end()) {
    hash += variantInfo->second.selectedVariant;
  }

  for (auto& mod : m_modManager.mods()) {
    if (auto replacement = mod->replacements().get<AssetReplacement::eMesh>(hash)) {
      return replacement;
    }
  }

  return nullptr;
}

std::vector<AssetReplacement>* AssetReplacer::getReplacementsForLight(XXH64_hash_t hash) {
  if (!RtxOptions::getEnableReplacementLights())
    return nullptr;

  for (auto& mod : m_modManager.mods()) {
    if (auto replacement = mod->replacements().get<AssetReplacement::eLight>(hash)) {
      return replacement;
    }
  }

  return nullptr;
}

std::vector<const WorldAnchoredInstancerGroup*> AssetReplacer::getWorldAnchoredInstancers() {
  std::vector<const WorldAnchoredInstancerGroup*> result;

  // Gated by the same option as keyed mesh replacements: these are replacement meshes too, just ones
  // that place themselves. Turning replacement meshes off and still drawing these would be surprising.
  if (!RtxOptions::getEnableReplacementMeshes()) {
    return result;
  }

  for (auto& mod : m_modManager.mods()) {
    for (const WorldAnchoredInstancerGroup& group : mod->replacements().worldAnchoredInstancers()) {
      result.push_back(&group);
    }
  }

  return result;
}

MaterialData* AssetReplacer::getReplacementMaterial(XXH64_hash_t hash) {
  if (!RtxOptions::getEnableReplacementMaterials())
    return nullptr;

  for (auto& mod : m_modManager.mods()) {
    MaterialData* material;
    if (mod->replacements().getObject(hash, material)) {
      return material;
    }
  }

  return nullptr;
}

void AssetReplacer::initialize(const Rc<DxvkContext>& context) {
  for (auto& mod : m_modManager.mods()) {
    mod->load(context);
  }
  updateSecretReplacements();
}

bool AssetReplacer::checkForChanges(const Rc<DxvkContext>& context) {
  ScopedCpuProfileZone();

  bool changed = false;
  for (auto& mod : m_modManager.mods()) {
    changed |= mod->checkForChanges(context);
  }
  if (changed) {
    updateSecretReplacements();
  }
  return changed;
}

bool AssetReplacer::areAllReplacementsLoaded() const {
  for (auto& mod : m_modManager.mods()) {
    if (mod->state().progressState != Mod::ProgressState::Loaded) {
      return false;
    }
  }

  return true;
}

std::vector<Mod::State> AssetReplacer::getReplacementStates() const {
  const auto& mods = m_modManager.mods();
  std::vector<Mod::State> modStates(mods.size());

  for (auto& mod : mods) {
    modStates.emplace_back(mod->state());
  }

  return modStates;
}

void AssetReplacer::updateSecretReplacements() {
  bool updated = false;

  m_variantInfos.clear();
  m_secretReplacements.clear();

  for (auto& mod : m_modManager.mods()) {
    if (mod->state().progressState != Mod::ProgressState::Loaded) {
      continue;
    }

    // Pull secret replacement info
    for (auto& secrets : mod->replacements().secretReplacements()) {
      for (auto& secret : secrets.second) {
        m_secretReplacements[secrets.first].emplace_back(
          SecretReplacement{secret.header,
                            secret.name,
                            secret.description,
                            secret.unlockHash,
                            secret.assetHash,
                            secret.replacementPath,
                            secret.bDisplayBeforeUnlocked,
                            secret.bExclusiveReplacement,
                            secret.variantId
          }
        );

        auto& numVariants = m_variantInfos[secrets.first].numVariants;
        numVariants = std::max(secret.variantId, numVariants);

        updated = true;
      }
    }
  }

  m_bSecretReplacementsUpdated = updated;
}

namespace {
  std::string tostr(const remixapi_MaterialHandle& h) {
    static_assert(sizeof h == sizeof uint64_t);
    return std::to_string(reinterpret_cast<uint64_t>(h));
  }
  std::string tostr(const remixapi_MeshHandle& h) {
    static_assert(sizeof h == sizeof uint64_t);
    return std::to_string(reinterpret_cast<uint64_t>(h));
  }
}

void AssetReplacer::makeMaterialWithTexturePreload(DxvkContext& ctx, remixapi_MaterialHandle handle, MaterialData&& data) {
  auto [iter, isNew] = m_extMaterials.emplace(handle, std::move(data));

  if (!isNew) {
    Logger::info("Ignoring repeated material registration (handle=" + tostr(handle) + ") ");
    return;
  }
}

const MaterialData* AssetReplacer::accessExternalMaterial(remixapi_MaterialHandle handle) const {
  auto found = m_extMaterials.find(handle);
  if (found == m_extMaterials.end()) {
    return nullptr;
  }
  return found->second ? &found->second.value() : nullptr;
}

void AssetReplacer::destroyExternalMaterial(remixapi_MaterialHandle handle) {
  m_extMaterials.erase(handle);
}

void AssetReplacer::registerExternalMesh(remixapi_MeshHandle handle, std::vector<RasterGeometry>&& submeshes) {
  if (m_extMeshes.count(handle) > 0) {
    //Logger::info("Ignoring repeated mesh registration (handle=" + tostr(handle) + ") ");
    return;
  }

  // Tag each submesh with the external mesh handle so capture + runtime
  // use the same identity for replacement-lookup parity.
  for (auto& submesh : submeshes) {
    submesh.externalMesh = handle;
  }

  m_extMeshes.emplace(handle, std::make_unique< std::vector<RasterGeometry>>(std::move(submeshes)));
}

const std::vector<RasterGeometry>& AssetReplacer::accessExternalMesh(remixapi_MeshHandle handle) const {
  auto found = m_extMeshes.find(handle);
  if (found == m_extMeshes.end()) {
    static const auto s_empty = std::vector<RasterGeometry> {};
    return s_empty;
  }
  return *found->second;
}

void AssetReplacer::destroyExternalMesh(remixapi_MeshHandle handle, uint32_t currentFrameId) {
  auto found = m_extMeshes.find(handle);
  if (found == m_extMeshes.end()) {
    return;
  }

  // The handle stops resolving now, but the geometry it owns is not released yet.
  //
  // Erasing outright frees the vertex and index buffers on the spot, and the GPU may still be reading them:
  // ray tracing hit shaders fetch attributes from a geometry's index buffer, and enablePreviousTLAS means the
  // previous frame's TLAS can still hold an instance whose BLAS was built from this geometry. Two crash dumps
  // faulted in precisely that memory, one 32% inside a terrain chunk's modified index buffer during Primary
  // Rays and one just past another during Volume Integrate Raytracing. Both happened while teleporting, which
  // destroys a great many meshes in one frame and so hits the window far more often than normal play.
  m_retiredExtMeshes.emplace_back(currentFrameId, std::move(found->second));
  m_extMeshes.erase(found);
}

void AssetReplacer::releaseRetiredExternalMeshes(uint32_t currentFrameId) {
  // Held for the geometry-data window plus the frames that may still be in flight.
  //
  // The window alone is not enough. It governs when a BLAS may be freed, and this geometry has to outlive the
  // BLAS built from it, not merely match it; and the destroy runs on the CS thread, where earlier submissions
  // can still be executing. The margin covers that.
  constexpr uint32_t kInFlightMargin = 3;
  const uint32_t hold = RtxOptions::numFramesToKeepGeometryData() + kInFlightMargin;
  if (currentFrameId <= hold) {
    return;
  }

  const uint32_t oldest = currentFrameId - hold;
  for (auto iter = m_retiredExtMeshes.begin(); iter != m_retiredExtMeshes.end(); ) {
    if (iter->first < oldest) {
      iter = m_retiredExtMeshes.erase(iter);
    } else {
      ++iter;
    }
  }
}

} // namespace dxvk

