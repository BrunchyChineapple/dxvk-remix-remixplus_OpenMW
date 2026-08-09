// src/dxvk/rtx_render/rtx_fork_capture.cpp
//
// Fork-owned file. Contains the implementations of fork_hooks:: functions
// for the GameCapturer capture path, lifted from rtx_game_capturer.cpp
// during the 2026-04-18 fork touchpoint-pattern refactor.
//
// See docs/fork-touchpoints.md for the full fork-hooks catalogue.
//
// NOTE: captureMaterialApiPath accesses GameCapturer::m_exporter and
// GameCapturer::m_pCap, which are private members. This file requires that
// GameCapturer declare fork_hooks::captureMaterialApiPath as a friend —
// see rtx_game_capturer.h.

// Replicate the BASE_DIR macro so the texture export paths resolve the same
// way they do in rtx_game_capturer.cpp.
#include "../../util/util_filesys.h"
#define BASE_DIR (util::RtxFileSys::path(util::RtxFileSys::Captures).string())

#include "rtx_fork_hooks.h"

#include "rtx_game_capturer.h"       // GameCapturer (full definition for friend access)
#include "rtx_instance_manager.h"    // RtInstance
#include "rtx_materials.h"           // LegacyMaterialData, kSurfaceMaterialInvalidTextureIndex
#include "rtx_texture_manager.h"     // RtxTextureManager, TextureRef
#include "rtx_constants.h"           // kEmptyHash
#include "rtx_options.h"             // RtxOptions::leftHandedCoordinateSystem()

#include "../../lssusd/game_exporter_types.h"   // lss::Export, lss::Material, lss::ext, lss::commonDirName
#include "../../lssusd/game_exporter_paths.h"
#include "../../lssusd/usd_include_begin.h"
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3d.h>
#include "../../lssusd/usd_include_end.h"

#include "../../util/log/log.h"
#include "../../util/util_string.h"

#include <mutex>
#include <unordered_set>

namespace dxvk {
namespace fork_hooks {

  namespace {
    // Texture hashes already written for the capture in progress.
    //
    // The albedo used to be exported once per *material*, named after the material hash, and a host that
    // derives several materials from one texture therefore wrote the same image out several times over.
    // OpenMW does exactly that on purpose -- its material key mixes the texture hash with roughness,
    // metallic, alpha test, blend and normal map -- so a measured interior capture came to 942 texture files
    // holding 911 distinct images, 2 GB on disk of which 822 MB was byte-identical copies. One 2048-square
    // albedo was written seven times at 21 MB a copy.
    //
    // Naming the file after the texture instead collapses those to one, and this set stops the redundant
    // export work as well as the redundant bytes: without it, several materials would each schedule an
    // async write to the same path and race each other over one file.
    //
    // Keyed on the capture id so a second capture in the same session starts clean rather than inheriting
    // the first one's claims and skipping textures it never wrote.
    std::mutex g_exportedTextureMutex;
    std::string g_exportedTextureCaptureId;
    std::unordered_set<XXH64_hash_t> g_exportedTextures;

    /// True the first time \a textureHash is seen in the capture identified by \a captureId.
    bool claimTextureExport(const std::string& captureId, XXH64_hash_t textureHash) {
      std::lock_guard<std::mutex> lock(g_exportedTextureMutex);
      if (g_exportedTextureCaptureId != captureId) {
        g_exportedTextureCaptureId = captureId;
        g_exportedTextures.clear();
      }
      return g_exportedTextures.insert(textureHash).second;
    }

    /// The file an albedo is written to. Named after the texture so that every material sharing it resolves
    /// to one file, falling back to the material name only when there is no usable texture identity.
    std::string albedoFileName(XXH64_hash_t textureHash, const std::string& matName) {
      if (textureHash == 0 || textureHash == kEmptyHash) {
        return matName + lss::ext::dds;
      }
      return dxvk::hashToString(textureHash) + lss::ext::dds;
    }
  }

  // ---------------------------------------------------------------------------
  // captureMaterialApiPath
  //
  // Full implementation of GameCapturer::captureMaterial, handling both the
  // standard D3D9 path (color texture valid — export directly) and the
  // API-submitted path (no direct color texture — locate via texture-manager
  // table by hash and export). Caches the populated lss::Material in
  // GameCapturer::m_pCap->materials keyed by runtimeMaterialHash.
  //
  // ACCESS NOTE: this function reads GameCapturer::m_exporter and
  // GameCapturer::m_pCap (both private). A friend declaration for this
  // function is required in GameCapturer. See rtx_game_capturer.h.
  // ---------------------------------------------------------------------------
  void captureMaterialApiPath(
      GameCapturer& capturer,
      const Rc<DxvkContext> ctx,
      const RtInstance& rtInstance,
      XXH64_hash_t runtimeMaterialHash,
      const LegacyMaterialData& materialData,
      bool bEnableOpacity) {
    lss::Material lssMat; // to be populated

    // Use runtime material hash so replacements resolve to the same material at runtime.
    const std::string matName = dxvk::hashToString(runtimeMaterialHash);
    lssMat.matName = matName;

    const auto& colorTexture = materialData.getColorTexture();
    XXH64_hash_t textureHash = colorTexture.getImageHash();

    // For API-submitted materials the LegacyMaterialData has no direct color texture; look
    // up the albedo texture via the instance's surface-material texture index instead.
    if (textureHash == 0 || textureHash == kEmptyHash) {
      const uint32_t albedoTexIndex = rtInstance.getAlbedoOpacityTextureIndex();
      if (albedoTexIndex != kSurfaceMaterialInvalidTextureIndex) {
        const RtxTextureManager& textureManager = ctx->getCommonObjects()->getTextureManager();
        const auto& textureTable = textureManager.getTextureTable();
        if (albedoTexIndex < textureTable.size() && textureTable[albedoTexIndex].isValid()) {
          textureHash = textureTable[albedoTexIndex].getImageHash();
        }
      }
    }

    if (colorTexture.isValid() && colorTexture.getImageView()) {
      // Standard D3D9 material path: export the color texture directly.
      auto* imageView = colorTexture.getImageView();
      if (imageView && imageView->image().ptr()) {
        try {
          const std::string albedoTexFilename(albedoFileName(textureHash, matName));
          // The path is recorded whether or not this material is the one that writes the file, so every
          // material sharing the texture references the single copy.
          if (claimTextureExport(capturer.m_pCap->idStr, textureHash)) {
            capturer.m_exporter.dumpImageToFile(ctx, BASE_DIR + lss::commonDirName::texDir,
                                                albedoTexFilename,
                                                imageView->image());
          }
          const std::string albedoTexPath = str::format(BASE_DIR + lss::commonDirName::texDir, albedoTexFilename);
          lssMat.albedoTexPath = albedoTexPath;
        } catch (const std::exception& e) {
          Logger::err(str::format("[GameCapturer] Failed to export D3D9 texture for material '", matName, "': ", e.what()));
        }
      } else {
        Logger::warn(str::format("[GameCapturer] D3D9 texture has invalid image for material: ", matName));
      }
    } else if (textureHash != 0 && textureHash != kEmptyHash) {
      // API-submitted material: locate the image via the texture-manager table by hash and export it.
      RtxTextureManager& textureManager = ctx->getCommonObjects()->getTextureManager();
      const auto& textureTable = textureManager.getTextureTable();

      const TextureRef* pFoundTexture = nullptr;
      for (const auto& textureRef : textureTable) {
        if (textureRef.isValid() && textureRef.getImageHash() == textureHash) {
          pFoundTexture = &textureRef;
          break;
        }
      }

      if (pFoundTexture && pFoundTexture->getImageView()) {
        auto* apiImageView = pFoundTexture->getImageView();
        if (apiImageView && apiImageView->image().ptr()) {
          const auto& imageInfo = apiImageView->image()->info();

          // Validate image has valid dimensions
          if (imageInfo.extent.width > 0 && imageInfo.extent.height > 0) {
            const std::string albedoTexFilename(albedoFileName(textureHash, matName));
            try {
              if (claimTextureExport(capturer.m_pCap->idStr, textureHash)) {
                capturer.m_exporter.dumpImageToFile(ctx, BASE_DIR + lss::commonDirName::texDir,
                                                    albedoTexFilename,
                                                    apiImageView->image());
              }
              const std::string albedoTexPath = str::format(BASE_DIR + lss::commonDirName::texDir, albedoTexFilename);
              lssMat.albedoTexPath = albedoTexPath;
            } catch (const std::exception& e) {
              Logger::warn(str::format("[GameCapturer] Failed to export API texture for material ", matName, ": ", e.what()));
            }
          } else {
            Logger::warn(str::format("[GameCapturer] API texture has invalid dimensions for material: ", matName,
                                     " (", imageInfo.extent.width, "x", imageInfo.extent.height, ")"));
          }
        } else {
          Logger::warn(str::format("[GameCapturer] API texture has null image for material: ", matName,
                                   " (hash: 0x", std::hex, textureHash, std::dec, ")"));
        }
      } else {
        Logger::warn(str::format("[GameCapturer] Could not resolve API texture hash for material: ", matName,
                                 " (hash: 0x", std::hex, textureHash, std::dec, ")"));
      }
    }

    // The PBR slots, exported the same way the albedo above is.
    //
    // These are reached through the instance rather than through LegacyMaterialData, because the legacy
    // material describes what the *game* submitted and holds colour textures only -- there is no field on it
    // that a normal or roughness map could occupy. The PBR data lives on the opaque surface material the
    // runtime built, and InstanceManager::bindMaterial copies its texture indices onto the instance for
    // exactly this purpose.
    //
    // Written with a distinct filename per slot: two slots of one material can legitimately hold the same
    // image -- a packed texture used as both roughness and metallic, say -- and naming the file after the
    // material alone would have the second export overwrite the first.
    {
      RtxTextureManager& textureManager = ctx->getCommonObjects()->getTextureManager();
      const auto& textureTable = textureManager.getTextureTable();

      const auto exportSlot = [&](const uint32_t textureIndex, const char* slotName) -> std::string {
        if (textureIndex == kSurfaceMaterialInvalidTextureIndex || textureIndex >= textureTable.size()) {
          return {};
        }
        const TextureRef& textureRef = textureTable[textureIndex];
        if (!textureRef.isValid() || textureRef.getImageView() == nullptr
            || textureRef.getImageView()->image().ptr() == nullptr) {
          return {};
        }
        const auto& imageInfo = textureRef.getImageView()->image()->info();
        if (imageInfo.extent.width == 0 || imageInfo.extent.height == 0) {
          return {};
        }

        const XXH64_hash_t slotHash = textureRef.getImageHash();
        if (slotHash == 0 || slotHash == kEmptyHash) {
          return {};
        }

        const std::string filename = str::format(slotName, "_", dxvk::hashToString(slotHash), ".dds");
        try {
          // Claimed on the hash so a texture shared by many materials is written once, matching the albedo
          // path. The slot name is part of the filename but not of the claim, because the same image in two
          // slots is still one file's worth of bytes.
          if (claimTextureExport(capturer.m_pCap->idStr, slotHash)) {
            capturer.m_exporter.dumpImageToFile(ctx, BASE_DIR + lss::commonDirName::texDir, filename,
                                                textureRef.getImageView()->image());
          }
        } catch (const std::exception& e) {
          Logger::warn(str::format("[GameCapturer] Failed to export ", slotName, " for material ", matName,
                                   ": ", e.what()));
          return {};
        }
        return str::format(BASE_DIR + lss::commonDirName::texDir, filename);
      };

      lssMat.normalTexPath = exportSlot(rtInstance.getNormalTextureIndex(), "normal");
      lssMat.roughnessTexPath = exportSlot(rtInstance.getRoughnessTextureIndex(), "roughness");
      lssMat.metallicTexPath = exportSlot(rtInstance.getMetallicTextureIndex(), "metallic");
    }

    lssMat.enableOpacity = bEnableOpacity;

    // Sampler state. LegacyMaterialData only carries a sampler on the D3D9 path, but the exporter writes
    // WrapModeU/V and FilterMode for every material regardless, so leaving this unset does not mean
    // "unspecified" -- it means the defaults in lss::Material::Sampler get written instead of the sampler
    // the surface is actually drawn with. For anything with texture coordinates outside 0..1 that is the
    // difference between a tiled texture and one flat edge texel.
    //
    // The API path can still recover it: the resolved surface material holds a sampler index into the
    // scene's sampler table, which is the same sampler the runtime draws with.
    const auto& sampler = materialData.getSampler();
    if (sampler == nullptr) {
      const uint32_t samplerIndex = rtInstance.getSamplerIndex();
      const auto& samplers = ctx->getCommonObjects()->getSceneManager().getSamplerTable();
      if (samplerIndex != kSurfaceMaterialInvalidTextureIndex && samplerIndex < samplers.size()
          && samplers[samplerIndex] != nullptr) {
        const auto& apiSamplerInfo = samplers[samplerIndex]->info();
        lssMat.sampler.addrModeU = apiSamplerInfo.addressModeU;
        lssMat.sampler.addrModeV = apiSamplerInfo.addressModeV;
        lssMat.sampler.filter = apiSamplerInfo.magFilter;
        lssMat.sampler.borderColor = apiSamplerInfo.borderColor;
      } else {
        Logger::warn(str::format("[GameCapturer] No sampler resolved for API material ", matName,
                                 " (sampler index ", samplerIndex, "); the capture keeps the default "
                                 "repeat/linear rather than the sampler the surface is drawn with"));
      }
    }
    if (sampler != nullptr) {
      const auto& samplerCreateInfo = sampler->info();
      lssMat.sampler.addrModeU = samplerCreateInfo.addressModeU;
      lssMat.sampler.addrModeV = samplerCreateInfo.addressModeV;
      lssMat.sampler.filter = samplerCreateInfo.magFilter;
      lssMat.sampler.borderColor = samplerCreateInfo.borderColor;
    }

    // Cache by runtime hash for proper replacement lookup.
    capturer.m_pCap->materials[runtimeMaterialHash].lssData = lssMat;
    Logger::debug("[GameCapturer][" + capturer.m_pCap->idStr + "][Mat:" + matName + "] New");
  }

  // ---------------------------------------------------------------------------
  // captureCoordSystemSkip
  //
  // Applies the global coordinate-system transform for the USD export stage.
  // When the game is configured as a left-handed coordinate system, the
  // handedness inversion is skipped entirely: external API content is already
  // in consistent Y-up space, so inverting would produce a wrong-handed result.
  // ---------------------------------------------------------------------------
  void captureCoordSystemSkip(lss::Export& exportPrep) {
    // For external API content with explicit left-handed coordinate system setting,
    // skip the coordinate transformation - it's already in consistent Y-up space.
    // External API cameras are always LHS, so if the game is configured as LHS,
    // assume external content.
    if (!RtxOptions::leftHandedCoordinateSystem()) {
      const bool bInvX = (!exportPrep.camera.view.bInv) && (exportPrep.camera.proj.bInv || exportPrep.camera.isLHS());
      const bool bInvY = (!exportPrep.camera.view.bInv) && exportPrep.camera.proj.bInv;
      const pxr::GfVec3d scale{ bInvX ? -1.0 : 1.0, bInvY ? -1.0 : 1.0, 1.0 };
      exportPrep.globalXform.SetScale(scale);
    }
  }

} // namespace fork_hooks
} // namespace dxvk
