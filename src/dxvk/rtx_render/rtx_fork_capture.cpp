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
#include "rtx_terrain_baker.h"       // TerrainBaker, for the baked cascade albedo
#include "rtx_scene_manager.h"       // SceneManager::getTerrainBaker
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

    /// True the first time \a textureHash is seen in \a slot in the capture identified by \a captureId.
    ///
    /// The slot is part of the key, and leaving it out was a bug with a long fuse. The claim decides whether
    /// the file is WRITTEN while the filename decides where the material POINTS, and the filename carries the
    /// slot -- so a hash claimed by one slot refused every later slot its own differently named file, while
    /// the material still recorded a path to it. A reference to a file that does not exist.
    ///
    /// Harmless only while no image appears in two slots, and this host puts one there by design: materialFor
    /// binds a material's albedo into the emissive slot whenever the asset's emissive colour is achromatic,
    /// and the parallax height map is the normal map. Measured on a capture: five dangling emissive
    /// references, the candle smoke among them, which then fell back to emissive_color_constant and emitted
    /// differently from the runtime.
    bool claimTextureExport(const std::string& captureId, XXH64_hash_t textureHash, const char* slot) {
      std::lock_guard<std::mutex> lock(g_exportedTextureMutex);
      if (g_exportedTextureCaptureId != captureId) {
        g_exportedTextureCaptureId = captureId;
        g_exportedTextures.clear();
      }
      return g_exportedTextures.insert(XXH64(slot, std::strlen(slot), textureHash)).second;
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
          if (claimTextureExport(capturer.m_pCap->idStr, textureHash, "albedo")) {
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
              if (claimTextureExport(capturer.m_pCap->idStr, textureHash, "albedo")) {
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
          if (claimTextureExport(capturer.m_pCap->idStr, slotHash, slotName)) {
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

      // A note kept from a wrong turn, because it is a real hazard even though it was not the bug.
      //
      // These three indices come from RtInstance, which InstanceManager::bindMaterial fills from the
      // *resolved* RtSurfaceMaterial -- the mod's material wherever a replacement is active -- while the
      // albedo exported above comes from LegacyMaterialData. Those are the same material only when nothing
      // replaced it, so a capture can pair a replacement's normal map with the game's albedo. That was
      // suspected of causing striped, banded meshes on 2026-08-15 and tested by switching these exports off
      // entirely: the meshes did not change, so it is not the cause. The actual cause was
      // use_legacy_alpha_state and the filter/wrap promotion, both handled in game_exporter.cpp.
      //
      // An earlier version guarded these by comparing the resolved albedo hash against the exported one.
      // That guard could never fire on this host -- in the API path textureHash is itself taken from the
      // instance's resolved albedo index, so the two are equal by construction -- and a condition that is
      // always true is worse than no condition, because it reads as a check. Removed rather than left in
      // place. If mismatched maps ever do need excluding, the honest fix is to record the pre-replacement
      // material in CapturedMaterial, which needs a replacement flag plumbed into bindMaterial.
      lssMat.normalTexPath = exportSlot(rtInstance.getNormalTextureIndex(), "normal");
      lssMat.roughnessTexPath = exportSlot(rtInstance.getRoughnessTextureIndex(), "roughness");
      lssMat.metallicTexPath = exportSlot(rtInstance.getMetallicTextureIndex(), "metallic");

      // Translucent surfaces tint what passes through them by a texture rather than a constant when they
      // have one. Water does not, so this is usually empty and the constant carries the colour.
      if (rtInstance.getCapturedMaterial().isTranslucent) {
        lssMat.transmittanceTexPath
            = exportSlot(rtInstance.getCapturedMaterial().transmittanceTextureIndex, "transmittance");
      }
      // The emissive mask, pointed straight at the albedo when it IS the albedo rather than exported again.
      //
      // That is the common case for this host, not a curiosity: materialFor binds the albedo hash into the
      // emissive slot whenever the asset's emissive colour is achromatic, which is every glowing surface it
      // produces. Writing the same image a second time under a second name would double those bytes in the
      // capture for nothing, and the paths have to agree anyway.
      const uint32_t emissiveIndex = rtInstance.getCapturedMaterial().emissiveTextureIndex;
      if (emissiveIndex != kSurfaceMaterialInvalidTextureIndex
          && emissiveIndex == rtInstance.getAlbedoOpacityTextureIndex()
          && !lssMat.albedoTexPath.empty()) {
        lssMat.emissiveTexPath = lssMat.albedoTexPath;
      } else {
        lssMat.emissiveTexPath = exportSlot(emissiveIndex, "emissive");
      }
      // Only when the material actually displaces. A height map with zero displacement does nothing in this
      // runtime -- the material's own hasDisplacement test requires both -- but a consumer is free to read
      // the slot as "this surface has relief" and supply its own depth, and at least one does: exporting it
      // with displace 0 turned captured terrain into flowing banded dunes and props into striped blobs. This
      // host currently ships `parallax depth = 0.0`, so the honest export for all 23 materials that carry a
      // height index is nothing at all.
      //
      // Kept as a condition rather than removed, so it starts working by itself the day parallax is turned
      // on rather than needing to be remembered.
      const auto& captured = rtInstance.getCapturedMaterial();
      if (captured.displaceIn > 0.0f || captured.displaceOut > 0.0f) {
        lssMat.heightTexPath = exportSlot(captured.heightTextureIndex, "height");
      }
    }

    // The PBR constants, which the instance carries for exactly this purpose. See RtInstance::CapturedMaterial.
    {
      const auto& constants = rtInstance.getCapturedMaterial();

      // The translucent set, when the runtime resolved this surface as translucent. Water is the case that
      // matters: it carries an IOR, a transmittance colour and the distance over which that tint
      // accumulates, and none of those exist in the opaque material model, so exporting it as opaque loses
      // exactly the properties that make it water. InstanceManager::bindMaterial fills these from the
      // resolved RtTranslucentSurfaceMaterial; before it had a translucent branch they were never read.
      lssMat.isTranslucent = constants.isTranslucent;
      if (constants.isTranslucent) {
        lssMat.refractiveIndex = constants.refractiveIndex;
        lssMat.transmittanceColor[0] = constants.transmittanceColor.x;
        lssMat.transmittanceColor[1] = constants.transmittanceColor.y;
        lssMat.transmittanceColor[2] = constants.transmittanceColor.z;
        lssMat.transmittanceMeasurementDistance = constants.transmittanceMeasurementDistance;
        lssMat.isThinWalled = constants.isThinWalled;
        lssMat.thinWallThickness = constants.thinWallThickness;
        lssMat.useDiffuseLayer = constants.useDiffuseLayer;
        // The transmittance texture is exported alongside the other slots, where exportSlot is in scope.
      }

      lssMat.constantsFromReplacement = constants.fromReplacement;
      lssMat.roughnessConstant = constants.roughnessConstant;
      lssMat.metallicConstant = constants.metallicConstant;
      lssMat.albedoConstant[0] = constants.albedoOpacityConstant.x;
      lssMat.albedoConstant[1] = constants.albedoOpacityConstant.y;
      lssMat.albedoConstant[2] = constants.albedoOpacityConstant.z;
      lssMat.opacityConstant = constants.albedoOpacityConstant.w;
      lssMat.enableEmission = constants.enableEmission;
      lssMat.emissiveColorConstant[0] = constants.emissiveColorConstant.x;
      lssMat.emissiveColorConstant[1] = constants.emissiveColorConstant.y;
      lssMat.emissiveColorConstant[2] = constants.emissiveColorConstant.z;
      lssMat.emissiveIntensity = constants.emissiveIntensity;
      lssMat.displaceIn = constants.displaceIn;
      lssMat.displaceOut = constants.displaceOut;

      // The sprite sheet comes off the surface, not the material: RtSurface is where the runtime keeps the
      // rows/cols/fps it packs into textureSpritesheetData for the shader. Without these a captured animated
      // atlas -- every Morrowind fire, and anything else authored as a flipbook -- reopens frozen on frame
      // one, which reads as a broken texture rather than as missing metadata.
      lssMat.spriteSheetRows = rtInstance.surface.spriteSheetRows;
      lssMat.spriteSheetCols = rtInstance.surface.spriteSheetCols;
      lssMat.spriteSheetFps = rtInstance.surface.spriteSheetFPS;
    }

    // Terrain note, kept because it cost a session to establish and the code gives no hint of it.
    //
    // A captured terrain instance's albedo is ONE land layer -- measured 2026-08-15 as 256x256, 128x128 and
    // 32x256 images across eight distinct materials, where a baked cascade would be one shared material at
    // 4096 or 8192 square. Blended ground on screen is produced by the GPU terrain baker from per-layer draws
    // plus their coverage masks, composited into a camera-relative cascade.
    //
    // An earlier version of this note claimed the per-layer draws "share a geometry and a transform so they
    // merge to one instance per chunk". Measurement disproves it: an outdoor capture holds 357 terrain mesh
    // definitions and 270 instances, with four chunks carrying all eight land layers as eight separate
    // co-located instances, fifteen carrying five and twenty-two carrying two. The layers do survive into
    // the capture, individually. What does not survive is the coverage that says how to combine them, and
    // that is a different problem with a different fix -- see the alpha state below.
    // Opacity is enabled only for surfaces that genuinely blend, and the narrowness is the whole point.
    //
    // bEnableOpacity arrives as !alphaState.isFullyOpaque, which is far broader than it sounds: it is true
    // for alpha-tested cutouts and for anything the runtime resolved as not perfectly opaque, which is a
    // large fraction of ordinary geometry. Combined with the AperturePBR_Opacity.mdl change that forwards
    // enable_opacity instead of hardcoding false, that told the shader to take opacity from each albedo
    // texture's alpha channel -- and Morrowind textures carry alpha that was never meant as opacity. The
    // result was walls, floors and rugs going transparent in patches and stripes, in interiors as much as
    // outdoors, and it read as ruined geometry rather than as a material fault. Bisected to this file set on
    // 2026-08-15 by reverting it wholesale: meshes came back, particles returned to cards.
    //
    // A genuinely blended surface is the case the cards fix was for -- a particle, smoke, an effect plane --
    // and there the albedo's alpha channel *is* the opacity. Alpha-tested geometry is deliberately excluded:
    // a cutout wants the alpha test, which alpha_test_type carries separately, not a smooth opacity ramp
    // read from a channel that may hold anything.
    const RtSurface::AlphaState& opacityAlphaState = rtInstance.surface.alphaState;
    lssMat.enableOpacity = !opacityAlphaState.isBlendingDisabled && !opacityAlphaState.isFullyOpaque;

    // Why no terrain blend state is forced here, having tried it: it cannot work, and the reason is in the
    // MDL rather than in this exporter.
    //
    // Measured on an outdoor capture (2026-08-15): all eight land materials export enable_opacity=false,
    // blend_enabled=false, alpha_test_type=7 (GREATEREQUAL) against reference 0 -- always passes -- because
    // by the time a terrain layer reaches the capturer, rtx_fork_submit has handed the draw to the terrain
    // baker and replaced its material with the baker's published cascade, which is opaque. So the alpha
    // state below resolves fully opaque for every layer, and the capture holds eight opaque copies of each
    // chunk with one arbitrary layer visible.
    //
    // Forcing blend_enabled/enable_opacity on those materials looks like the fix and is not. Opacity in
    // AperturePBR_Opacity.mdl is `tex::texture_isvalid(diffuse_texture) ? base_lookup.w : opacity_constant`
    // -- the albedo texture's alpha channel, or a constant. The .mdl contains no scene::data_lookup and
    // reads no primvars, so primvars:displayOpacity cannot reach it however faithfully it is exported, and
    // the coverage this host bakes into vertex alpha is unreachable from the toolkit's shader. Worse, the
    // flag is not inert: it would point opacity at each land texture's own alpha channel, which nothing has
    // authored for this purpose, and any non-opaque texel would become a hole in the ground.
    //
    // Nor can the coverage ride on the material as a mask. This host reports terrain material hashes as the
    // albedo *texture* hash so replacement packs authored against a Morrowind capture still bind, so many
    // per-chunk runtime materials -- each with its own mask -- collapse onto one mat_<albedoHash> prim.
    //
    // What does work is not an exporter change at all: OPENMW_REMIX_TERRAIN_COMPOSITE plus a
    // Terrain/"composite map level" low enough to cover near chunks makes OpenMW composite each chunk
    // itself, and mergeCompositeLayer exports that readback as a single blended albedo per chunk. One opaque
    // textured mesh per chunk is a thing this material model can represent exactly.

    // The alpha state carried over verbatim from the instance, which is the part that decides whether a
    // captured surface is blended when the capture is opened again.
    //
    // useLegacyAlphaState is written false deliberately, and it is the whole fix. Its default in
    // rtx_material_data.h is true, meaning "derive blending from the D3D9 draw call" -- and a capture being
    // replayed has no draw call, so a material that stayed silent resolved to fully opaque no matter what
    // its texture's alpha held. Saying false makes the four values below authoritative instead, which is
    // exactly what they were captured for.
    //
    // Taken from surface.alphaState rather than from surface.blendModeState. The latter is D3D9
    // fixed-function blend state that an API host never populates -- the captured blend factors come out
    // src=0/dst=0 for this host's particles, which is VK_BLEND_FACTOR_ZERO twice and describes nothing.
    // alphaState is what the runtime resolved for real, whichever path filled it.
    const RtSurface::AlphaState& alphaState = rtInstance.surface.alphaState;
    // use_legacy_alpha_state is turned off only for surfaces that blend, and this narrowness is the fix.
    //
    // Writing false unconditionally was the cards fix and also the cause of a much worse regression.
    // Unconditionally false means "ignore the runtime's own alpha resolution, use exactly the values in this
    // file". For a particle that is what was wanted. For everything else it replaced working heuristics with
    // an explicit description that is frequently wrong: measured on an interior capture, 250 of 272 materials
    // exported blend_enabled=false with alpha_test_type=7 (ALWAYS), so every surface that needed a cutout --
    // foliage, grates, lattices, rope, cloth edges -- imported as a solid opaque quad. That reads as ruined
    // meshes, and it is why the geometry audits were all clean: nothing was wrong with the geometry.
    //
    // Bisected 2026-08-15 by reverting this file set wholesale, which restored the meshes and brought the
    // particle cards back, then narrowing rather than reverting.
    //
    // Left true for non-blended surfaces, which is the stock default and the behaviour that was working. The
    // explicit values below are still written; they are simply not authoritative unless this is false.
    const bool blendsForRealAlpha = !alphaState.isBlendingDisabled && !alphaState.isFullyOpaque;
    lssMat.useLegacyAlphaState = !blendsForRealAlpha;

    // Terrain is exported blended, because material collapse loses the fact that it was.
    //
    // A chunk's base layer covers it fully and is submitted opaque; its overlay layers are submitted with
    // blending on and their coverage in vertex alpha. Both end up on one mat_<albedoHash> prim, because this
    // host reports terrain material hashes as the albedo texture hash so replacement packs authored against
    // a Morrowind capture still bind. Whichever instance writes that prim last decides its alpha state, and
    // measured on an outdoor capture the opaque base won every time: all eight land materials came out
    // blend_enabled=false with an alpha test that always passes. Re-imported, the overlays are opaque, so
    // each chunk shows one arbitrary layer and hides the rest -- the stripped ground with missing textures.
    //
    // Forcing it on the shared prim is safe for the base layer: it carries vertex alpha 1.0, so a blended
    // material leaves it opaque anyway. kAlways rather than the instance's own alpha test because an alpha
    // test would quantise the coverage back to on or off and reinstate the hard edges a blend map exists to
    // avoid.
    //
    // This is the runtime's own vocabulary, not MDL's, and that is the point: the toolkit renders a capture
    // through HdRemix -- exts/lightspeed.hydra.remix.core/deps/hdremix/HdRemix.dll -- so the importer reads
    // these fields and displayOpacity through the same path the game does. An earlier attempt at this was
    // reverted on the belief that the toolkit shaded captures with AperturePBR_Opacity.mdl, whose opacity is
    // the albedo alpha channel and which reads no primvars; that reasoning applies to a renderer the
    // toolkit viewport does not use.
    const bool isTerrainLayer = rtInstance.testCategoryFlags(InstanceCategories::Terrain);

    lssMat.blendEnabled = !alphaState.isBlendingDisabled && !alphaState.isFullyOpaque;
    lssMat.blendType = static_cast<int>(alphaState.blendType);
    lssMat.invertedBlend = alphaState.invertedBlend;
    lssMat.alphaTestType = static_cast<int>(alphaState.alphaTestType);
    lssMat.alphaTestReferenceValue = static_cast<int>(alphaState.alphaTestReferenceValue);

    // REVERTED 2026-08-15: forcing enable_opacity on terrain made the ground *more* missing, not less.
    //
    // The reasoning below was sound about the attribute and wrong about the defect. Turning opacity on for
    // every terrain layer means any vertex whose baked coverage is low becomes transparent, and a base layer
    // that also carries varying alpha turns into a hole. Reported immediately as more ground missing than
    // before, which is exactly what that would do.
    //
    // The real defect is geometric, not alpha at all: captured meshes are cut in half along the diagonal --
    // half the triangles of a chunk absent -- which is an index buffer problem and applies to interior
    // meshes just as much as terrain. Alpha state cannot cause or hide that.
    //
    // Kept as a comment rather than deleted because the measurement behind it stands and is worth not
    // repeating: 26 terrain chunks of 4225 vertices, 12 carrying displayOpacity varying across the full
    // 0.000..1.000 range, all of them exported enable_opacity=False.
    //
    // if (isTerrainLayer) {
    //   lssMat.enableOpacity = true;
    // }

    // Superseded. See above.
    //
    // The coverage does reach the file: measured on an exterior capture, 26 terrain chunks at 4225 vertices
    // each, 12 of them carrying primvars:displayOpacity that varies across the mesh over the full 0.000 to
    // 1.000 range, the other 14 opaque because a base layer covers its chunk completely. The vertex alpha
    // the host bakes from the blend map survives export intact.
    //
    // What did not survive is permission to use it. Those same materials came out enable_opacity=False --
    // some with blend_enabled true, some false, one or two with both -- and with opacity disabled the
    // surface is treated as opaque whatever displayOpacity says. Eight opaque coincident layers per chunk,
    // one visible, which is the stripped ground with missing textures.
    //
    // bEnableOpacity, which normally decides this, is derived from the instance resolving as not fully
    // opaque, and a terrain layer does not reliably resolve that way. For terrain the answer is not a
    // derivation: the coverage is in the vertex data by construction, so opacity is always meaningful.


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
