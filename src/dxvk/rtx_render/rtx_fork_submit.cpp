// src/dxvk/rtx_render/rtx_fork_submit.cpp
//
// Fork-owned file. Contains the implementations of fork_hooks:: functions
// for the SceneManager::submitExternalDraw path, lifted from
// rtx_scene_manager.cpp during the 2026-04-18 fork touchpoint-pattern refactor.
//
// See docs/fork-touchpoints.md for the full fork-hooks catalogue.
//
// NOTE: externalDrawObjectPicking accesses SceneManager::m_drawCallMeta, which
// is a private member. This file requires that SceneManager declare
// fork_hooks::externalDrawObjectPicking as a friend, OR that DrawCallMetaInfo
// and m_drawCallMeta be made accessible via a public accessor. Flagged for
// Phase 4 build-validation fixup.

#include "rtx_fork_hooks.h"

#include "rtx_asset_replacer.h"   // AssetReplacer, AssetReplacement
#include "rtx_options.h"          // RtxOptions::*, fast_unordered_set, InstanceCategories
#include "rtx_scene_manager.h"    // SceneManager, DrawCallMetaInfo
#include "rtx_context.h"          // RtxContext, for the terrain bake's Rc<RtxContext>
#include "rtx_terrain_baker.h"    // TerrainBaker::ExternalLayer, enableBaking, debugDisableBinding

#include <algorithm>              // std::max, std::min, std::sort for the candidate search
#include <cfloat>                 // DBL_MAX, for seeding the chunk footprint scan
#include <cmath>                  // std::abs
#include <atomic>                 // std::atomic, for the bounded coverage diagnostic
#include <cstring>                // std::memcpy, for reading packed vertex colour
#include <string>                 // std::string, for the unsolvable-draw diagnostic

#include "dxvk_device.h"          // DxvkDevice::getCommon()->getResources()
#include "../../util/util_env.h"  // env::getEnvVar, for the opt-in hash statistics

namespace dxvk {
namespace fork_hooks {

  // ---------------------------------------------------------------------------
  // externalDrawMeshReplacement
  //
  // Checks for a USD mesh/light replacement keyed on the API mesh handle hash,
  // same as the D3D9 draw-call path. Returns the replacement vector pointer if
  // one is found (caller must call determineMaterialData + drawReplacements and
  // then return), or null if no replacement exists.
  // ---------------------------------------------------------------------------
  std::vector<AssetReplacement>* externalDrawMeshReplacement(
      AssetReplacer& replacer, XXH64_hash_t meshHash) {
    // Check for mesh/light replacements keyed on the API mesh handle, same as
    // the D3D9 draw-call path. This lets .usd replacements target API-submitted
    // meshes (e.g. hash the remix API mesh handle and author a replacement).
    std::vector<AssetReplacement>* pReplacements = replacer.getReplacementsForMesh(meshHash);

    // Count hits and distinct misses, for the same reason the material lookup below does.
    //
    // Comparing an OpenMW capture's mesh_ names against a pack answers a different question than this
    // one, and on the material side that distinction wasted a lot of time: capture naming and runtime
    // binding are not the same thing. This is the only place that knows whether a mesh replacement
    // actually attached.
    //
    // Misses are counted by distinct hash rather than per draw, because a hash is looked up once per draw
    // and a handful of meshes would otherwise dominate the total. The distinct count is the useful one: it
    // says how many of the host's meshes are unknown to the pack.
    static std::atomic<uint64_t> s_lookups { 0 };
    static std::atomic<uint64_t> s_hits { 0 };
    static dxvk::mutex s_missMutex;
    static fast_unordered_set s_missedHashes;
    static fast_unordered_set s_hitHashes;
    // Distinct-identity tracking is opt-in, via DXVK_RTX_REPLACEMENT_HASH_STATS=1, and off by default.
    //
    // It exists only to feed the log line below, and it was costing a global mutex on a path that runs once
    // per draw -- every draw in the frame serialising to maintain a diagnostic. Capping the sets was not
    // enough: the cap has to be reached before the lock disappears, and a host that mints a fresh identity
    // per particle system per frame takes a long time to reach any sane cap. A measured session sat at
    // 38,983 distinct misses against a 65,536 cap, so it paid the lock for its entire length and never
    // saturated.
    //
    // With it off, the hot path touches two relaxed atomics and nothing else. The log line still reports
    // lookups and how many bound, which are the figures worth having continuously; the distinct-identity
    // breakdown is a coverage question, asked deliberately when someone wants it.
    static const bool s_trackDistinct = env::getEnvVar("DXVK_RTX_REPLACEMENT_HASH_STATS") == "1";

    const uint64_t lookupCount = s_lookups.fetch_add(1, std::memory_order_relaxed) + 1;
    if (pReplacements != nullptr) {
      s_hits.fetch_add(1, std::memory_order_relaxed);
    }

    const bool shouldLog = (lookupCount % 200000) == 0;

    if (s_trackDistinct) {
      std::lock_guard<dxvk::mutex> lock(s_missMutex);
      if (pReplacements != nullptr) {
        s_hitHashes.insert(meshHash);
      } else {
        s_missedHashes.insert(meshHash);
      }
      if (shouldLog && RtxOptions::ForkLogging::meshLookups()) {
        Logger::info(str::format("[RTX-Replacement] mesh lookups ", lookupCount, ": ",
          s_hits.load(std::memory_order_relaxed), " bound a replacement; distinct meshes ",
          s_hitHashes.size(), " matched and ", s_missedHashes.size(), " did not"));
      }
    } else if (shouldLog && RtxOptions::ForkLogging::meshLookups()) {
      Logger::info(str::format("[RTX-Replacement] mesh lookups ", lookupCount, ": ",
        s_hits.load(std::memory_order_relaxed),
        " bound a replacement (DXVK_RTX_REPLACEMENT_HASH_STATS=1 for the distinct-identity breakdown)"));
    }

    return pReplacements;
  }

  // ---------------------------------------------------------------------------
  // externalDrawMaterialReplacement
  //
  // Checks for a USD material replacement via getReplacementMaterial() and
  // updates the caller's material pointer in-place if one is found.
  // ---------------------------------------------------------------------------
  void externalDrawMaterialReplacement(
      AssetReplacer& replacer, const MaterialData*& material, MaterialData& mergeStorage) {
    // Say how often each lookup succeeds, because from the outside a bound replacement and an unbound one
    // can look identical.
    //
    // Most entries in a real pack are partial -- an `over` setting nothing but a roughness constant -- so a
    // replacement that binds correctly may produce no visible difference at all. That makes "it looks the
    // same" uninformative either way, and it is the only signal a host has otherwise. Counting the two
    // lookups separately also distinguishes a pack authored against summed API material hashes from one
    // authored against capture keys, which need completely different fixes.
    static std::atomic<uint64_t> s_lookups { 0 };
    static std::atomic<uint64_t> s_hitsByMaterialHash { 0 };
    static std::atomic<uint64_t> s_hitsByAlbedoHash { 0 };
    const uint64_t lookupCount = s_lookups.fetch_add(1, std::memory_order_relaxed) + 1;

    // Check for material replacement (matches the D3D9 draw path behavior).
    MaterialData* pReplacementMaterial = replacer.getReplacementMaterial(material->getHash());
    if (pReplacementMaterial != nullptr) {
      s_hitsByMaterialHash.fetch_add(1, std::memory_order_relaxed);
    }

    // Then by albedo texture hash, which is what a mat_<hex> key from a capture actually is.
    //
    // The two disagree for API materials and that is the whole problem. For a D3D9 draw
    // LegacyMaterialData::updateCachedHash sets the material hash *to* the albedo image hash, so a
    // capture names every material after its albedo texture. An API material instead sums every texture
    // slot and folds in its constants (rtx_material_data.h, WRITE_TEXTURE_HASH / WRITE_CONSTANT_HASH), so
    // the lookup above can never find a capture-authored key however faithful the host's texture hashes
    // are -- and a host that reproduces Remix's texture hash exactly, as the OpenMW integration now does,
    // otherwise gets nothing for it.
    //
    // Additive rather than a change of material identity: the summed hash is still tried first, so
    // materials authored against it keep working, and this only fills in the miss.
    if (pReplacementMaterial == nullptr && material->getType() == MaterialDataType::Opaque) {
      const auto& albedo = material->getOpaqueMaterialData().getAlbedoOpacityTexture();
      if (albedo.isValid()) {
        const XXH64_hash_t albedoHash = albedo.getImageHash();
        if (albedoHash != 0 && albedoHash != kEmptyHash) {
          pReplacementMaterial = replacer.getReplacementMaterial(albedoHash);
          if (pReplacementMaterial != nullptr) {
            s_hitsByAlbedoHash.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    }

    // Periodic rather than per-draw: this runs for every external draw, so anything per-call would drown
    // the log and cost more than the lookup.
    if (lookupCount % 2000000 == 0 && RtxOptions::ForkLogging::materialLookups()) {
      Logger::info(str::format("[RTX-Replacement] material lookups ", lookupCount,
        ": ", s_hitsByMaterialHash.load(std::memory_order_relaxed), " matched the material hash, ",
        s_hitsByAlbedoHash.load(std::memory_order_relaxed), " matched the albedo texture hash instead"));
    }

    if (pReplacementMaterial == nullptr) {
      return;
    }

    // Merge over the host's material rather than replacing it, which is what
    // SceneManager::determineMaterialData does for a D3D9 draw: it copies the replacement and then calls
    // mergeLegacyMaterial to fold the game's own material back in.
    //
    // Swapping wholesale was wrong and visibly so. Most entries in a real replacement pack are partial --
    // an `over` on a captured material that sets nothing but reflection_roughness_constant, with no
    // textures at all. merge() assigns from the argument for every field the USD did not explicitly author,
    // so the pack keeps its roughness and the albedo comes back from the host. Without it, every one of
    // those partial overrides handed the surface a material with no albedo texture and rendered it black.
    mergeStorage = *pReplacementMaterial;
    if (mergeStorage.getType() == MaterialDataType::Opaque
        && material->getType() == MaterialDataType::Opaque) {
      mergeStorage.getOpaqueMaterialData().merge(material->getOpaqueMaterialData());
    }
    material = &mergeStorage;
  }

  // ---------------------------------------------------------------------------
  // externalDrawTextureIdentity
  //
  // Resolves the albedo texture hash that identifies an API-submitted draw, for
  // category lookup and object picking. For API materials the albedo hash is
  // what D3D9's setupCategoriesForTexture() pattern keys off, so it is read
  // directly from the material's opaque data.
  //
  // Call this before externalDrawMaterialReplacement -- see the header for why
  // the order is load-bearing rather than incidental.
  // ---------------------------------------------------------------------------
  XXH64_hash_t externalDrawTextureIdentity(const MaterialData* material) {
    if (material == nullptr || material->getType() != MaterialDataType::Opaque) {
      return kEmptyHash;
    }

    const auto& albedo = material->getOpaqueMaterialData().getAlbedoOpacityTexture();
    return albedo.isValid() ? albedo.getImageHash() : kEmptyHash;
  }

  // ---------------------------------------------------------------------------
  // externalDrawTextureCategories
  //
  // Auto-applies all texture-based instance categories for API-submitted draws,
  // looking the already-resolved identity hash up against every RtxOption
  // category set.
  // ---------------------------------------------------------------------------
  void externalDrawTextureCategories(
      XXH64_hash_t outTextureHash,
      DrawCallState& drawCall) {
    if (outTextureHash != 0 && outTextureHash != kEmptyHash) {
      auto applyCategory = [&](const fast_unordered_set& hashSet, InstanceCategories cat) {
        if (hashSet.find(outTextureHash) != hashSet.end()) {
          drawCall.setCategory(cat, true);
        }
      };

      applyCategory(RtxOptions::skyBoxTextures(), InstanceCategories::Sky);
      applyCategory(RtxOptions::ignoreTextures(), InstanceCategories::Ignore);
      applyCategory(RtxOptions::worldSpaceUiTextures(), InstanceCategories::WorldUI);
      applyCategory(RtxOptions::worldSpaceUiBackgroundTextures(), InstanceCategories::WorldMatte);
      applyCategory(RtxOptions::particleTextures(), InstanceCategories::Particle);
      applyCategory(RtxOptions::beamTextures(), InstanceCategories::Beam);
      applyCategory(RtxOptions::decalTextures(), InstanceCategories::DecalStatic);
      applyCategory(RtxOptions::terrainTextures(), InstanceCategories::Terrain);

      // Let the vertex colour's alpha reach opacity for particles, which is where a particle's fade lives.
      //
      // The same defect as the terrain block below, in the other category the fork's own note said did not
      // exist: "restricted to terrain because it is the only category whose coverage lives in vertex alpha
      // rather than in its albedo". A particle is the second. osgParticle interpolates a colour range and an
      // alpha range across a particle's lifetime and multiplies them, and that product is the whole of how a
      // puff appears and dissipates -- the texture supplies the shape, the vertex alpha supplies the age.
      //
      // An API-submitted material leaves the alpha arguments at LegacyMaterialData's defaults, SelectArg1
      // with arg1 = Texture, which selects the albedo's alpha and discards the vertex colour. So the fade was
      // computed by the host, packed into color0, bound, read into surfaceInteraction.vertexColor.a, and then
      // never consulted: every particle stayed at the opacity of its texture for its whole life and vanished
      // at full strength instead of dissolving. On Dynamic Ambient Visual Effects candle smoke, whose NIF also
      // carries a 0.5 material emissive, that is a chain of hard-edged blobs rising at uniform brightness
      // rather than a wisp -- and because emissiveRadiance is scaled by this same alphaOpacity, the emission
      // did not taper either.
      //
      // Modulate rather than SelectArg2 so a texture that does carry alpha still contributes its shape; the
      // product of the two is what the rasteriser blends and what osgParticle's own renderer draws.
      //
      // API draws only, by virtue of living in this function. A legacy D3D9 particle derives these fields
      // from real fixed-function stage state, and overwriting that would be a regression rather than a fix.
      if (drawCall.testCategoryFlags(InstanceCategories::Particle)) {
        LegacyMaterialData& particleMaterial = drawCall.modifyMaterialData();
        particleMaterial.textureAlphaOperation = DxvkRtTextureOperation::Modulate;
        particleMaterial.textureAlphaArg1Source = RtTextureArgSource::Texture;
        particleMaterial.textureAlphaArg2Source = RtTextureArgSource::VertexColor0;
      }

      // Terrain-as-Decals, mirrored from the D3D9 layer.
      //
      // d3d9_rtx.cpp does this swap for legacy draws and nothing did it for API draws, which is why
      // selecting "Terrain-as-Decals" in the developer menu appeared to do nothing at all: the option was
      // read only inside the D3D9 draw path. The mode itself is not D3D9-specific -- it is a category
      // swap, and the decal pipeline it hands off to lives in rtx_render, which API draws already traverse.
      //
      // Kept identical to the legacy version on purpose, including the over-modulate compensation, so the
      // two paths cannot drift.
      if (drawCall.testCategoryFlags(InstanceCategories::Terrain)) {
        {
          // Let the vertex colour's alpha reach opacity, which is what carries a terrain layer's coverage.
          //
          // Without this the coverage is computed, submitted and then discarded. opaque_surface_material
          // _interaction.slangh composes opacity through chooseTextureArgument against
          // textureAlphaArg1Source / Arg2Source, and an API-submitted material leaves those at their
          // defaults -- Texture, None, SelectArg1 -- which selects the albedo's own alpha and nothing else.
          // A terrain diffuse is typically BC1 with no alpha at all, so opacity came out as a flat 1 and
          // every layer covered its chunk completely, which is exactly the hard-edged ground this is meant
          // to fix.
          //
          // Modulate rather than SelectArg2 so a layer diffuse that does carry alpha still contributes it.
          // Set here on the fork side rather than through InstanceInfoBlendEXT because that extension is
          // only read when the material sets useDrawCallAlphaState, which also diverts alpha *testing* to a
          // legacy draw call an API host does not have.
          LegacyMaterialData& terrainMaterial = drawCall.modifyMaterialData();
          terrainMaterial.textureAlphaOperation = DxvkRtTextureOperation::Modulate;
          terrainMaterial.textureAlphaArg1Source = RtTextureArgSource::Texture;
          terrainMaterial.textureAlphaArg2Source = RtTextureArgSource::VertexColor0;
        }

        // Terrain-as-Decals is a separate decision from the above, and conflating the two was a mistake.
        //
        // The coverage fix has to apply to every terrain draw whatever mode is selected, whereas the decal
        // swap is optional -- and harmful for a host that submits terrain per layer. It relabels the *base*
        // layer as a decal too, since the category is all it can see, leaving a stack of decals with no
        // opaque ground beneath them. A layer whose material is genuinely alpha blended already reaches the
        // unordered TLAS and accumulates transparency there, which is what a blended layer wants and needs
        // no decal offsetting at all.
        if (RtxOptions::terrainAsDecalsEnabledIfNoBaker() && !TerrainBaker::enableBaking()) {
          // Terrain is deliberately NOT removed, unlike the legacy path which swaps the category outright.
          //
          // Downstream fork code still needs to know this is terrain after the swap: the instance manager
          // forces the vertex-colour alpha arguments onto the surface for terrain draws, because the copy
          // from the draw call only runs under isFirstUpdateThisFrame and a chunk's layers share one
          // geometry, so they merge instead. Dropping the category here made that gate miss and the
          // coverage instruction reverted to LegacyMaterialData's defaults -- opacity from the albedo's own
          // alpha, which for a BC1 terrain diffuse is a flat 1.
          //
          // Keeping both is harmless: isDecal comes from DECAL_CATEGORY_FLAGS, which Terrain is not part of.
          drawCall.setCategory(InstanceCategories::DecalStatic, true);

          // The legacy path also promotes Modulate2x/4x to Force_Modulate2x when
          // terrainAsDecalsAllowOverModulate is set, to compensate for multilayer blending. Not mirrored:
          // that option defaults off and is not set here, the texture operation it rewrites is private to
          // LegacyMaterialData, and widening access for an unused path is not worth it. If terrain layers
          // come out too dark once they are submitted per pass, this is the knob that was left out.
        }
      }
      applyCategory(RtxOptions::animatedWaterTextures(), InstanceCategories::AnimatedWater);
      applyCategory(RtxOptions::ignoreLights(), InstanceCategories::IgnoreLights);
      applyCategory(RtxOptions::antiCullingTextures(), InstanceCategories::IgnoreAntiCulling);
      applyCategory(RtxOptions::motionBlurMaskOutTextures(), InstanceCategories::IgnoreMotionBlur);
      applyCategory(RtxOptions::hideInstanceTextures(), InstanceCategories::Hidden);
    }
  }

  // ---------------------------------------------------------------------------
  // externalDrawObjectPicking
  //
  // Stores per-draw texture hash metadata in SceneManager::m_drawCallMeta when
  // object picking is active, mirroring the D3D9 draw path which populates
  // m_drawCallMeta in processDrawCallState. API draws supply their own
  // drawCallID via remixapi_InstanceInfoObjectPickingEXT, so we store it here.
  //
  // ACCESS NOTE: this function uses SceneManager::m_drawCallMeta (private) and
  // SceneManager::DrawCallMetaInfo (private nested type). A friend declaration
  // for this function is required in SceneManager, or the member / type must be
  // made accessible via a public helper. See file-level comment above.
  // ---------------------------------------------------------------------------
  void externalDrawObjectPicking(
      DxvkDevice& device,
      DrawCallState& drawCall,
      XXH64_hash_t textureHash,
      SceneManager& scene) {
    // Store texture hash metadata for object picking (mirrors the D3D9 draw
    // path which populates m_drawCallMeta in processDrawCallState). API draws
    // supply their own drawCallID via remixapi_InstanceInfoObjectPickingEXT,
    // so we hash it in directly here.
    const bool objectPickingActive = device.getCommon()->getResources().getRaytracingOutput()
      .m_primaryObjectPicking.isValid();
    if (objectPickingActive && drawCall.drawCallID != 0 &&
        textureHash != 0 && textureHash != kEmptyHash) {
      auto meta = SceneManager::DrawCallMetaInfo {};
      meta.legacyTextureHash = textureHash;

      std::lock_guard lock { scene.m_drawCallMeta.mutex };
      auto [iter, isNew] = scene.m_drawCallMeta.infos[scene.m_drawCallMeta.ticker].emplace(drawCall.drawCallID, meta);
      ONCE_IF_FALSE(isNew, Logger::warn(
        "Found multiple API draw calls with the same \'objectPickingValue\'. "
        "Some objects might not be available through object picking"));
    }
  }

  // ---------------------------------------------------------------------------
  // externalDrawTerrainBake
  //
  // Composites one API-submitted terrain layer into the baker's cascade set, and repoints the draw at the
  // baked material. Mirrors what RtxContext::bakeTerrain does for a D3D9 draw, with the one difference
  // that the layer is composited rather than rasterised -- see rtx_fork_terrain_bake.cpp.
  //
  // Returns true when the draw was baked, in which case `material` now names the baked terrain material.
  // ---------------------------------------------------------------------------
  namespace {
    // Solves a*p + b*q + c = v for three samples, by Cramer's rule. Returns false when the samples are
    // collinear in the horizontal plane, which makes the system singular.
    bool solveAffine2D(const double p[3], const double q[3], const double v[3],
                       double& a, double& b, double& c) {
      const double det =
          p[0] * (q[1] - q[2]) - q[0] * (p[1] - p[2]) + (p[1] * q[2] - p[2] * q[1]);

      // Scale-relative rather than absolute: a chunk measured in tens of thousands of world units has a
      // determinant many orders of magnitude larger than one measured in metres, so a fixed epsilon would
      // either reject valid chunks or accept degenerate ones depending only on the game's unit scale.
      const double magnitude =
          std::max({ std::abs(p[0]), std::abs(p[1]), std::abs(p[2]),
                     std::abs(q[0]), std::abs(q[1]), std::abs(q[2]), 1.0 });
      if (std::abs(det) < 1e-9 * magnitude * magnitude) {
        return false;
      }

      const double invDet = 1.0 / det;
      a = invDet * (v[0] * (q[1] - q[2]) - q[0] * (v[1] - v[2]) + (v[1] * q[2] - v[2] * q[1]));
      b = invDet * (p[0] * (v[1] - v[2]) - v[0] * (p[1] - p[2]) + (p[1] * v[2] - p[2] * v[1]));
      c = invDet * (p[0] * (q[1] * v[2] - q[2] * v[1]) - q[0] * (p[1] * v[2] - p[2] * v[1])
                    + v[0] * (p[1] * q[2] - p[2] * q[1]));
      return true;
    }

    // Places a solved horizontal-plane affine back into a row applied to (x, y, z, 1).
    Vector4 rowForAxes(uint32_t axisP, uint32_t axisQ, double a, double b, double c) {
      Vector4 row { 0.0f, 0.0f, 0.0f, static_cast<float>(c) };
      (&row.x)[axisP] = static_cast<float>(a);
      (&row.x)[axisQ] = static_cast<float>(b);
      return row;
    }
  }

  bool externalDrawTerrainBake(const Rc<DxvkContext>& ctx, SceneManager& scene,
                               DrawCallState& drawCall, const MaterialData*& material) {
    // Overlay terrain layers are coplanar with the base layer and with each other, and a path tracer cannot
    // composite coplanar surfaces. Marking them as decals is what makes the blend physically possible.
    //
    // OpenMW's terrain blending is a rasteriser technique: draw the base, then draw each overlay at the same
    // coordinates with EqualDepth and SRC_ALPHA/ONE, and let the framebuffer accumulate. There is no
    // framebuffer here. A ray hits all of these surfaces at the same t, so there is no order to composite in
    // and traversal returns whichever it likes -- which is why correct coverage, correct blend type and a
    // correctly blend-enabled material still produced no blending. Nothing was wrong with the blend; it had
    // nowhere to happen.
    //
    // Remix already solves this for decals and says so in rtx.decalTextures' own description: a small offset
    // is applied "to prevent coplanar geometric cases (which poses problems for ray tracing)". The offset
    // supplies the ordering that coplanar geometry lacks, and enableDecalMaterialBlending then blends each
    // decal down onto what it sits above.
    //
    // SingleOffset rather than Static: Static offsets each coplanar part separately, and a terrain chunk is a
    // heightfield, so every triangle would take its own offset index -- inconsistent across the chunk and
    // quick to exhaust rtx.decals.maxOffsetIndex. SingleOffset gives one offset per draw call, and the global
    // index still advances between draws, so layer two lands above layer one in submission order.
    //
    // Only overlays. RtxOptions::terrainAsDecalsEnabledIfNoBaker cannot make this distinction because it
    // matches on texture hash and Morrowind reuses the same land textures for the base layer, so it tags the
    // ground itself and leaves a stack of decals over nothing. The coverage mask is the honest discriminator:
    // submitTerrainLayers sends one for every layer it draws over the base, and the base has none.
    // Terrain overlay layers are deliberately NOT tagged as decals, and the reason is worth recording so it
    // is not tried a fourth time.
    //
    // Decals are the right shape for the problem: resolve.slangh bins them and composites them onto the
    // surface underneath, sorted by surface.decalSortOrder, so coplanar geometry is fine and no offset is
    // needed. The implementation is sized for bullet holes. decalSortOrder is eight bits, packed at
    // rtx_materials.h:260 and assigned from a single global per-frame counter at rtx_instance_manager.cpp:1209
    // that wraps silently past 255 in a Release build; and numDecalResolveBins is 4, so at most four layers
    // composite at a pixel. A Morrowind exterior submits hundreds of terrain chunks with up to eight passes
    // each, so both limits are exceeded by a wide margin.
    //
    // Tagging terrain would also spend that shared 255-entry counter on ground, degrading the ordering of the
    // decals the game actually has -- blood, scorch marks, bullet holes -- so it is worse than merely
    // ineffective.
    if (!TerrainBaker::enableBaking() || !drawCall.testCategoryFlags(InstanceCategories::Terrain)) {
      return false;
    }
    if (material == nullptr || material->getType() != MaterialDataType::Opaque) {
      return false;
    }

    const RasterGeometry& geo = drawCall.getGeometryData();

    // The UV map is solved from the submitted vertices rather than transported, which keeps it correct
    // whether or not the host folded its own texture matrix into the texcoords it sent. That needs the
    // buffers to be host-visible; rtx_geometry_utils.cpp branches on exactly this for skinning.
    if (!geo.positionBuffer.defined() || !geo.texcoordBuffer.defined() || geo.vertexCount < 4) {
      ONCE(Logger::warn("[RTX Terrain Baker] An API terrain draw has no CPU-readable vertex data, so its "
                        "layer UV mapping cannot be derived. Leaving the draw unbaked."));
      return false;
    }

    const Matrix4& objectToWorld = drawCall.getTransformData().objectToWorld;

    // mapPtr takes the attribute's offset within the slice, and omitting it is not a subtle error.
    //
    // A GeometryBuffer is a view onto a slice of a larger buffer, and offsetFromSlice is where this
    // attribute begins inside it. API-submitted geometry interleaves everything in one vertex struct, so
    // the texcoord's offset is non-zero -- and mapPtr()'s argument defaults to 0, so the no-argument call
    // compiled and silently read texcoords from the position's bytes. That is why every terrain draw
    // reported collinear samples and nothing was ever baked: it was solving the position against itself.
    // rtx_types.h:527 is the pattern this should have followed.
    const uint8_t* const positionBase = static_cast<const uint8_t*>(
        geo.positionBuffer.mapPtr(static_cast<VkDeviceSize>(geo.positionBuffer.offsetFromSlice())));
    const uint8_t* const texcoordBase = static_cast<const uint8_t*>(
        geo.texcoordBuffer.mapPtr(static_cast<VkDeviceSize>(geo.texcoordBuffer.offsetFromSlice())));

    if (positionBase == nullptr || texcoordBase == nullptr) {
      return false;
    }

    // A spread of candidates, from which the triple that best defines a plane is chosen by measurement.
    //
    // Picking fixed indices does not work, and the way it fails is quiet. Terrain is a square grid, so the
    // obvious spread of 0, N/3, 2N/3 lands on exact multiples of the row stride whenever the grid's side
    // divides by three -- a 9x9 chunk gives 0, 27, 54, which are three vertices of the *same row*, with
    // identical world y and identical v. Perfectly collinear, and it is the reduced-LOD chunks that are
    // sized that way, so distant terrain silently refused to bake while near terrain worked.
    //
    // Sampling more widely and then choosing by determinant removes the whole class of problem: it does not
    // matter how the grid is laid out or how big it is, only that three of these candidates span it.
    constexpr uint32_t kMaxCandidates = 16;
    const uint32_t candidateCount = std::min(kMaxCandidates, geo.vertexCount);

    Vector3 world[kMaxCandidates];
    Vector2 uv[kMaxCandidates];
    uint32_t sampleIndex[kMaxCandidates];

    for (uint32_t i = 0; i < candidateCount; i++) {
      // Spread across the whole buffer, first and last inclusive.
      sampleIndex[i] = candidateCount > 1
          ? static_cast<uint32_t>((static_cast<uint64_t>(i) * (geo.vertexCount - 1u)) / (candidateCount - 1u))
          : 0u;

      const uint8_t* positionBytes = positionBase + geo.positionBuffer.stride() * sampleIndex[i];
      const uint8_t* texcoordBytes = texcoordBase + geo.texcoordBuffer.stride() * sampleIndex[i];

      const Vector3 object = *reinterpret_cast<const Vector3*>(positionBytes);
      const Vector4 transformed = objectToWorld * Vector4(object.x, object.y, object.z, 1.0f);
      world[i] = Vector3 { transformed.x, transformed.y, transformed.z };
      uv[i] = *reinterpret_cast<const Vector2*>(texcoordBytes);
    }

    // Which two axes span the ground is decided by trying them and checking, not by assuming.
    //
    // Shape alone cannot tell: the obvious test is that terrain is wide and flat so the thinnest axis is up,
    // but a small chunk on a mountainside is taller than it is wide, and that picks a vertical axis and makes
    // the solve genuinely degenerate. Shape still orders the candidates, because it is right almost always
    // and trying the likely one first keeps the common path at a single solve.
    //
    // The check is what makes it sound. A terrain layer's UV depends only on horizontal position, so a map
    // solved in the correct pair of axes reproduces the mesh's own UVs at every vertex, while one solved in a
    // pair that includes the vertical merely interpolates the three points it was handed. Testing the solved
    // map against every remaining candidate separates those two, and nothing cheaper does.
    Vector3 spread { 0.0f, 0.0f, 0.0f };
    for (uint32_t i = 1; i < candidateCount; i++) {
      spread.x = std::max(spread.x, std::abs(world[i].x - world[0].x));
      spread.y = std::max(spread.y, std::abs(world[i].y - world[0].y));
      spread.z = std::max(spread.z, std::abs(world[i].z - world[0].z));
    }

    uint32_t candidateUpAxis[3] = { 0u, 1u, 2u };
    std::sort(candidateUpAxis, candidateUpAxis + 3, [&spread](uint32_t lhs, uint32_t rhs) {
      return (&spread.x)[lhs] < (&spread.x)[rhs];
    });

    TerrainBaker::ExternalLayer layer;

    uint32_t axisP = 0;
    uint32_t axisQ = 1;
    bool solved = false;

    for (uint32_t attempt = 0; attempt < 3 && !solved; attempt++) {
      const uint32_t upAxis = candidateUpAxis[attempt];
      axisP = (upAxis + 1) % 3;
      axisQ = (upAxis + 2) % 3;

      const auto axisP_of = [&world, axisP](uint32_t i) { return static_cast<double>((&world[i].x)[axisP]); };
      const auto axisQ_of = [&world, axisQ](uint32_t i) { return static_cast<double>((&world[i].x)[axisQ]); };

      // The furthest candidate from the first, then the one furthest off that line. Standard way to avoid a
      // degenerate triangle, and here it also guarantees the triple is not a single grid row.
      uint32_t iB = 0;
      double bestDistance = 0.0;
      for (uint32_t i = 1; i < candidateCount; i++) {
        const double dp = axisP_of(i) - axisP_of(0);
        const double dq = axisQ_of(i) - axisQ_of(0);
        const double distance = dp * dp + dq * dq;
        if (distance > bestDistance) {
          bestDistance = distance;
          iB = i;
        }
      }
      if (iB == 0) {
        continue;
      }

      uint32_t iC = 0;
      double bestArea = 0.0;
      for (uint32_t i = 1; i < candidateCount; i++) {
        if (i == iB) {
          continue;
        }
        const double area = std::abs((axisP_of(iB) - axisP_of(0)) * (axisQ_of(i) - axisQ_of(0))
                                     - (axisQ_of(iB) - axisQ_of(0)) * (axisP_of(i) - axisP_of(0)));
        if (area > bestArea) {
          bestArea = area;
          iC = i;
        }
      }
      if (iC == 0) {
        continue;
      }

      const uint32_t triple[3] = { 0u, iB, iC };
      double p[3], q[3], u[3], v[3];
      for (uint32_t i = 0; i < 3; i++) {
        p[i] = axisP_of(triple[i]);
        q[i] = axisQ_of(triple[i]);
        u[i] = uv[triple[i]].x;
        v[i] = uv[triple[i]].y;
      }

      double aU, bU, cU, aV, bV, cV;
      if (!solveAffine2D(p, q, u, aU, bU, cU) || !solveAffine2D(p, q, v, aV, bV, cV)) {
        continue;
      }

      // Tolerance relative to the layer's own tiling, because that is the scale a UV error means anything
      // against: a terrain layer repeats many times across a chunk, so a UV of 16 is one part in sixteen
      // rather than a large absolute number.
      double uvScale = 1.0;
      for (uint32_t i = 0; i < candidateCount; i++) {
        uvScale = std::max({ uvScale, std::abs(static_cast<double>(uv[i].x)),
                             std::abs(static_cast<double>(uv[i].y)) });
      }
      const double tolerance = 1e-3 * uvScale;

      bool reproducesEveryCandidate = true;
      for (uint32_t i = 0; i < candidateCount && reproducesEveryCandidate; i++) {
        const double checkP = axisP_of(i);
        const double checkQ = axisQ_of(i);
        const double predictedU = aU * checkP + bU * checkQ + cU;
        const double predictedV = aV * checkP + bV * checkQ + cV;
        reproducesEveryCandidate = std::abs(predictedU - uv[i].x) <= tolerance
                                && std::abs(predictedV - uv[i].y) <= tolerance;
      }
      if (!reproducesEveryCandidate) {
        continue;
      }

      layer.diffuseU = rowForAxes(axisP, axisQ, aU, bU, cU);
      layer.diffuseV = rowForAxes(axisP, axisQ, aV, bV, cV);
      solved = true;
    }

    if (!solved) {
      // Reports what it read. "Not solvable" on its own cannot distinguish a degenerate chunk from samples
      // being taken badly, and both of those have now happened in turn.
      std::string samples;
      for (uint32_t i = 0; i < candidateCount; i++) {
        samples += str::format(" [", sampleIndex[i], "] (", world[i].x, ", ", world[i].y, ", ", world[i].z,
                               ") uv (", uv[i].x, ", ", uv[i].y, ")");
      }
      ONCE(Logger::warn(str::format(
          "[RTX Terrain Baker] No pair of axes yields a UV mapping that reproduces an API terrain draw's own "
          "texcoords, so it cannot be baked. ", candidateCount, " candidates of ", geo.vertexCount,
          " vertices; position stride ", geo.positionBuffer.stride(),
          " offset ", geo.positionBuffer.offsetFromSlice(),
          ", texcoord stride ", geo.texcoordBuffer.stride(),
          " offset ", geo.texcoordBuffer.offsetFromSlice(), ";", samples)));
      return false;
    }

    // The chunk footprint is measured from the vertices, not read from RasterGeometry::boundingBox.
    //
    // That bounding box is never populated for API-submitted geometry -- nothing in rtx_remix_api.cpp
    // writes it -- so it holds the default sentinel from util_bounding_box.h, minPos +FLT_MAX and maxPos
    // -FLT_MAX. Running that through a span-and-origin normalisation does not produce an obviously wrong
    // answer, which is what made it expensive: the span comes out 6.8e38, so the scale underflows to about
    // 1.5e-39 and the offset lands on -lo/span, which is exactly 0.5. The result is a chunk map that
    // reports 0.5 everywhere. Constant, therefore singular, therefore every cascade was skipped and the
    // bake dispatched nothing at all while looking entirely healthy from the outside.
    //
    // Only reached once the UV solve above has succeeded, which matters for cost: the Terrain category is
    // assigned by texture hash, so cliffs and rocks wearing a land texture arrive here too, and some of
    // them carry millions of vertices. Those fail the solve and return long before this scan.
    double minP = DBL_MAX, maxP = -DBL_MAX;
    double minQ = DBL_MAX, maxQ = -DBL_MAX;

    for (uint32_t i = 0; i < geo.vertexCount; i++) {
      const Vector3 object = *reinterpret_cast<const Vector3*>(
          positionBase + geo.positionBuffer.stride() * i);
      const Vector4 transformed = objectToWorld * Vector4(object.x, object.y, object.z, 1.0f);

      const double p = static_cast<double>((&transformed.x)[axisP]);
      const double q = static_cast<double>((&transformed.x)[axisQ]);

      minP = std::min(minP, p);
      maxP = std::max(maxP, p);
      minQ = std::min(minQ, q);
      maxQ = std::max(maxQ, q);
    }

    const double spanP = maxP - minP;
    const double spanQ = maxQ - minQ;

    if (spanP < 1e-6 || spanQ < 1e-6) {
      return false;
    }
    layer.chunkU = rowForAxes(axisP, axisQ, 1.0 / spanP, 0.0, -minP / spanP);
    layer.chunkV = rowForAxes(axisP, axisQ, 0.0, 1.0 / spanQ, -minQ / spanQ);

    // The coverage mask travels in the height slot; see Runtime::createTexturedMaterial in OpenMW for why
    // that slot and not a spare scalar.
    const OpaqueMaterialData& opaque = material->getOpaqueMaterialData();
    layer.diffuse = opaque.getAlbedoOpacityTexture();
    layer.mask = opaque.getHeightTexture();

    if (layer.mask.isValid()) {
      // The mask's UV mapping is stated by the host, through remixapi_MaterialInfoOpaqueTerrainEXT, and
      // composed here onto the world -> texcoord map solved above. Both halves are therefore exact: the
      // host's half is read straight off the texture matrix it draws with, and this half comes from the
      // submitted vertices.
      //
      // An earlier version of this derived the host's half instead, reading the blend map's width and
      // rebuilding OpenMW's inset from it. That was wrong twice over, which is a fair warning about the
      // approach rather than about the arithmetic. It read the scale off the wrong quantity -- OpenMW
      // upscales an ESM3 blend map 2x when it builds the image, so a 34-wide image belongs to a 16-quad
      // chunk and the width implies 33 -- and it dropped the recentring that OpenMW's scale-about-the-middle
      // carries. Neither is visible from this side of the boundary, which is the point: a host's sampling
      // convention is the host's to declare.
      const Vector3& maskTransformU = opaque.getTerrainMaskTransformU();
      const Vector3& maskTransformV = opaque.getTerrainMaskTransformV();

      // maskUv = row.x * u + row.y * v + row.z, where (u, v) is itself affine in world position. Composing
      // the two gives one row in world position, which is all the shader consumes.
      const auto composeMaskRow = [&](const Vector3& row) {
        return Vector4 {
          row.x * layer.diffuseU.x + row.y * layer.diffuseV.x,
          row.x * layer.diffuseU.y + row.y * layer.diffuseV.y,
          row.x * layer.diffuseU.z + row.y * layer.diffuseV.z,
          row.x * layer.diffuseU.w + row.y * layer.diffuseV.w + row.z };
      };

      layer.maskU = composeMaskRow(maskTransformU);
      layer.maskV = composeMaskRow(maskTransformV);
    }

    DrawCallTransforms& transformData = drawCall.modifyTransformData();

    // Registering every layer rather than only a chunk's first is harmless: calculateTerrainBBOX unions
    // the registered boxes, and union is idempotent.
    Rc<RtxContext> rtxContext = static_cast<RtxContext*>(ctx.ptr());
    if (!scene.getTerrainBaker().bakeExternalLayer(rtxContext, drawCall, layer, true,
                                                  transformData.textureTransform)) {
      return false;
    }

    if (TerrainBaker::debugDisableBinding()) {
      return true;
    }

    // Everything from here is what bakeTerrain does after a successful D3D9 bake, unchanged: the cascade
    // is sampled by generating texcoords from view positions in the ray tracing shaders rather than from
    // the mesh's own UVs, and the legacy material still has to name the baked texture and its sampler
    // because SceneManager patches the opaque material's samplers from it.
    material = scene.getTerrainBaker().getMaterialData();

    // Reports the binding half once, for the same reason the bake reports its own: a grey ground means the
    // albedo index never resolved, and that is indistinguishable from a bake that wrote nothing unless both
    // ends say what they saw. Remove alongside the bake's diagnostic.
    ONCE({
      if (material == nullptr) {
        Logger::warn("[RTX Terrain Baker] Bake succeeded but the baker published no material data.");
      } else {
        const TextureRef& albedo = material->getOpaqueMaterialData().getAlbedoOpacityTexture();
        const Matrix4& tf = transformData.textureTransform;
        Logger::warn(str::format(
          "[RTX Terrain Baker] Binding: albedo valid ", albedo.isValid() ? 1 : 0,
          ", imageEmpty ", albedo.isImageEmpty() ? 1 : 0,
          ", view ", albedo.getImageView() != nullptr ? 1 : 0,
          "; textureTransform rows (", tf[0].x, ", ", tf[0].y, ", ", tf[0].z, ", ", tf[0].w, ") (",
          tf[1].x, ", ", tf[1].y, ", ", tf[1].z, ", ", tf[1].w, ") (",
          tf[2].x, ", ", tf[2].y, ", ", tf[2].z, ", ", tf[2].w, ") (",
          tf[3].x, ", ", tf[3].y, ", ", tf[3].z, ", ", tf[3].w, ")"));
      }
    });

    transformData.texgenMode = TexGenMode::CascadedViewPositions;

    LegacyMaterialData overrideMaterial;
    overrideMaterial.colorTextures[0] = material->getOpaqueMaterialData().getAlbedoOpacityTexture();
    overrideMaterial.samplers[0] = scene.getTerrainBaker().getTerrainSampler();

    // Opacity comes from the cascade and from nowhere else, stated rather than left to the defaults.
    //
    // externalDrawTextureCategories ran just before this and pointed the alpha arguments at the vertex
    // colour, because that is where a terrain layer's coverage lives when nothing bakes it. Here it has
    // been baked: the cascade's own alpha already carries the composited coverage, and modulating it by
    // this layer's vertex alpha a second time would turn the finished ground into a semi-transparent
    // window onto whatever is behind it. Only overlaid layers would show it, so it would read as the
    // overlays being wrong rather than as double-counting.
    //
    // These three values happen to be LegacyMaterialData's defaults, so assigning a fresh one was already
    // enough. Written out anyway: the correctness of the bake should not rest on the default value of a
    // field in a struct owned by another subsystem.
    overrideMaterial.textureAlphaOperation = DxvkRtTextureOperation::SelectArg1;
    overrideMaterial.textureAlphaArg1Source = RtTextureArgSource::Texture;
    overrideMaterial.textureAlphaArg2Source = RtTextureArgSource::None;

    overrideMaterial.updateCachedHash();
    drawCall.modifyMaterialData() = overrideMaterial;

    return true;
  }

} // namespace fork_hooks
} // namespace dxvk
