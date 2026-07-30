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

#include <algorithm>              // std::max for the solver's scale-relative epsilon
#include <cmath>                  // std::abs

#include "dxvk_device.h"          // DxvkDevice::getCommon()->getResources()

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
    return replacer.getReplacementsForMesh(meshHash);
  }

  // ---------------------------------------------------------------------------
  // externalDrawMaterialReplacement
  //
  // Checks for a USD material replacement via getReplacementMaterial() and
  // updates the caller's material pointer in-place if one is found.
  // ---------------------------------------------------------------------------
  void externalDrawMaterialReplacement(
      AssetReplacer& replacer, const MaterialData*& material, MaterialData& mergeStorage) {
    // Check for material replacement (matches the D3D9 draw path behavior).
    MaterialData* pReplacementMaterial = replacer.getReplacementMaterial(material->getHash());

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
        }
      }
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

          ONCE(Logger::info(str::format(
              "[RTX Terrain] Vertex-colour coverage enabled for API terrain draws. Decal swap: ",
              (RtxOptions::terrainAsDecalsEnabledIfNoBaker() && !TerrainBaker::enableBaking()) ? "on" : "off")));
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
          drawCall.removeCategory(InstanceCategories::Terrain);
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
    if (geo.positionBuffer.mapPtr() == nullptr || !geo.texcoordBuffer.defined()
        || geo.texcoordBuffer.mapPtr() == nullptr || geo.vertexCount < 3) {
      ONCE(Logger::warn("[RTX Terrain Baker] An API terrain draw has no CPU-readable vertex data, so its "
                        "layer UV mapping cannot be derived. Leaving the draw unbaked."));
      return false;
    }

    const Matrix4& objectToWorld = drawCall.getTransformData().objectToWorld;

    // Spread the three samples across the mesh. Adjacent vertices in a terrain grid are very nearly
    // collinear, which would make the solve singular for a reason that has nothing to do with the chunk
    // being degenerate.
    const uint32_t sampleIndex[3] = { 0u, geo.vertexCount / 3u, (2u * geo.vertexCount) / 3u };

    Vector3 world[3];
    Vector2 uv[3];
    for (uint32_t i = 0; i < 3; i++) {
      const uint8_t* positionBytes =
          static_cast<const uint8_t*>(geo.positionBuffer.mapPtr()) + geo.positionBuffer.stride() * sampleIndex[i];
      const uint8_t* texcoordBytes =
          static_cast<const uint8_t*>(geo.texcoordBuffer.mapPtr()) + geo.texcoordBuffer.stride() * sampleIndex[i];

      const Vector3 object = *reinterpret_cast<const Vector3*>(positionBytes);
      const Vector4 transformed = objectToWorld * Vector4(object.x, object.y, object.z, 1.0f);
      world[i] = Vector3 { transformed.x, transformed.y, transformed.z };
      uv[i] = *reinterpret_cast<const Vector2*>(texcoordBytes);
    }

    // Which two axes are horizontal is decided by the chunk's own shape rather than by rtx.zUp: terrain is
    // wide and flat, so the axis with the least extent is the vertical one. That keeps this correct under
    // any scene orientation without having to agree with the runtime about which convention is in force.
    const AxisAlignedBoundingBox& bbox = geo.boundingBox;
    const Vector3 extent = bbox.maxPos - bbox.minPos;
    uint32_t upAxis = 0;
    for (uint32_t i = 1; i < 3; i++) {
      if ((&extent.x)[i] < (&extent.x)[upAxis]) {
        upAxis = i;
      }
    }
    const uint32_t axisP = (upAxis + 1) % 3;
    const uint32_t axisQ = (upAxis + 2) % 3;

    double p[3], q[3], u[3], v[3];
    for (uint32_t i = 0; i < 3; i++) {
      p[i] = (&world[i].x)[axisP];
      q[i] = (&world[i].x)[axisQ];
      u[i] = uv[i].x;
      v[i] = uv[i].y;
    }

    TerrainBaker::ExternalLayer layer;

    double a, b, c;
    if (!solveAffine2D(p, q, u, a, b, c)) {
      ONCE(Logger::warn("[RTX Terrain Baker] An API terrain draw's sample vertices are collinear, so its "
                        "layer UV mapping is not solvable. Leaving the draw unbaked."));
      return false;
    }
    layer.diffuseU = rowForAxes(axisP, axisQ, a, b, c);
    if (!solveAffine2D(p, q, v, a, b, c)) {
      return false;
    }
    layer.diffuseV = rowForAxes(axisP, axisQ, a, b, c);

    // The chunk footprint comes from the geometry's own bounds, in world space. This is what defines the
    // layer's coverage, so every cascade texel outside it is left to whichever chunk does own it.
    const Vector4 boundsMin = objectToWorld * Vector4(bbox.minPos.x, bbox.minPos.y, bbox.minPos.z, 1.0f);
    const Vector4 boundsMax = objectToWorld * Vector4(bbox.maxPos.x, bbox.maxPos.y, bbox.maxPos.z, 1.0f);

    const double spanP = static_cast<double>((&boundsMax.x)[axisP]) - static_cast<double>((&boundsMin.x)[axisP]);
    const double spanQ = static_cast<double>((&boundsMax.x)[axisQ]) - static_cast<double>((&boundsMin.x)[axisQ]);
    if (std::abs(spanP) < 1e-6 || std::abs(spanQ) < 1e-6) {
      return false;
    }
    layer.chunkU = rowForAxes(axisP, axisQ, 1.0 / spanP, 0.0, -static_cast<double>((&boundsMin.x)[axisP]) / spanP);
    layer.chunkV = rowForAxes(axisP, axisQ, 0.0, 1.0 / spanQ, -static_cast<double>((&boundsMin.x)[axisQ]) / spanQ);

    // The coverage mask travels in the height slot; see Runtime::createTexturedMaterial in OpenMW for why
    // that slot and not a spare scalar.
    const OpaqueMaterialData& opaque = material->getOpaqueMaterialData();
    layer.diffuse = opaque.getAlbedoOpacityTexture();
    layer.mask = opaque.getHeightTexture();

    if (layer.mask.isValid()) {
      // OpenMW samples an (s+1)-wide blend map over an s-quad chunk, so its BlendmapTexMat insets the UVs
      // rather than using the full range: scale s/(s+1) with a quarter-texel nudge
      // (components/terrain/material.cpp). Reconstructed from the mask image's own width, so nothing has
      // to be sent for it, and applied as (uv + offset) * scale to match preMultTranslate's ordering.
      //
      // Getting this wrong does not merely soften the blend, it reintroduces the defect being fixed:
      // neighbouring chunks would each stretch their own mask, and their coverage would disagree along
      // the shared edge. If layers look shifted or seams persist at chunk boundaries, this is the line to
      // question.
      const uint32_t maskWidth = layer.mask.getImageView() != nullptr
          ? layer.mask.getImageView()->image()->info().extent.width : 0u;
      if (maskWidth > 1u) {
        const float s = static_cast<float>(maskWidth - 1u);
        const float scale = s / (s + 1.0f);
        const float offset = 1.0f / (4.0f * s);

        layer.maskU = Vector4 { layer.chunkU.x * scale, layer.chunkU.y * scale, layer.chunkU.z * scale,
                                (layer.chunkU.w + offset) * scale };
        layer.maskV = Vector4 { layer.chunkV.x * scale, layer.chunkV.y * scale, layer.chunkV.z * scale,
                                (layer.chunkV.w - offset) * scale };
      } else {
        layer.maskU = layer.chunkU;
        layer.maskV = layer.chunkV;
      }
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
    transformData.texgenMode = TexGenMode::CascadedViewPositions;

    LegacyMaterialData overrideMaterial;
    overrideMaterial.colorTextures[0] = material->getOpaqueMaterialData().getAlbedoOpacityTexture();
    overrideMaterial.samplers[0] = scene.getTerrainBaker().getTerrainSampler();
    overrideMaterial.updateCachedHash();
    drawCall.modifyMaterialData() = overrideMaterial;

    return true;
  }

} // namespace fork_hooks
} // namespace dxvk
