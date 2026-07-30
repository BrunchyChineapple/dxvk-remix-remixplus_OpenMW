/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

// Terrain baking for API-submitted draws.
//
// bakeDrawCall, the D3D9 path, re-issues the game's own draw into the cascade render target using the
// game's vertex and pixel pipeline. That is why it needs DxvkContextState, DxvkRaytracingInstanceState
// and DrawParameters: it is replaying a draw, and the texture stages and their matrices ride along
// inside the D3D9 constant buffers it saves and restores.
//
// An API host has no such pipeline to replay, and does not need one. The bake is an orthographic
// projection straight down at the ground, and a terrain layer's UV is a function of horizontal position
// only, so the map from cascade texel to layer UV is affine and the whole bake is a 2D resample. That
// removes the vertex data, the pipeline state, the render target, the viewport bookkeeping and the
// constant-buffer juggling in one go, and leaves a compute dispatch whose entire input is six affine
// rows.
//
// Everything downstream is shared with the D3D9 path untouched: the same cascade images, the same
// material published through getMaterialData, the same viewToCascade0TextureSpace transform and the same
// TexGenMode::CascadedViewPositions sampling in the ray tracing shaders.

#include "rtx_context.h"
#include "rtx_terrain_baker.h"
#include "rtx_scene_manager.h"
#include "rtx_texture_manager.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx/pass/terrain/fork_terrain_bake.h"

#include <rtx_shaders/fork_terrain_bake.h>

namespace dxvk {

  namespace {
    class TerrainBakeShader : public ManagedShader {
      SHADER_SOURCE(TerrainBakeShader, VK_SHADER_STAGE_COMPUTE_BIT, fork_terrain_bake)

      PUSH_CONSTANTS(TerrainBakeArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(TERRAIN_BAKE_DIFFUSE)
        TEXTURE2D(TERRAIN_BAKE_MASK)
        RW_TEXTURE2D(TERRAIN_BAKE_OUTPUT)
        SAMPLER(TERRAIN_BAKE_SAMPLER)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(TerrainBakeShader);

    // Applies one row of a world-to-UV map to a world position.
    inline float applyRow(const Vector4& row, const Vector3& world) {
      return row.x * world.x + row.y * world.y + row.z * world.z + row.w;
    }

    // Composes a world-to-UV map with a tile-to-world map into a single tile-to-UV row pair, which is
    // what the shader consumes. Both are affine, so the composition is exact and costs two extra
    // evaluations per row rather than a matrix multiply in the shader.
    inline void composeRows(const Vector4& worldU, const Vector4& worldV,
                            const Vector3& tileOrigin, const Vector3& tileAxisU, const Vector3& tileAxisV,
                            float4& outU, float4& outV) {
      const float baseU = applyRow(worldU, tileOrigin);
      const float baseV = applyRow(worldV, tileOrigin);

      // The axes are directions, so the row's translation term must not be applied to them; taking the
      // difference against the origin's value removes it.
      outU = float4 { applyRow(worldU, tileOrigin + tileAxisU) - baseU,
                      applyRow(worldU, tileOrigin + tileAxisV) - baseU,
                      baseU, 0.0f };
      outV = float4 { applyRow(worldV, tileOrigin + tileAxisU) - baseV,
                      applyRow(worldV, tileOrigin + tileAxisV) - baseV,
                      baseV, 0.0f };
    }
  }

  void TerrainBaker::onFrameBeginExternal(Rc<RtxContext> ctx) {
    m_needsMaterialDataUpdate = true;

    // updateTextureFormat is deliberately not called: its whole body warns when the bound render target
    // is sRGB, and there is no bound render target here. The format the cascades are created with comes
    // from getTerrainTexture, not from the context.
    calculateBakingParameters(ctx);

    // Clearing every frame is not optional on this path, unlike the D3D9 one where it is an option.
    // A compositing bake accumulates into the cascade with read-modify-write, so last frame's ground
    // would otherwise show through wherever this frame's terrain does not cover it -- and the cascades
    // are camera-relative, so that is a moving smear behind the player rather than a static error.
    for (uint32_t i = 0; i < ReplacementMaterialTextureType::Count; i++) {
      if (m_materialTextures[i].texture.isValid()) {
        clearMaterialTexture(ctx, static_cast<ReplacementMaterialTextureType::Enum>(i));
      }
    }
  }

  bool TerrainBaker::bakeExternalLayer(Rc<RtxContext> ctx, const DrawCallState& drawCallState,
                                       const ExternalLayer& layer, bool isFirstLayerOfChunk,
                                       Matrix4& textureTransformOut) {
    if (!layer.diffuse.isValid()) {
      return false;
    }

    ScopedGpuProfileZone(ctx, "Terrain Baker: Bake External Layer");

    RtxTextureManager& textureManager = ctx->getCommonObjects()->getTextureManager();
    const uint32_t currentFrameIndex = ctx->getDevice()->getCurrentFrameId();

    if (m_bakingParams.frameIndex != currentFrameIndex) {
      onFrameBeginExternal(ctx);
    }

    // Contribute to next frame's cascade footprint. The BBOX is consumed by calculateTerrainBBOX in
    // onFrameEnd, so a chunk registered now sizes the cascade map from the following frame -- same
    // one-frame lag the D3D9 path has, and the reason a first frame falls back to defaultHalfWidth.
    // Only once per chunk, or a chunk with five layers would weigh five times as much as one with one.
    if (cascadeMap.useTerrainBBOX() && isFirstLayerOfChunk) {
      m_terrainMeshBBOXes.emplace_back(AxisAlignedBoundingBoxLink(drawCallState));
    }

    const RtxMipmap::Resource& cascadeResource = getTerrainTexture(
      ctx, textureManager, ReplacementMaterialTextureType::AlbedoOpacity,
      m_bakingParams.cascadeMapResolution.width, m_bakingParams.cascadeMapResolution.height);

    const Rc<DxvkImageView>& cascadeView =
      cascadeResource.views.empty() ? cascadeResource.view : cascadeResource.views[0];

    if (cascadeView == nullptr) {
      ONCE(Logger::err("[RTX Terrain Baker] No albedo cascade texture available for an API terrain draw. Skipping the bake."));
      return false;
    }

    if (!TerrainBaker::debugDisableBinding()) {
      textureTransformOut = m_bakingParams.viewToCascade0TextureSpace;
    }

    if (TerrainBaker::debugDisableBaking()) {
      return true;
    }

    m_materialTextures[ReplacementMaterialTextureType::AlbedoOpacity].markAsBaked();

    const Rc<DxvkSampler> linearSampler = ctx->getResourceManager().getSampler(
      VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);

    // The mask binding must always be populated even when this layer has none, because a descriptor left
    // unwritten is not merely unread -- it is invalid, and validation layers reject the dispatch.
    const bool hasMask = layer.mask.isValid();

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, TerrainBakeShader::getShader());
    ctx->bindResourceView(TERRAIN_BAKE_DIFFUSE, layer.diffuse.getImageView(), nullptr);
    ctx->bindResourceView(TERRAIN_BAKE_MASK,
                          hasMask ? layer.mask.getImageView() : layer.diffuse.getImageView(), nullptr);
    ctx->bindResourceView(TERRAIN_BAKE_OUTPUT, cascadeView, nullptr);
    ctx->bindResourceSampler(TERRAIN_BAKE_SAMPLER, linearSampler);

    for (uint32_t iCascade = 0; iCascade < m_bakingParams.numCascades; iCascade++) {
      Vector2i cascade2DIndex;
      cascade2DIndex.y = iCascade / m_bakingParams.cascadeMapSize.x;
      cascade2DIndex.x = iCascade - cascade2DIndex.y * m_bakingParams.cascadeMapSize.x;

      // Cascade tile coordinates to world position.
      //
      // Derived from the cascade's own projection rather than recomputed from levelHalfWidth, so the two
      // cannot drift apart -- the last cascade in particular is expanded to cover the whole map and its
      // half width is not the formula the others follow.
      //
      // The y flip matches the negative-height viewport bakeDrawCall uses: clip +1 is the top of the
      // tile, so tile v of 0 is clip y of +1. Depth is irrelevant because the projection is orthographic
      // and the view looks straight down, which is what makes the whole map affine in two dimensions.
      const Matrix4 clipToWorld =
        m_bakingParams.inverseSceneView * inverse(m_bakingParams.bakingCameraOrthoProjection[iCascade]);

      auto tileToWorld = [&clipToWorld](float u, float v) {
        const Vector4 clip { 2.0f * u - 1.0f, 1.0f - 2.0f * v, 0.0f, 1.0f };
        const Vector4 world = clipToWorld * clip;
        const float invW = world.w != 0.0f ? 1.0f / world.w : 1.0f;
        return Vector3 { world.x * invW, world.y * invW, world.z * invW };
      };

      const Vector3 tileOrigin = tileToWorld(0.0f, 0.0f);
      const Vector3 tileAxisU = tileToWorld(1.0f, 0.0f) - tileOrigin;
      const Vector3 tileAxisV = tileToWorld(0.0f, 1.0f) - tileOrigin;

      TerrainBakeArgs args = {};
      composeRows(layer.chunkU, layer.chunkV, tileOrigin, tileAxisU, tileAxisV, args.chunkU, args.chunkV);
      composeRows(layer.diffuseU, layer.diffuseV, tileOrigin, tileAxisU, tileAxisV, args.diffuseU, args.diffuseV);
      if (hasMask) {
        composeRows(layer.maskU, layer.maskV, tileOrigin, tileAxisU, tileAxisV, args.maskU, args.maskV);
      }

      args.cascadeOffset = uint2 {
        cascade2DIndex.x * m_bakingParams.cascadeLevelResolution.width,
        cascade2DIndex.y * m_bakingParams.cascadeLevelResolution.height };
      args.cascadeExtent = uint2 {
        m_bakingParams.cascadeLevelResolution.width,
        m_bakingParams.cascadeLevelResolution.height };
      args.hasMask = hasMask ? 1u : 0u;

      ctx->pushConstants(0, sizeof(args), &args);

      const VkExtent3D workgroups = util::computeBlockCount(
        VkExtent3D { m_bakingParams.cascadeLevelResolution.width,
                     m_bakingParams.cascadeLevelResolution.height, 1 },
        VkExtent3D { TERRAIN_BAKE_TILE_SIZE, TERRAIN_BAKE_TILE_SIZE, 1 });
      ctx->dispatch(workgroups.width, workgroups.height, 1);
    }

    updateMaterialData(ctx);
    return true;
  }

}
