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

#include <algorithm>  // std::min, std::max for the dispatch rect
#include <atomic>     // std::atomic, for the one-shot diagnostic
#include <cfloat>     // FLT_MAX, for seeding the corner bounds
#include <cmath>      // std::abs, std::floor, std::ceil
#include <string>     // std::string, for the one-shot diagnostic

namespace dxvk {

  namespace {
    class TerrainBakeShader : public ManagedShader {
      SHADER_SOURCE(TerrainBakeShader, VK_SHADER_STAGE_COMPUTE_BIT, fork_terrain_bake)

      PUSH_CONSTANTS(TerrainBakeArgs)

      // SAMPLER2D, not TEXTURE2D plus a standalone SAMPLER. The shader declares these as Sampler2D, which
      // is a combined image sampler, and the descriptor types have to match or the sample is undefined --
      // it returned zero, so every layer baked black over the whole cascade.
      BEGIN_PARAMETER()
        SAMPLER2D(TERRAIN_BAKE_DIFFUSE)
        SAMPLER2D(TERRAIN_BAKE_MASK)
        RW_TEXTURE2D(TERRAIN_BAKE_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(TerrainBakeShader);

    // Applies one row of a world-to-UV map to a world position.
    inline float applyRow(const Vector4& row, const Vector3& world) {
      return row.x * world.x + row.y * world.y + row.z * world.z + row.w;
    }

    // The rect of a cascade level a draw can actually write, in texels, as [offset, offset + extent).
    struct TileRect {
      VkOffset2D offset;
      VkExtent2D extent;
    };

    // Inverts the composed tile-to-chunk-UV map to find which texels of a cascade level this draw covers.
    //
    // The shader writes a texel only where the chunk UV lands inside the unit square, so the texels that
    // can be written are the preimage of that square. Both maps are affine, so the preimage is the
    // quadrilateral through the images of the square's four corners, and its bounding box is the rect to
    // dispatch. Returns false when there is nothing to do -- either the map is singular, which happens when
    // a cascade's projection collapses against the chunk's plane, or the chunk falls outside this level.
    inline bool solveTileRect(const float4& chunkU, const float4& chunkV,
                              const VkExtent2D& cascadeExtent, TileRect& out) {
      // Rows are (a, b, c) evaluated as a*u + b*v + c, so the linear part is [[a1 b1], [a2 b2]].
      const float a1 = chunkU.x, b1 = chunkU.y, c1 = chunkU.z;
      const float a2 = chunkV.x, b2 = chunkV.y, c2 = chunkV.z;

      const float det = a1 * b2 - b1 * a2;

      // Scale-relative, for the same reason the UV solver's epsilon is: these coefficients are reciprocals
      // of a chunk's world extent, so their magnitude tracks the game's unit scale and a fixed epsilon
      // would mean different things in different games.
      const float magnitude = std::max({ std::abs(a1), std::abs(b1), std::abs(a2), std::abs(b2), 1.0f });
      if (std::abs(det) < 1e-9f * magnitude * magnitude) {
        return false;
      }

      const float invDet = 1.0f / det;

      float tuMin = FLT_MAX, tuMax = -FLT_MAX;
      float tvMin = FLT_MAX, tvMax = -FLT_MAX;

      for (uint32_t corner = 0; corner < 4; corner++) {
        const float cu = static_cast<float>(corner & 1u);
        const float cv = static_cast<float>((corner >> 1) & 1u);

        const float du = cu - c1;
        const float dv = cv - c2;

        const float tu = invDet * (b2 * du - b1 * dv);
        const float tv = invDet * (a1 * dv - a2 * du);

        tuMin = std::min(tuMin, tu);
        tuMax = std::max(tuMax, tu);
        tvMin = std::min(tvMin, tv);
        tvMax = std::max(tvMax, tv);
      }

      // One texel of slack on each side, then clamp to the level. The shader tests coverage per texel
      // regardless, so slack costs a few rejected threads while a tight bound risks dropping the edge
      // texel of a chunk and leaving a seam -- the asymmetry is the whole reason to be generous here.
      const auto toTexelRange = [](float lo, float hi, uint32_t limit, int32_t& outLo, int32_t& outHi) {
        outLo = static_cast<int32_t>(std::floor(lo * static_cast<float>(limit))) - 1;
        outHi = static_cast<int32_t>(std::ceil(hi * static_cast<float>(limit))) + 1;
        outLo = std::max(outLo, 0);
        outHi = std::min(outHi, static_cast<int32_t>(limit));
      };

      int32_t x0, x1, y0, y1;
      toTexelRange(tuMin, tuMax, cascadeExtent.width, x0, x1);
      toTexelRange(tvMin, tvMax, cascadeExtent.height, y0, y1);

      if (x1 <= x0 || y1 <= y0) {
        return false;
      }

      out.offset = VkOffset2D { x0, y0 };
      out.extent = VkExtent2D { static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0) };
      return true;
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

    // A tiling diffuse repeats; a coverage mask is stretched once across the chunk and must clamp, which is
    // also what OpenMW sets on its own blend map textures (chunkmanager.cpp). Sharing one repeating sampler
    // between them would wrap the mask at the chunk edge and put a neighbour's coverage on this chunk.
    const Rc<DxvkSampler> diffuseSampler = ctx->getResourceManager().getSampler(
      VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    const Rc<DxvkSampler> maskSampler = ctx->getResourceManager().getSampler(
      VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    // The mask binding must always be populated even when this layer has none, because a descriptor left
    // unwritten is not merely unread -- it is invalid, and validation layers reject the dispatch.
    const bool hasMask = layer.mask.isValid();

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, TerrainBakeShader::getShader());

    // View and sampler go to the same binding: these are combined image samplers.
    ctx->bindResourceView(TERRAIN_BAKE_DIFFUSE, layer.diffuse.getImageView(), nullptr);
    ctx->bindResourceSampler(TERRAIN_BAKE_DIFFUSE, diffuseSampler);

    ctx->bindResourceView(TERRAIN_BAKE_MASK,
                          hasMask ? layer.mask.getImageView() : layer.diffuse.getImageView(), nullptr);
    ctx->bindResourceSampler(TERRAIN_BAKE_MASK, hasMask ? maskSampler : diffuseSampler);

    ctx->bindResourceView(TERRAIN_BAKE_OUTPUT, cascadeView, nullptr);

    // Every other RTX compute pass in the runtime sets this before pushing constants, and leaving it out
    // wrote the bake's arguments into the D3D9 bank that the fixed-function path uses for its own.
    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    // One-shot instrumentation, deliberately verbose.
    //
    // Four attempts at this pass have failed in ways that could not be seen from the source, and each
    // diagnosis by inspection cost a play session and was wrong. So it states what it actually did, once
    // per process, rather than being reasoned about a fifth time. Remove when the bake is trusted.
    static std::atomic<bool> s_hasReported { false };
    const bool report = !s_hasReported.exchange(true);
    std::string diagnostic;
    uint32_t dispatchCount = 0;

    if (report) {
      const DxvkImageCreateInfo& cascadeInfo = cascadeView->image()->info();
      const DxvkImageView* diffuseView = layer.diffuse.getImageView();

      diagnostic = str::format(
        "[RTX Terrain Baker] First bake. numCascades ", m_bakingParams.numCascades,
        ", cascadeMapSize ", m_bakingParams.cascadeMapSize.x, "x", m_bakingParams.cascadeMapSize.y,
        ", cascadeLevelResolution ", m_bakingParams.cascadeLevelResolution.width, "x",
        m_bakingParams.cascadeLevelResolution.height,
        ", cascadeMapResolution ", m_bakingParams.cascadeMapResolution.width, "x",
        m_bakingParams.cascadeMapResolution.height,
        "; cascade image ", cascadeInfo.extent.width, "x", cascadeInfo.extent.height,
        " format ", static_cast<uint32_t>(cascadeInfo.format),
        " usage 0x", std::hex, cascadeInfo.usage, std::dec,
        " mipLevels ", cascadeInfo.mipLevels,
        ", perMipViews ", cascadeResource.views.size(),
        "; hasMask ", hasMask ? 1 : 0,
        ", diffuse ", diffuseView != nullptr ? diffuseView->image()->info().extent.width : 0u, "x",
        diffuseView != nullptr ? diffuseView->image()->info().extent.height : 0u,
        " format ", diffuseView != nullptr ? static_cast<uint32_t>(diffuseView->info().format) : 0u);
    }

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

      // Only the part of this level the chunk can reach. Skipping a level entirely is the common case for
      // the near cascades, which cover a few metres around the camera and are simply not on this chunk.
      TileRect rect;
      const bool haveRect = solveTileRect(args.chunkU, args.chunkV,
                                          m_bakingParams.cascadeLevelResolution, rect);

      if (report) {
        diagnostic += str::format(
          "\n  cascade ", iCascade, " chunkU (", args.chunkU.x, ", ", args.chunkU.y, ", ", args.chunkU.z,
          ") chunkV (", args.chunkV.x, ", ", args.chunkV.y, ", ", args.chunkV.z,
          ") diffuseU (", args.diffuseU.x, ", ", args.diffuseU.y, ", ", args.diffuseU.z, ")");
        if (haveRect) {
          diagnostic += str::format(" -> rect ", rect.offset.x, ",", rect.offset.y, " ",
                                    rect.extent.width, "x", rect.extent.height);
        } else {
          diagnostic += " -> SKIPPED (chunk not in this cascade, or map singular)";
        }
      }

      if (!haveRect) {
        continue;
      }

      args.cascadeOffset = uint2 {
        cascade2DIndex.x * m_bakingParams.cascadeLevelResolution.width,
        cascade2DIndex.y * m_bakingParams.cascadeLevelResolution.height };
      args.cascadeExtent = uint2 {
        m_bakingParams.cascadeLevelResolution.width,
        m_bakingParams.cascadeLevelResolution.height };
      args.tileOffset = uint2 { static_cast<uint32_t>(rect.offset.x), static_cast<uint32_t>(rect.offset.y) };
      args.hasMask = hasMask ? 1u : 0u;

      // How much of the source texture one cascade texel covers, in source texels, then the mip that
      // averages exactly that much. Standard mip selection, done here because the shader has no
      // derivatives to take: the composed rows give the UV change per unit of tile space directly, so
      // dividing by the level resolution gives the change per texel and no estimation is involved.
      //
      // Each cascade is coarser than the one before it, so each gets its own mip -- which is the whole
      // point of a cascade map and is lost entirely if every level samples mip 0.
      {
        const VkExtent3D diffuseExtent = layer.diffuse.getImageView()->image()->info().extent;

        const auto footprintAlong = [&](float rowUCoefficient, float rowVCoefficient, uint32_t levelDim) {
          const float du = rowUCoefficient * static_cast<float>(diffuseExtent.width)
                         / static_cast<float>(levelDim);
          const float dv = rowVCoefficient * static_cast<float>(diffuseExtent.height)
                         / static_cast<float>(levelDim);
          return std::sqrt(du * du + dv * dv);
        };

        const float footprint = std::max(
          footprintAlong(args.diffuseU.x, args.diffuseV.x, m_bakingParams.cascadeLevelResolution.width),
          footprintAlong(args.diffuseU.y, args.diffuseV.y, m_bakingParams.cascadeLevelResolution.height));

        args.diffuseMip = std::max(0.0f, std::log2(std::max(footprint, 1e-6f)));
      }

      ctx->pushConstants(0, sizeof(args), &args);

      const VkExtent3D workgroups = util::computeBlockCount(
        VkExtent3D { rect.extent.width, rect.extent.height, 1 },
        VkExtent3D { TERRAIN_BAKE_TILE_SIZE, TERRAIN_BAKE_TILE_SIZE, 1 });
      ctx->dispatch(workgroups.width, workgroups.height, 1);
      dispatchCount++;
    }

    if (report) {
      diagnostic += str::format("\n  dispatches issued: ", dispatchCount);
      Logger::warn(diagnostic);
    }

    // Leave the context as it was found.
    //
    // This is not tidiness. The context is shared with the ray tracing passes that run after the scene is
    // submitted, and a compute shader plus three resource views left bound corrupted them: every exterior
    // mesh came out flat brown, which reads as a material bug and is not one. rtx_terrain_baker.cpp does
    // the same thing after its own secondary-texture bind, for the same reason.
    ctx->bindResourceView(TERRAIN_BAKE_DIFFUSE, nullptr, nullptr);
    ctx->bindResourceSampler(TERRAIN_BAKE_DIFFUSE, nullptr);
    ctx->bindResourceView(TERRAIN_BAKE_MASK, nullptr, nullptr);
    ctx->bindResourceSampler(TERRAIN_BAKE_MASK, nullptr);
    ctx->bindResourceView(TERRAIN_BAKE_OUTPUT, nullptr, nullptr);

    updateMaterialData(ctx);
    return true;
  }

}
