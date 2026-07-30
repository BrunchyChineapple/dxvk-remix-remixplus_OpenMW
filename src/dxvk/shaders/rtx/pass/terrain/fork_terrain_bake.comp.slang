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
#include "rtx/pass/terrain/fork_terrain_bake.h"

layout(binding = TERRAIN_BAKE_DIFFUSE)
Sampler2D<float4> DiffuseSampler;

layout(binding = TERRAIN_BAKE_MASK)
Sampler2D<float4> MaskSampler;

layout(binding = TERRAIN_BAKE_OUTPUT)
RWTexture2D<float4> OutCascade;

layout(binding = TERRAIN_BAKE_SAMPLER)
SamplerState LinearSampler;

layout(push_constant)
ConstantBuffer<TerrainBakeArgs> cb;

float2 evaluate(float4 rowU, float4 rowV, float2 t) {
  const float3 h = float3(t, 1.0f);
  return float2(dot(rowU.xyz, h), dot(rowV.xyz, h));
}

// Composites one terrain layer over whatever the cascade already holds.
//
// Read-modify-write on the cascade rather than blending through a render target, because there is no
// rasterisation here to blend into. Correctness therefore depends on layers being dispatched in the
// order the game draws them, with a barrier between, which is what the host guarantees: OpenMW submits
// its terrain passes base-first and the bake follows that submission order.
[shader("compute")]
[numthreads(TERRAIN_BAKE_TILE_SIZE, TERRAIN_BAKE_TILE_SIZE, 1)]
void main(uint2 threadId : SV_DispatchThreadID) {
  if (any(threadId >= cb.cascadeExtent)) {
    return;
  }

  // Texel centre, so the sample lands mid-texel rather than on its corner.
  const float2 t = (float2(threadId) + 0.5f) / float2(cb.cascadeExtent);

  // Outside this draw's own patch of ground there is nothing to say, and saying nothing is required
  // rather than merely tidy: every other chunk writes these same cascade texels.
  const float2 chunkUv = evaluate(cb.chunkU, cb.chunkV, t);
  if (any(chunkUv < 0.0f) || any(chunkUv > 1.0f)) {
    return;
  }

  const float2 diffuseUv = evaluate(cb.diffuseU, cb.diffuseV, t);
  const float3 layer = DiffuseSampler.SampleLevel(diffuseUv, 0.0f).rgb;

  // The base layer has no mask and is opaque; overlaid layers take their coverage from the blend map's
  // alpha, which is the whole reason a single albedo texture cannot represent one of these layers and
  // the reason this bake exists.
  float coverage = 1.0f;
  if (cb.hasMask != 0) {
    const float2 maskUv = evaluate(cb.maskU, cb.maskV, t);
    coverage = saturate(MaskSampler.SampleLevel(maskUv, 0.0f).a);
  }

  const uint2 dst = cb.cascadeOffset + threadId;
  const float4 prev = OutCascade[dst];

  // Alpha accumulates as a coverage mark rather than being blended: it records that this texel has
  // terrain on it at all, which is what distinguishes baked ground from the cleared cascade.
  OutCascade[dst] = float4(lerp(prev.rgb, layer, coverage), max(prev.a, coverage));
}
