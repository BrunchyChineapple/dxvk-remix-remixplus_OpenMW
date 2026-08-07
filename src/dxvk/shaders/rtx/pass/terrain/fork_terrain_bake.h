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
#pragma once

// Both inputs are combined image samplers, so each takes a view and a sampler at the same binding. That
// is the convention every other sampling compute pass here follows -- see generate_mipmap.comp.slang --
// and it is not merely stylistic: the descriptor types declared in the shader and in the pipeline's
// parameter list have to agree, or the sampled result is undefined. They did not agree in the first
// version of this pass, which read every diffuse texture as zero and baked the ground black.
//
// It also lets the two inputs take different samplers, which they need: a diffuse tiles across the chunk
// and must repeat, while a coverage mask is stretched once over it and must clamp.
#define TERRAIN_BAKE_DIFFUSE        0
#define TERRAIN_BAKE_MASK           1
#define TERRAIN_BAKE_OUTPUT         2

#define TERRAIN_BAKE_TILE_SIZE      8

// Bakes one terrain layer of one draw call into one cascade level.
//
// Everything here is an affine map from normalised cascade-tile coordinates to a texture UV, because
// the bake is an orthographic top-down projection and a terrain layer's UV is a function of horizontal
// world position alone. The chain (tile coordinate -> world position -> layer UV) is therefore affine
// end to end and is composed on the CPU, which is why the shader needs no matrices, no vertex data and
// no knowledge of scene orientation or the cascade's projection.
//
// Each row is (a, b, c) evaluated as a*u + b*v + c, with the fourth component unused. Two rows per map.
struct TerrainBakeArgs {
  // Tile coordinate -> chunk UV. Defines coverage: the layer contributes only where this lands inside
  // the unit square, which is exactly the draw's own footprint on the ground.
  float4 chunkU;
  float4 chunkV;

  // Tile coordinate -> diffuse UV. Carries the layer's tiling, so it repeats many times per chunk.
  float4 diffuseU;
  float4 diffuseV;

  // Tile coordinate -> mask UV. Stretched once across the chunk, which is where a layer's coverage
  // actually lives; see components/terrain/material.cpp in OpenMW.
  float4 maskU;
  float4 maskV;

  // Where this cascade level sits in the combined cascade map, and how big it is, in texels.
  uint2 cascadeOffset;
  uint2 cascadeExtent;

  // First texel of the cascade level this dispatch covers, so the dispatch can be sized to the draw's
  // footprint instead of to the level.
  //
  // A chunk usually occupies a small part of a cascade -- the whole point of a cascade map is that the far
  // levels are coarse and cover ground far past the chunk in hand -- and dispatching a whole level per
  // chunk per layer costs about 2M workgroups that do nothing but fail the coverage test below. The rect
  // is only a bound on the work: the coverage test remains the authority on what gets written, so a rect
  // that is slightly too generous is merely slower and one that is too tight would lose ground.
  uint2 tileOffset;

  // Zero for the base layer, which has no mask and covers its chunk fully.
  uint hasMask;

  // Mip level to take the diffuse from, computed on the CPU because a compute dispatch has no implicit
  // derivatives and therefore no automatic filtering.
  //
  // One cascade texel covers several source texels -- about 2.8 of them for a 4096 level against a 2048
  // land texture, and proportionally more as the level resolution drops -- so taking mip 0 undersamples by
  // that factor. It does not look like a filtering bug from inside the game; it looks like the ground
  // having a fine speckle that the source texture does not have.
  float diffuseMip;
};
