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

#define TERRAIN_BAKE_DIFFUSE        0
#define TERRAIN_BAKE_MASK           1
#define TERRAIN_BAKE_OUTPUT         2
#define TERRAIN_BAKE_SAMPLER        3

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

  // Zero for the base layer, which has no mask and covers its chunk fully.
  uint hasMask;
  uint pad0;
  uint pad1;
  uint pad2;
};
