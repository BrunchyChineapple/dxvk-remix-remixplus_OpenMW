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
#pragma once

#include "../dxvk/rtx_render/rtx_hashing.h"
#include <vulkan/vulkan_core.h>

#include "usd_include_begin.h"
#include <pxr/base/gf/matrix4d.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/sdf/path.h>
#include "usd_include_end.h"


#include <stdint.h>
#include <limits>
#include <map>

static_assert(std::numeric_limits<float>::is_iec559);
static_assert(std::numeric_limits<double>::is_iec559);

// While the interface for USD transform matrices implies that a variety of types are accepted, the documentation
// says that this is merely for consistency. You must provide a 4x4 matrix of doubles or you will get an error.

namespace lss {
using Id = size_t;
static constexpr Id kInvalidId(-1);

struct SampledXform {
  double time;
  pxr::GfMatrix4d xform;
};
using SampledXforms = std::vector<SampledXform>;

struct SampledBoneXform {
  double time;
  pxr::VtMatrix4dArray xforms;
};
using SampledBoneXforms = std::vector<SampledBoneXform>;

struct Skeleton {
  pxr::VtArray<pxr::TfToken> jointNames;
  pxr::VtMatrix4dArray bindPose;
  pxr::VtMatrix4dArray restPose;
};

enum CoordSys {
  RHS,
  LHS
};

struct Camera {
  // Note: FoV in radians.
  float         fov = NAN;
  float         aspectRatio = NAN;
  float         nearPlane = NAN;
  float         farPlane = NAN;
  float         firstTime = NAN;
  float         finalTime = NAN;
  bool          isReverseZ = false;
  SampledXforms xforms;
  struct CamMat {
    bool     bInv = false;
    CoordSys coord = RHS;
    inline bool isLHS() const { return coord == LHS; }
  } view, proj;
  // Do XOR here to check if we need to manually change basis for Projection * View matrix
  inline bool isLHS() const { return view.isLHS() ^ proj.isLHS(); }
};

struct SphereLight {
  std::string   lightName;
  float         color[3];
  float         radius;
  float         intensity;
  float         firstTime = NAN;
  float         finalTime = NAN;
  bool          shapingEnabled = false;
  float         coneAngleDegrees = 180.f;
  float         coneSoftness = 0.f;
  float         focusExponent = 0.f;
  SampledXforms xforms;
};

struct DistantLight {
  std::string  lightName;
  float        color[3];
  float        intensity;
  float        angleDegrees;
  pxr::GfVec3f direction;
  float        firstTime = NAN;
  float        finalTime = NAN;
};

struct Material {
  std::string matName;
  std::string albedoTexPath;
  bool        enableOpacity = false;
  // The alpha state, which is what actually tells the runtime a captured surface is not opaque.
  //
  // enableOpacity above is an MDL-side concept and does not reach RtSurface::AlphaState at all. The runtime
  // reads these six from the USD material instead -- see the property table in rtx_material_data.h, whose
  // second column is the USD name -- and InstanceManager::calculateAlphaState consumes them.
  //
  // useLegacyAlphaState is the one that matters most and it defaults to TRUE in that table. True means
  // "derive blending from the D3D9 draw call", and a capture being replayed has no D3D9 draw call, so a
  // material that says nothing resolves to fully opaque. That is why every captured particle opened as a
  // card: not a missing texture, not a missing opacity flag, but a material that never claimed to be
  // blended in the vocabulary the runtime reads. NVIDIA's own Morrowind pack sets exactly these by hand
  // (`custom bool inputs:blend_enabled`, `custom bool inputs:use_legacy_alpha_state`, ...), which is the
  // clearest evidence that this is the intended interface and that the exporter simply never wrote it.
  // The PBR constants. Found by listing what the runtime reads out of a USD material against what this
  // exporter writes back -- 44 properties were read and never written, and these are the ones the host
  // populates, so they were rendered and then dropped. A captured material previously came back at the
  // default roughness with no emissive, no metallic and no albedo tint.
  float       roughnessConstant = 0.7f;
  float       metallicConstant = 0.f;
  float       albedoConstant[3] { 1.f, 1.f, 1.f };
  float       opacityConstant = 1.f;
  bool        enableEmission = false;
  float       emissiveColorConstant[3] { 0.f, 0.f, 0.f };
  float       emissiveIntensity = 0.f;
  std::string emissiveTexPath;
  // Parallax, and the sprite sheet that animates a texture atlas. The sheet comes off RtSurface rather than
  // the surface material, since that is where the runtime keeps it.
  std::string heightTexPath;
  float       displaceIn = 0.f;
  float       displaceOut = 0.f;
  int         spriteSheetRows = 1;
  int         spriteSheetCols = 1;
  int         spriteSheetFps = 0;
  bool        useLegacyAlphaState = false;
  bool        blendEnabled = false;
  int         blendType = 0;          // BlendType::kAlpha
  bool        invertedBlend = false;
  int         alphaTestType = 7;      // AlphaTestType::kAlways, i.e. no test
  int         alphaTestReferenceValue = 0;
  // Defaulted because the exporter writes these unconditionally into every material's WrapModeU/V and
  // FilterMode attributes. A capture path that cannot supply a sampler -- which is any material submitted
  // through the API, where LegacyMaterialData carries none -- previously left them uninitialised, so
  // whatever was on the stack was written out as the wrap mode. Repeat and linear are the right defaults:
  // repeat is what tiled texture coordinates need, and a texture sampled with coordinates outside 0..1
  // under clamp collapses onto one edge texel and renders as a single flat colour.
  struct Sampler {
    VkSamplerAddressMode addrModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSamplerAddressMode addrModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkFilter             filter = VK_FILTER_LINEAR;
    VkClearColorValue    borderColor {};
  } sampler;
  // The PBR slots the capture format previously had no room for, which is why a captured material arrived
  // in the toolkit with only its albedo filled in no matter what the game supplied.
  //
  // A host that binds these -- OpenMW detects `_n`, `_nh` and `_spec` beside every diffuse and hands the
  // results to the API -- had them used for rendering and then dropped on export, so the toolkit could not
  // show them and an author could not swap them for something better. Each is empty when the material has
  // no such texture, and the exporter writes an attribute only for the ones that are set, so a material
  // with albedo alone still exports exactly as it did before.
  std::string normalTexPath;
  std::string roughnessTexPath;
  std::string metallicTexPath;
};

using Index = int;
using Pos = pxr::GfVec3f;
using Norm = pxr::GfVec3f;
using Texcoord = pxr::GfVec2f;
using Color = pxr::GfVec4f;
using BlendWeight = float;
using BlendIdx = int;
template <typename BufferT>
using Buf = pxr::VtArray<BufferT> ;
template<typename BufferT>
using BufSet = std::map<float,Buf<BufferT>>;
struct MeshBuffers {
  BufSet<Index>       idxBufs;
  BufSet<Pos>         positionBufs;
  BufSet<Norm>        normalBufs;
  BufSet<Texcoord>    texcoordBufs;
  BufSet<Color>       colorBufs;
  BufSet<BlendWeight> blendWeightBufs;
  BufSet<BlendIdx>    blendIndicesBufs;
};

struct RenderingMetaData {
  bool alphaTestEnabled;
  uint32_t alphaTestReferenceValue;
  uint32_t alphaTestCompareOp;
  bool alphaBlendEnabled;
  uint32_t srcColorBlendFactor;
  uint32_t dstColorBlendFactor;
  uint32_t colorBlendOp;
  uint32_t srcAlphaBlendFactor;
  uint32_t dstAlphaBlendFactor;
  uint32_t alphaBlendOp;
  uint32_t writeMask;
  uint32_t textureColorArg1Source;
  uint32_t textureColorArg2Source;
  uint32_t textureColorOperation;
  uint32_t textureAlphaArg1Source;
  uint32_t textureAlphaArg2Source;
  uint32_t textureAlphaOperation;
  uint32_t tFactor;
  bool isTextureFactorBlend;
  bool isVertexColorBakedLighting;
};

struct Mesh {
  std::string meshName;
  std::unordered_map<const char*, XXH64_hash_t> componentHashes;
  std::unordered_map<const char*, bool> categoryFlags;
  uint32_t     numVertices = 0;
  uint32_t     numIndices = 0;
  bool         isDoubleSided = false;
  Id           matId = kInvalidId;
  MeshBuffers  buffers;
  pxr::GfVec3f origin = pxr::GfVec3f{0.f,0.f,0.f};
  uint32_t     numBones = 0;
  uint32_t     bonesPerVertex = 0;
  pxr::VtMatrix4dArray boneXForms;
  bool         isLhs = false;
};

struct Instance {
  std::string       instanceName;
  float             firstTime = NAN;
  float             finalTime = NAN;
  Id                matId = kInvalidId;
  Id                meshId = kInvalidId;
  SampledXforms     xforms;
  bool              isSky;
  SampledBoneXforms boneXForms;
  RenderingMetaData metadata;
};

template <typename T>
using IdMap = std::unordered_map<Id,T>;
struct Export {
  std::string debugId;
  struct Meta {
    std::string windowTitle;
    std::string exeName;
    std::string iconPath;
    std::string geometryHashRule;
    double metersPerUnit;
    double timeCodesPerSecond;
    double startTimeCode;
    double endTimeCode;
    size_t numFramesCaptured;
    bool bReduceMeshBuffers;
    bool isZUp;
    std::unordered_map<std::string, std::string> renderingSettingsDict;
    bool bCorrectBakedTransforms;
  } meta;
  std::string baseExportPath;
  bool bExportInstanceStage;
  std::string instanceStagePath;
  std::string bakedSkyProbePath;
  pxr::SdfPath omniDefaultCameraSdfPath;
  IdMap<Material> materials;
  IdMap<Mesh> meshes;
  IdMap<Instance> instances;
  Camera camera;
  IdMap<SphereLight> sphereLights;
  IdMap<DistantLight> distantLights;
  pxr::GfVec3f stageOrigin = pxr::GfVec3f{0.f,0.f,0.f};
  pxr::GfMatrix4d globalXform = pxr::GfMatrix4d{1.0};
};

}