#pragma once

// rtx_fork_hooks.h — declarations for the fork-owned hook functions that
// upstream files call into. Each hook's implementation lives in a dedicated
// rtx_fork_*.cpp file, keeping upstream files' fork footprint to one-line
// call sites only.
//
// See docs/fork-touchpoints.md for the index of every hook and which
// upstream file calls it.

// Brings in AssetReplacer, AssetReplacement, MaterialData, DrawCallState,
// and XXH64_hash_t transitively via rtx_types.h.
#include "rtx_asset_replacer.h"

// util/rc for Rc<DxvkContext> in capture hook signatures.
#include "../../util/rc/util_rc_ptr.h"

// std::filesystem::path for textureHashPathLookup's input parameter.
#include <filesystem>

// remixapi_ErrorCode for mutateTextureHashOption return type.
#include <remix/remix_c.h>

// Windows types required for the overlay hooks (HWND, UINT, WPARAM, LPARAM).
// This project is Windows-only; WIN32_LEAN_AND_MEAN keeps the include small.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

// Forward-declare lss::Export for the coord-system hook (avoids pulling the
// full USD pxr include chain into this header).
namespace lss {
  struct Export;
} // namespace lss

// remixapi_LightHandle for the light-manager hooks. Guard against redefinition
// when rtx_light_manager.h is also included in the same translation unit.
#ifndef REMIXAPI_LIGHTHANDLE_DEFINED
#define REMIXAPI_LIGHTHANDLE_DEFINED
using remixapi_LightHandle = struct remixapi_LightHandle_T*;
#endif

// d3d9_device.h provides D3D9DeviceEx; avoid pulling it into every TU.
// Only the implementation files that call device-path hooks include it directly.
namespace dxvk { class D3D9DeviceEx; }

// RaytraceArgs (for updateAtmosphereConstants) and Resources::RaytracingOutput
// (for dispatchScreenOverlay) both require the full type to appear in a function
// declaration. rtx_resources.h is the canonical header for both; it is already
// pulled transitively by most translation units that include rtx_fork_hooks.h.
#include "rtx_resources.h"
#include "rtx/pass/raytrace_args.h"

// Tonemap shader args structs (ToneMappingApplyToneMappingArgs)
// needed by fork_hooks::populateTonemapOperatorArgs.
#include "rtx/pass/tonemap/tonemapping.h"

// Atomic counters in the inline noteExternalDraw below.
#include <atomic>

namespace dxvk {

  // Forward declarations for types whose full definitions the hook header does
  // not need, but whose names appear in hook signatures.
  class DxvkContext;
  class DxvkDevice;
  class GameCapturer;
  class GameOverlay;
  struct LightManager;
  class RtInstance;
  class RtxContext;
  class SceneManager;
  struct LegacyMaterialData;
  struct RtLight;
  struct TextureRef;

  namespace fork_hooks {

    // Constructs the RtxAtmosphere instance during RtxContext initialization.
    // Must be called after GlobalTime::get().init() in the RtxContext constructor.
    // NOTE: requires RtxContext to declare this as a friend for access to the
    // private m_atmosphere and m_device members. See rtx_context.h.
    // Implementation in rtx_fork_atmosphere.cpp.
    void initAtmosphere(RtxContext& ctx);

    // Sets constants.skyMode, detects sky mode transitions (clearing skybox
    // buffers when switching to Numos), and when Numos
    // is active ensures m_atmosphere exists, calls initialize /
    // computeLuts, and writes atmosphereArgs into the constant block.
    // NOTE: requires RtxContext to declare this as a friend for access to private
    // members m_atmosphere, m_lastSkyMode, m_skyColorFormat, m_skyRtColorFormat,
    // and m_device. See rtx_context.h.
    // Implementation in rtx_fork_atmosphere.cpp.
    void updateAtmosphereConstants(RtxContext& ctx, RaytraceArgs& constants);

    // Ensures m_atmosphere is initialized (idempotent) and binds the three
    // atmosphere LUT textures unconditionally, because they are declared in
    // common_bindings.slangh for all shader passes.
    // NOTE: requires RtxContext to declare this as a friend for access to private
    // members m_atmosphere and m_device. See rtx_context.h.
    // Implementation in rtx_fork_atmosphere.cpp.
    void bindAtmosphereLuts(RtxContext& ctx);

    // Returns true when the caller should skip rasterized sky rendering because
    // Numos mode is active.
    // No private-member access — uses only the public RtxOptions::skyMode() API.
    // No friend declaration needed.
    // Implementation in rtx_fork_atmosphere.cpp.
    bool injectRtxAtmosphereSkySkip();

    // Checks for a USD mesh/light replacement keyed on the API mesh handle hash.
    // Returns the replacement vector if one exists, null otherwise.
    // Call site is responsible for calling determineMaterialData + drawReplacements
    // and returning early when a non-null value is returned.
    // Implementation in rtx_fork_submit.cpp.
    std::vector<AssetReplacement>* externalDrawMeshReplacement(
      AssetReplacer& replacer, XXH64_hash_t meshHash);

    // Checks for a USD material replacement and updates the material pointer in-place.
    //
    // mergeStorage is caller-owned scratch that must outlive the use of `material`: a replacement is
    // merged over the host's material rather than swapped for it, so the result is a new object that
    // belongs to neither the replacer nor the caller. Written only when a replacement is found.
    // Implementation in rtx_fork_submit.cpp.
    void externalDrawMaterialReplacement(
      AssetReplacer& replacer, const MaterialData*& material, MaterialData& mergeStorage);

    // Resolves the albedo texture hash that identifies an API-submitted draw, for category lookup and
    // for object picking.
    //
    // MUST be called before externalDrawMaterialReplacement, and this ordering is the whole point of the
    // function existing separately. A replacement is merged over the host's material, so afterwards the
    // albedo slot may name a texture the USD supplied rather than one the host ever submitted. Reading
    // the identity from that merged material breaks two things at once: the category sets hold host
    // hashes and stop matching, so a replaced surface can never be tagged Terrain or Particle; and the
    // hash handed to object picking is one ImGUI::AddTexture never saw, so the developer menu finds no
    // feature flags for it and offers nothing but "Copy Texture hash".
    //
    // D3D9 has no equivalent problem because a legacy draw keeps its own material and replacements are
    // looked up from that hash rather than folded into it. This restores the same invariant.
    // Implementation in rtx_fork_submit.cpp.
    XXH64_hash_t externalDrawTextureIdentity(const MaterialData* material);

    // Auto-applies all texture-based instance categories (Sky, Ignore, WorldUI, etc.) for a draw whose
    // identity hash has already been resolved by externalDrawTextureIdentity.
    // Implementation in rtx_fork_submit.cpp.
    void externalDrawTextureCategories(
      XXH64_hash_t textureHash,
      DrawCallState& drawCall);

    // Stores per-draw texture hash metadata in SceneManager::m_drawCallMeta
    // when object picking is active, mirroring the D3D9 draw path.
    // NOTE: requires SceneManager to declare fork_hooks::externalDrawObjectPicking
    // as a friend (or m_drawCallMeta to be made public) for the implementation
    // in rtx_fork_submit.cpp to compile. Flagged for Phase 4 fixup.
    // Implementation in rtx_fork_submit.cpp.
    void externalDrawObjectPicking(
      DxvkDevice& device,
      DrawCallState& drawCall,
      XXH64_hash_t textureHash,
      SceneManager& scene);

    // Forwards keyboard (WM_KEY*, WM_CHAR, WM_SYSCHAR) AND mouse
    // (WM_MOUSEMOVE, WM_{L,R,M,X}BUTTON*, WM_MOUSE{,H}WHEEL) messages to
    // ImGui_ImplWin32_WndProcHandler so ImGui's keyboard + mouse state stays
    // in sync on the legacy WndProc path. Used when a game menu captures raw
    // input OR the plugin HUD pulls focus via the Remix API — either case
    // stops overlayWndProc from receiving messages directly and the legacy
    // wndProcHandler fallback becomes the only delivery path.
    // Mouse coords in lParam are translated from gameHwnd client-space to
    // overlayHwnd client-space when the two differ, so ImGui hit-tests
    // correctly. Wheel lParam is screen-space per Windows convention and
    // forwards without translation.
    // NOTE: requires GameOverlay to declare this as a friend for access to the
    // private m_hwnd member. See rtx_overlay_window.h.
    // Implementation in rtx_fork_overlay.cpp.
    void overlayInputForward(
      GameOverlay& overlay, HWND gameHwnd, UINT msg, WPARAM wParam, LPARAM lParam);

    // Alpha-composites a plugin-uploaded RGBA pixel buffer over the final
    // tone-mapped output image using the ScreenOverlayShader compute shader.
    // Called from RtxContext::dispatchScreenOverlay (one-line delegate) after
    // tone mapping and before screenshot capture. No-ops when no overlay is
    // pending this frame.
    // NOTE: requires RtxContext to declare this as a friend for access to
    // private members m_pendingScreenOverlay, m_screenOverlay*, and m_device.
    // See rtx_context.h.
    // Implementation in rtx_fork_overlay.cpp.
    void dispatchScreenOverlay(RtxContext& ctx, Resources::RaytracingOutput& rtOutput);

    // Implements the full captureMaterial logic for both D3D9 and API-submitted
    // materials. For D3D9 materials, exports the color texture directly. For
    // API-submitted materials (colorTexture invalid), resolves the albedo texture
    // via the texture-manager table and exports it by hash. Caches the result in
    // GameCapturer::m_pCap->materials keyed by runtimeMaterialHash.
    // NOTE: requires GameCapturer to declare this as a friend for access to
    // private members m_exporter and m_pCap. See rtx_game_capturer.h.
    // Implementation in rtx_fork_capture.cpp.
    void captureMaterialApiPath(
      GameCapturer& capturer,
      const Rc<DxvkContext> ctx,
      const RtInstance& rtInstance,
      XXH64_hash_t runtimeMaterialHash,
      const LegacyMaterialData& materialData,
      bool bEnableOpacity);

    // Applies the coordinate-system transform for the USD export stage.
    // Skips the view/proj handedness inversion when the game is configured
    // as a left-handed coordinate system, since external API content is
    // already in consistent Y-up space.
    // Implementation in rtx_fork_capture.cpp.
    void captureCoordSystemSkip(lss::Export& exportPrep);

    // Drains the three deferred external-light mutation queues (erases, updates,
    // activations) and auto-instances persistent lights. Called at the top of
    // LightManager::prepareSceneData before linearization.
    // NOTE: requires LightManager to declare this as a friend for access to
    // private members. See rtx_light_manager.h.
    // Implementation in rtx_fork_light.cpp.
    void flushPendingLightMutations(LightManager& mgr);

    // Shared static-sleep update logic for both the indexed (externally-tracked)
    // and hash-map (external API) light paths. Preserves temporal denoiser data
    // by skipping the copy once isStaticCount >= numFramesToPutLightsToSleep.
    // Pass externalId = kInvalidExternallyTrackedLightId to skip id restoration
    // (hash-map path). Stamps frameLastTouched unconditionally.
    // Implementation in rtx_fork_light.cpp.
    void updateLightStaticSleep(
      RtLight* light,
      const RtLight& newLight,
      DxvkDevice* device,
      uint64_t externalId);

    // Per-frame weather preset blender update. Reads __weather.target and
    // __weather.blend_seconds from the GameStateStore and writes blended weather
    // params to the Derived layer of their underlying RTX_OPTIONs. Dormant when
    // no target is set — zero behavioural change vs upstream.
    // NOTE: requires RtxContext to declare this as a friend for access to the
    // private m_weatherBlender member. See rtx_context.h. (Wired in Task 3.)
    // Implementation in rtx_fork_weather.cpp.
    void updateWeatherBlender(class RtxContext& ctx, float deltaTimeSeconds);

    // Emplaces a new external light into m_externalLights and stamps
    // frameLastTouched. Called from the "new light" branch of addExternalLight.
    // NOTE: requires LightManager to declare this as a friend for access to
    // m_externalLights. See rtx_light_manager.h.
    // Implementation in rtx_fork_light.cpp.
    void setExternalLightEmplace(
      LightManager& mgr,
      remixapi_LightHandle handle,
      const RtLight& rtlight);

    // Queues a light handle for deferred erase in m_pendingExternalLightErases.
    // Called from LightManager::removeExternalLight.
    // NOTE: requires LightManager to declare this as a friend for access to
    // m_pendingExternalLightErases. See rtx_light_manager.h.
    // Implementation in rtx_fork_light.cpp.
    void disableExternalLightQueue(LightManager& mgr, remixapi_LightHandle handle);

    // Inserts a handle into m_persistentExternalLights if non-null.
    // Called from LightManager::registerPersistentExternalLight.
    // NOTE: requires LightManager to declare this as a friend for access to
    // m_persistentExternalLights. See rtx_light_manager.h.
    // Implementation in rtx_fork_light.cpp.
    void registerPersistentLight(LightManager& mgr, remixapi_LightHandle handle);

    // Removes a handle from m_persistentExternalLights if non-null.
    // Called from LightManager::unregisterPersistentExternalLight.
    // NOTE: requires LightManager to declare this as a friend for access to
    // m_persistentExternalLights. See rtx_light_manager.h.
    // Implementation in rtx_fork_light.cpp.
    void unregisterPersistentLight(LightManager& mgr, remixapi_LightHandle handle);

    // Copies all persistent-light handles into m_pendingExternalActiveLights.
    // Called from LightManager::queueAutoInstancePersistent.
    // NOTE: requires LightManager to declare this as a friend for access to
    // m_persistentExternalLights and m_pendingExternalActiveLights. See
    // rtx_light_manager.h.
    // Implementation in rtx_fork_light.cpp.
    void queueAutoInstancePersistent(LightManager& mgr);

    // Pins ImGui and ImPlot to the dev menu's private contexts at wndProcHandler
    // entry. Prevents context corruption when plugin activity on other threads
    // drifts GImGui off the dev menu's context between frames.
    // Call site passes m_context and m_plotContext from ImGUI directly — no friend
    // declaration needed.
    // Implementation in rtx_fork_overlay.cpp.
    void imguiContextPin(struct ImGuiContext* ctx, struct ImPlotContext* plotCtx);

    // Renders the atmosphere preset buttons and parameter tree inside the
    // "Sky Tuning" collapsing header. Owns the skyModeCombo static and branches
    // on SkyMode::Numos vs SkyboxRasterization.
    // No private-member access — uses only public RtxOptions and ImGui APIs.
    // No friend declaration needed.
    // Implementation in rtx_fork_atmosphere.cpp.
    void showAtmosphereUI();

    // Invokes the registered plugin draw callback for the Plugin tab in the dev
    // menu. Called from the kTab_Wrapper switch case in ImGUI::showMainMenu.
    // No private-member access — delegates to remixapi_imgui_InvokeDrawCallback().
    // No friend declaration needed.
    // Implementation in rtx_fork_overlay.cpp.
    void wrapperTabDraw();

    // Attempts to resolve a material texture path of the form "0x<hex>" against
    // the texture manager's hash table (populated by API-uploaded textures via
    // remixapi_CreateTexture). If the path matches the hex-hash pattern and a
    // registered texture with that image hash exists, writes the resolved
    // TextureRef into outRef and returns true. Returns false in all other
    // cases (path is not a hex string, hash not found, parse failure), in
    // which case the caller must fall back to the normal asset-path lookup.
    // No private-member access — uses public TextureManager::getTextureTable().
    // No friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    bool textureHashPathLookup(
      DxvkContext& ctx,
      const std::filesystem::path& path,
      TextureRef& outRef);

    // Looks up an RtxOption<fast_unordered_set> by full option name and adds
    // (add == true) or removes (add == false) the parsed hash in the user
    // config layer. Used by remixapi_AddTextureHash / remixapi_RemoveTextureHash
    // to mutate per-category texture hash sets at runtime.
    // NOTE: the caller is responsible for holding the remix-api static mutex
    // (s_mutex in rtx_remix_api.cpp) across this call, per the lock ordering
    // rule documented alongside s_mutex.
    // No private-member access — uses only public RtxOption / RtxOptionLayer APIs.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode mutateTextureHashOption(
      const char* textureCategory,
      const char* textureHash,
      bool add);

    // Copies pPixelData into a host-visible staging buffer and stores it in the
    // fork-owned s_pendingScreenOverlay optional. A null pPixelData or zero dims
    // clears the pending overlay. The overlay is flushed to the render thread by
    // presentScreenOverlayFlush at the next present boundary.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode drawScreenOverlay(
      D3D9DeviceEx* remixDevice,
      const void*   pPixelData,
      uint32_t      width,
      uint32_t      height,
      remixapi_Format format,
      float         opacity);

    // Drains s_pendingScreenOverlay (if set) onto the render thread via EmitCs,
    // forwarding the staging buffer to RtxContext::setScreenOverlayData. Called
    // from both remixapi_Present paths (inner-namespace and extern-C) after the
    // light/mesh flush EmitCs and before the endScene callback dispatch.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    void presentScreenOverlayFlush(D3D9DeviceEx* remixDevice);

    // Reads RtxOptions::showUI() and maps the internal UIType to remixapi_UIState.
    // Returns REMIXAPI_UI_STATE_NONE if the device is not registered.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_UIState getUiState(D3D9DeviceEx* remixDevice);

    // Acquires the device lock and calls getImgui().switchMenu(uiType) after
    // mapping the remixapi_UIState to internal UIType. Returns GENERAL_FAILURE
    // if the device or its common objects are null.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode setUiState(D3D9DeviceEx* remixDevice, remixapi_UIState state);

    // Reports the exportable Win32 handle, dedicated-allocation size, tiling and
    // format behind a shared D3D9 surface, so a host owning its own window can
    // import Remix output into another API (OpenGL, for the OpenMW integration).
    // The surface must have been created with a non-null pSharedHandle. Only the
    // allocation size genuinely needs renderer-side help: Vulkan pads allocations
    // per driver, so the caller cannot derive it, and glTextureStorageMem2DEXT
    // requires it exactly. The handle stays owned by Remix.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode getSurfaceExternalMemory(
      D3D9DeviceEx*                     remixDevice,
      IDirect3DSurface9*                surface,
      remixapi_dxvk_ExternalMemoryInfo* out_info);

    // Creates an exportable image for a host to draw its 2D interface into, and reports the external
    // memory behind it so the host can import it. Remix then composites it over the path-traced image
    // every frame, once enabled.
    //
    // The alternative already exists and is DrawScreenOverlay, which takes CPU pixels. At 4K that is
    // roughly 33MB pushed across the bus per frame, which for the OpenMW host is precisely the round
    // trip that had to be removed to reach a playable frame time, so paying it again to get a menu on
    // screen is not a trade worth making.
    //
    // Remix creates the image rather than adapting one from a shared D3D9 surface, for control over
    // the format. D3D9 would force D3DFMT_A8R8G8B8, which DXVK maps to VK_FORMAT_B8G8R8A8_UNORM, and
    // an OpenGL producer writing RGBA8 into memory Vulkan reads as BGRA8 has its channels swapped.
    // Owning the allocation means asking for VK_FORMAT_R8G8B8A8_UNORM and needing no swizzle.
    //
    // Created in VK_IMAGE_LAYOUT_GENERAL deliberately. The producer is another API which does not
    // participate in Vulkan layout tracking, so there is no correct moment to transition; GENERAL can
    // be sampled and needs no ownership transfer.
    //
    // Calling again with different dimensions replaces the image and invalidates the previous handle.
    // The handle stays owned by Remix.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode createScreenOverlayImage(
      D3D9DeviceEx*                     remixDevice,
      uint32_t                          width,
      uint32_t                          height,
      remixapi_dxvk_ExternalMemoryInfo* out_info);

    // Turns compositing of the shared overlay image on or off, and sets its opacity.
    //
    // Separate from creation so a host can stop compositing without giving up the allocation -- during
    // a loading screen, for instance, where the interface is not being drawn and a stale image would
    // otherwise stay on screen.
    remixapi_ErrorCode setScreenOverlayEnabled(
      remixapi_Bool enabled,
      float         opacity);

    // Hands the shared overlay to the compositing pass, or reports that there is none.
    //
    // Exists because the state lives beside the other API entry points while the dispatch that
    // consumes it lives in rtx_fork_overlay.cpp.
    bool getSharedScreenOverlay(Rc<DxvkImageView>& out_view, float& out_opacity);

    // Creates on first call, and reports, the exportable binary semaphore pair used to order the
    // copy into a shared surface against a foreign API's sampling of it. Companion to
    // getSurfaceExternalMemory: that one makes the pixels reachable from another API, this one makes
    // reading them defined. Handles stay owned by Remix.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode getOutputSyncSemaphores(
      D3D9DeviceEx*                 remixDevice,
      remixapi_dxvk_OutputSyncInfo* out_info);

    // As the upstream copy entry point, but brackets the blit with the semaphore pair above so a
    // consumer in another API has a defined ordering. Deliberately separate from
    // remixapi_dxvk_CopyRenderingOutput rather than a flag on it, so the upstream path keeps its
    // exact behaviour for every existing caller.
    //
    // waitForConsumer is the caller's assertion that its consumer signalled since the last call.
    // Binary semaphores have to stay balanced and OpenGL cannot import a timeline semaphore, so an
    // unmatched wait would block the render thread with no way to recover. Only the caller knows
    // whether its consumer ran, so this is not inferred here.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    // signalCopyComplete false serves a consumer that can signal a shared semaphore but cannot wait on
    // one. The signal must then be suppressed rather than merely ignored: copyComplete is binary, so one
    // nobody waits on stays signalled and makes the next signal invalid.
    remixapi_ErrorCode copyRenderingOutputSynced(
      D3D9DeviceEx*                         remixDevice,
      IDirect3DSurface9*                    destination,
      remixapi_dxvk_CopyRenderingOutputType type,
      bool                                  waitForConsumer,
      bool                                  signalCopyComplete = true);

    // Records that the host has presented, which permanently disqualifies dispatchDevMenuOverlay.
    //
    // The two are drivers of the same ImGui frame and ImGui has one unlocked global context, so exactly
    // one of them may own a frame. Called from the present path, which is the winner; see
    // dispatchDevMenuOverlay for the reasoning and for the crash signature this closes.
    void notifyHostPresented();

    // Arms dispatchDevMenuOverlay for a host that consumes Remix's output through the copy entry
    // points rather than by presenting. Idempotent; called from copyRenderingOutputSynced, whose use
    // is itself the signal that such a host is driving the runtime.
    // hostWindow is handed to ImGui's Win32 backend, which derives its display size from that
    // window's client rect.
    // hostWindow defines the menu's coordinate space: ImGui takes its display size from that window's
    // client rect, and GameOverlay positions its raw-input sink over it and hit-tests the cursor
    // against it. It must therefore be the window the user actually sees and clicks in, which is not
    // necessarily the window Remix's swapchain was created on.
    //
    // overrideExisting distinguishes the host stating its choice from the runtime guessing: the copy
    // entry point arms this every frame with the swapchain's window as a fallback, and must not
    // clobber an explicit nomination.
    // Implementation in rtx_fork_overlay.cpp.
    void enableDevMenuOverlay(HWND hostWindow, bool overrideExisting);

    // Feeds ImGui the cursor position and mouse button state by polling, for a host nominated through
    // enableDevMenuOverlay.
    //
    // Needed because GameOverlay's sink delivers mouse buttons over raw input only, and raw-input
    // registration is per-process per-device: a natively hosted game that registers the mouse itself --
    // SDL does, for relative mouse mode -- replaces the sink's registration, and WM_INPUT stops
    // arriving. Cursor *position* keeps working, because ImGui's Win32 backend has a GetCursorPos
    // fallback for it, so the symptom is a menu whose cursor tracks the mouse perfectly and ignores
    // every click.
    //
    // Polling sidesteps the registration conflict entirely and needs no message pump, which matters
    // because the only thread that reliably runs for such a host is the render thread. Must be called
    // between ImGui_ImplWin32_NewFrame and ImGui::NewFrame so its events are the last word.
    // No-op unless enableDevMenuOverlay has been called, so games that present normally are unaffected.
    // Implementation in rtx_fork_overlay.cpp.
    void pollDevMenuMouse();

    // What became of one external-API draw by the time the runtime had finished with it.
    //
    // Every field is something a host cannot otherwise observe. DrawInstance returns success the moment
    // the work is queued, and everything that can go on to make the draw invisible -- an ignored
    // material, an empty mesh, a zeroed instance mask, a hidden instance, an alpha test that rejects
    // every hit -- happens afterwards on the render thread. Without this a host sees healthy submit
    // counts and an empty frame, with no way to tell "never submitted" from "submitted and discarded"
    // from "in the acceleration structure but unhittable".
    struct ExternalDrawReport {
      bool     accepted;         // processDrawCallState returned an instance
      uint64_t meshHash;
      uint32_t vertexCount;      // as registered, not as submitted -- zero here means CreateMesh lost it
      uint32_t indexCount;
      float    worldPos[3];      // objectToWorld translation as the runtime reads it
      uint32_t instanceMask;     // VkAccelerationStructureInstanceKHR::mask; zero is unhittable
      bool     hidden;
      bool     fullyOpaque;
      uint32_t alphaTestType;    // AlphaTestType; kNever (0) rejects every hit
      uint32_t tlasSurfaceCount;
    };

    // Reports an ExternalDrawReport on a geometric schedule -- the first draw, then the tenth,
    // hundredth and so on. Dense enough to catch a problem that only appears once the world is up, and
    // free once the counts are large, which matters at several hundred draws a frame.
    //
    // Inline here rather than in a rtx_fork_*.cpp, against this header's usual convention, because the
    // caller is rtx_scene_manager.cpp: an out-of-line definition puts the symbol in a fork TU that also
    // holds the API entry points, and referencing it from the scene manager makes the linker pull that
    // object -- and its D3D9DeviceEx dependencies -- into the unit-test targets, which do not link the
    // d3d9 module. This function needs nothing but logging, so there is no reason for it to create that
    // edge. Function-local statics in an inline function are one object across the program, so the
    // counters stay global.
    inline void noteExternalDraw(const ExternalDrawReport& report) {
      static std::atomic<uint64_t> s_total { 0 };
      static std::atomic<uint64_t> s_rejected { 0 };
      static std::atomic<uint64_t> s_unhittable { 0 };
      static std::atomic<uint64_t> s_blended { 0 };
      static std::atomic<uint64_t> s_alphaTested { 0 };
      static std::atomic<uint64_t> s_nextReportAt { 1 };

      const uint64_t total = s_total.fetch_add(1, std::memory_order_relaxed) + 1;
      if (!report.accepted) {
        s_rejected.fetch_add(1, std::memory_order_relaxed);
      } else if (report.instanceMask == 0) {
        s_unhittable.fetch_add(1, std::memory_order_relaxed);
      }

      // Opacity split, as a running total rather than a single sample.
      //
      // A surface that is not fully opaque cannot take the fast closest-hit path: every ray intersection
      // runs an anyhit shader and samples the opacity channel. On geometry that is genuinely opaque that
      // cost buys nothing, and it is paid per intersection, so it scales with resolution -- which is the
      // shape of the cost actually measured here.
      //
      // calculateAlphaState derives it as !blendEnabled && alphaTestType == kAlways, so the two reasons a
      // surface loses opacity are distinguishable and want separate counters. They point at different
      // code: blending is the material's blend type, alpha testing is its threshold. One sample cannot
      // tell them apart from a surface that is legitimately transparent, and a single early draw is not
      // representative of a populated frame.
      if (report.accepted) {
        // AlphaTestType::kAlways, spelled as its value rather than the enum on purpose: the enum lives in
        // surface_shared.h, which this header does not include, and the note above on staying out of
        // rtx_fork_*.cpp is about not adding dependency edges here. kAlways means every fragment passes,
        // which is how "no alpha testing" is expressed.
        constexpr uint32_t kAlphaTestAlways = 7;
        const bool alphaTested = report.alphaTestType != kAlphaTestAlways;
        if (alphaTested) {
          s_alphaTested.fetch_add(1, std::memory_order_relaxed);
        }
        if (!report.fullyOpaque && !alphaTested) {
          // Opacity lost to blending alone, which is the case a mesh submitted through the API should
          // almost never be in unless it really is translucent.
          s_blended.fetch_add(1, std::memory_order_relaxed);
        }
      }

      uint64_t due = s_nextReportAt.load(std::memory_order_relaxed);
      if (total < due) {
        return;
      }
      // Claim the milestone before logging, so concurrent callers do not all report the same one.
      if (!s_nextReportAt.compare_exchange_strong(due, due * 10, std::memory_order_relaxed)) {
        return;
      }

      Logger::info(str::format(
        "[RTX-ExternalDraw] draw ", total,
        ": mesh 0x", std::hex, report.meshHash, std::dec,
        " verts ", report.vertexCount, " indices ", report.indexCount,
        " at ", report.worldPos[0], ", ", report.worldPos[1], ", ", report.worldPos[2],
        " | accepted ", report.accepted ? "yes" : "NO",
        " mask 0x", std::hex, report.instanceMask, std::dec,
        " hidden ", report.hidden ? "YES" : "no",
        " fullyOpaque ", report.fullyOpaque ? "yes" : "NO",
        " alphaTest ", report.alphaTestType,
        " | tlas surfaces ", report.tlasSurfaceCount,
        " | running: ", s_rejected.load(std::memory_order_relaxed), " rejected, ",
        s_unhittable.load(std::memory_order_relaxed), " with a zero instance mask, ",
        s_blended.load(std::memory_order_relaxed), " non-opaque from blending alone, ",
        s_alphaTested.load(std::memory_order_relaxed), " alpha tested"));
    }

    // Rasterises the developer menu into rtOutput.m_finalOutput from inside the injectRTX chain.
    //
    // Needed because the normal overlay draw lives in D3D9SwapChainEx::PresentImage and targets the
    // WSI swapchain image, which puts it out of reach of a host that reads m_finalOutput instead of
    // presenting -- and entirely undrawn for a host whose presenter never runs at all.
    //
    // No-op until enableDevMenuOverlay has been called, so games that present normally are
    // unaffected and never get a second ImGui frame.
    // Uses only public DxvkContext API, so no friend declaration is required.
    // Implementation in rtx_fork_overlay.cpp.
    void dispatchDevMenuOverlay(RtxContext& ctx, Resources::RaytracingOutput& rtOutput);

    // Stub: the DX11 shared-memory export backbuffer path is not ported to this
    // fork. Validates arguments then returns GENERAL_FAILURE so callers fall back;
    // the vtable slot is populated so the struct layout matches the plugin ABI.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode getSharedD3D11TextureHandle(
      D3D9DeviceEx* remixDevice,
      void**       out_sharedHandle,
      uint32_t*    out_width,
      uint32_t*    out_height);

    // Retrieves the D3D9CommonTexture from a D3D9 texture pointer, gets the
    // underlying DxvkImage, and returns image->getHash() in out_hash.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode dxvkGetTextureHash(
      IDirect3DTexture9* texture,
      uint64_t*          out_hash);

    // Creates a DxvkImage + view + staging buffer for the supplied pixel data,
    // copies data into staging, schedules an EmitCs lambda that transitions the
    // image, uploads all mip levels, transitions to shader-read, and registers
    // the texture with the texture manager and ImGui catalog.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode createTexture(
      D3D9DeviceEx*            remixDevice,
      const remixapi_TextureInfo* info,
      remixapi_TextureHandle*  out_handle);

    // Schedules an EmitCs lambda that searches the texture table by hash and
    // releases the texture reference via RtxTextureManager::releaseTexture.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode destroyTexture(
      D3D9DeviceEx*        remixDevice,
      remixapi_TextureHandle handle);

    // Fires the beginScene callback on the first submission of the frame
    // (whichever API call establishes s_inFrame first). Atomically exchanges
    // s_inFrame to true; calls s_beginCallback once per frame when it was false.
    // Called from remixapi_DrawInstance and remixapi_DrawLightInstance.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    void notifyBeginScene();

    // Stores the three per-frame bridge callbacks. Called from the
    // remixapi_RegisterCallbacks one-liner delegate in rtx_remix_api.cpp.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    void registerCallbacks(
      PFN_remixapi_BridgeCallback beginSceneCallback,
      PFN_remixapi_BridgeCallback endSceneCallback,
      PFN_remixapi_BridgeCallback presentCallback);

    // Clears all four frame-boundary state vars (s_inFrame, s_beginCallback,
    // s_endCallback, s_presentCallback) to their null/false defaults. Called
    // from remixapi_Shutdown.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    void shutdownCallbacks();

    // Fires the endScene callback if s_inFrame is set (i.e. the frame was
    // started via DrawInstance or DrawLightInstance). Called from
    // remixapi_Present immediately before the native remixDevice->Present()
    // so the endScene callback fires before GPU presentation.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    void presentEndSceneDispatch();

    // Fires the present callback and resets s_inFrame to false. Called from
    // remixapi_Present immediately after the native remixDevice->Present()
    // returns successfully. Separated from presentEndSceneDispatch because the
    // native Present call sits between the two callback fires.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    void presentCallbackDispatch();

    // Populates the three fork-added extern-C-linked vtable slots
    // (RegisterCallbacks, AutoInstancePersistentLights, UpdateLightDefinition)
    // in the remixapi_Interface vtable. Called from remixapi_InitializeLibrary
    // after the upstream and anonymous-namespace fork slots are filled inline.
    // Anonymous-namespace fork additions (AddTextureHash, CreateTexture,
    // GetUIState, CreateLightBatched, etc.) are assigned inline in upstream
    // where their symbols are visible — only externally linked symbols can be
    // named from this translation unit.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    void remixApiVtableInit(remixapi_Interface& interf);

    // Sets SceneManager's atomic VRAM-compaction request flag; the render
    // thread consumes it in manageTextureVram on the next tick. Returns
    // REMIX_DEVICE_WAS_NOT_REGISTERED if remixDevice is null. Lock-free;
    // callers need not hold the remix-api static mutex.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode requestVramCompaction(D3D9DeviceEx* remixDevice);

    // Sets SceneManager's atomic texture-VRAM-free request flag; the render
    // thread consumes it in manageTextureVram on the next tick (which then
    // calls textureManager.clear()). Returns REMIX_DEVICE_WAS_NOT_REGISTERED
    // if remixDevice is null. Lock-free; callers need not hold the remix-api
    // static mutex.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode requestTextureVramFree(D3D9DeviceEx* remixDevice);

    // Fills out_stats with DXVK per-category totals, driver-view heap info
    // (matches Task Manager / nvidia-smi — gap vs totalAllocatedBytes exposes
    // non-DXVK allocations such as NGX, RT pipeline state, descriptor pools,
    // NRC, etc.), and the fork-side texture manager's table size. Returns
    // INVALID_ARGUMENTS if out_stats is null or REMIX_DEVICE_WAS_NOT_REGISTERED
    // if remixDevice is null.
    // No private-member access; no friend declaration needed.
    // Implementation in rtx_fork_api_entry.cpp.
    remixapi_ErrorCode getVramStats(
      D3D9DeviceEx*        remixDevice,
      remixapi_VramStats*  out_stats);

    // Populates the tonemap-operator-related fields of the global tonemapper's
    // shader args struct (tonemapOperator + per-operator param blocks for
    // Hable, AgX, Lottes, Psycho17). Called from
    // DxvkToneMapping::dispatchApplyToneMapping.
    // No private-member access — uses public RtxOption accessors only.
    // Implementation in rtx_fork_tonemap.cpp.
    void populateTonemapOperatorArgs(ToneMappingApplyToneMappingArgs& args);

    // Renders the Tonemapping Operator combo + per-operator parameter sliders
    // inside DxvkToneMapping::showImguiSettings. Called from the fork-hook call
    // site replacing the old "Finalize With ACES" checkbox.
    // No private-member access — uses only public RtxOption / RemixGui / ImGui APIs.
    // Implementation in rtx_fork_tonemap.cpp.
    void showTonemapOperatorUI();

    // Renders the weather preset UI inside the existing atmosphere ImGui tree.
    // Includes preset dropdown, blend duration slider, current/target/progress
    // display, "Pause Weather Blender" toggle, and per-preset slider sub-trees.
    // No private-member access. (Wired in Task 4.)
    // Implementation in rtx_fork_weather.cpp.
    void showWeatherUI();

    // --- FSR 3.1 upscaling, RCAS sharpening and frame generation -------------
    //
    // The FidelityFX runtime is delay-loaded (see the /DELAYLOAD link arg in
    // src/dxvk/meson.build), so every one of these hooks is safe to call on a
    // machine with no amd_fidelityfx_vk.dll: they degrade to no-ops / "not
    // supported" rather than raising a delay-load exception.

    // True when amd_fidelityfx_vk.dll could be loaded in this process. Probed
    // once and latched. Every FFX entry point in the fork sits behind this.
    // Implementation in rtx_fork_fsr.cpp.
    bool fsrRuntimeAvailable();

    // True when FSR is the selected upscaler *and* its pass is active this
    // frame. Called from RtxContext::getCurrentUpscaler.
    // Requires friend access to RtxContext::m_common.
    // Implementation in rtx_fork_fsr.cpp.
    bool isFsrUpscalerActive(RtxContext& ctx);

    // Fills downscaleExtent with the render resolution FSR wants for the given
    // display resolution. Called from RtxContext::setDownscaleExtent.
    // Requires friend access to RtxContext::m_common.
    // Implementation in rtx_fork_fsr.cpp.
    void setFsrDownscaleExtent(RtxContext& ctx, const VkExtent3D& upscaleExtent, VkExtent3D& downscaleExtent);

    // Runs the FSR upscale pass. Called from RtxContext::dispatchUpscale in
    // place of the per-upscaler dispatch bodies.
    // Requires friend access to RtxContext private members.
    // Implementation in rtx_fork_fsr.cpp.
    void dispatchFsrUpscale(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput);

    // Runs the standalone RCAS sharpening pass after the main upscale. No-ops
    // unless sharpening is non-zero and the active upscaler is one that does
    // not sharpen internally (DLSS / DLSS-RR / XeSS / TAA-U).
    // Requires friend access to RtxContext::m_currentUpscaler.
    // Implementation in rtx_fork_rcas.cpp.
    void dispatchRcasSharpening(RtxContext& ctx, const Resources::RaytracingOutput& rtOutput);

    // Configures and prepares FSR frame generation for the current frame.
    // No-ops unless FSR frame generation is the selected, enabled and supported
    // backend. Called from the same site as RtxContext::dispatchDLFG.
    // Requires friend access to RtxContext private members.
    // Implementation in rtx_fork_fsr_framegen.cpp.
    void dispatchFsrFrameGeneration(RtxContext& ctx, const Rc<DxvkImage>& hudLessBackBuffer);

    // Texture mip-bias contribution when FSR is the active upscaler; returns 0
    // when it is not. Called from Resources::getSampler and
    // SceneManager::getTotalMipBias / getCalculatedUpscalingMipBias.
    // No private-member access.
    // Implementation in rtx_fork_fsr.cpp.
    float fsrUpscalingMipBias(DxvkDevice* device);

    // Camera jitter sequence length FSR expects, matching
    // ffxFsr3UpscalerGetJitterPhaseCount. Returns 0 when FSR is not the active
    // upscaler, in which case the caller keeps its own sequence length.
    // No private-member access.
    // Implementation in rtx_fork_fsr.cpp.
    uint32_t fsrJitterSequenceLength(uint32_t finalWidth, uint32_t renderWidth);

    // True when FSR frame generation is selected, enabled and supported on this
    // device. The DXVK-side mirror of DxvkContext::isDLFGEnabled; consumed by
    // D3D9SwapChainEx to decide which presenter to create.
    // No private-member access.
    // Implementation in rtx_fork_fsr_framegen.cpp.
    bool isFsrFrameGenEnabled(DxvkDevice* device);

    // True when either frame-generation backend is currently enabled. Used by
    // the dev menu's V-Sync guard, which must disable the V-Sync widgets for
    // FSR FG as well as DLSS-G.
    // Implementation in rtx_fork_upscaler_ui.cpp.
    bool anyFrameGenerationEnabled();

    // True when this device supports at least one frame-generation backend.
    // Gates whether the menus show a "Frame Generation Settings" section.
    // Implementation in rtx_fork_upscaler_ui.cpp.
    bool anyFrameGenerationSupported(const Rc<DxvkContext>& ctx, bool isDlfgSupported);

    // Draws the whole frame-generation panel: the Off / DLSS / FSR technology
    // combo (which is itself the enable control - see
    // fork_hooks::applyFrameGenerationType), a status line, and whatever extra
    // controls the selected backend has. Replaces the call to the upstream
    // ImGUI::showDLFGOptions, which is left untouched but unused because its
    // body is built around a per-backend enable checkbox this UI does not have.
    // Implementation in rtx_fork_upscaler_ui.cpp.
    void showFrameGenerationOptions(const Rc<DxvkContext>& ctx, bool isDlfgSupported);

    // Draws the FSR upscaler sub-panel: preset combo plus render-resolution
    // readout. Called from the upscaler-settings switch in both menus.
    // Implementation in rtx_fork_upscaler_ui.cpp.
    void showFsrUpscalerSettings(const Rc<DxvkContext>& ctx);

    // Draws the shared post-upscale RCAS sharpness slider. Shown for any active
    // upscaler; the option itself lives at rtx.sharpening.sharpness.
    // Implementation in rtx_fork_upscaler_ui.cpp.
    void showSharedSharpnessSlider();
    // Submits the weather precipitation emitter (rain / snow / blowing sand) for
    // this frame. Called from RtxContext::injectRTX immediately BEFORE
    // SceneManager::prepareSceneData, which is where the particle simulation
    // runs — submitting after it would lose this frame's spawn contexts.
    // Dormant unless rtx.weather.precipitation.intensity is non-zero, which the
    // weather blender drives from the per-preset precipitation fields.
    // No private-member access. Implementation in rtx_fork_precipitation.cpp.
    void submitPrecipitation(class RtxContext& ctx);

    // Renders the global (non-per-preset) precipitation controls — budget, spawn
    // volume, collision. The per-preset look values are generated into the
    // weather preset editor automatically by WEATHER_PRESET_FIELD_LIST.
    // No private-member access. Implementation in rtx_fork_precipitation.cpp.
    void showPrecipitationUI();

  } // namespace fork_hooks

} // namespace dxvk
