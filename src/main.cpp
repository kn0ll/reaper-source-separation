#define REAPERAPI_MINIMAL
#define REAPERAPI_WANT_plugin_register
#define REAPERAPI_WANT_ShowMessageBox
#define REAPERAPI_WANT_GetResourcePath
#define REAPERAPI_WANT_GetExePath
#define REAPERAPI_WANT_CountSelectedMediaItems
#define REAPERAPI_WANT_GetSelectedMediaItem
#define REAPERAPI_WANT_GetActiveTake
#define REAPERAPI_WANT_GetMediaItemTake_Source
#define REAPERAPI_WANT_GetMediaItemTake_Track
#define REAPERAPI_WANT_GetMediaSourceFileName
#define REAPERAPI_WANT_GetMediaItemInfo_Value
#define REAPERAPI_WANT_GetMediaTrackInfo_Value
#define REAPERAPI_WANT_InsertTrackAtIndex
#define REAPERAPI_WANT_GetTrack
#define REAPERAPI_WANT_GetNumTracks
#define REAPERAPI_WANT_GetSetMediaTrackInfo_String
#define REAPERAPI_WANT_GetTrackColor
#define REAPERAPI_WANT_SetTrackColor
#define REAPERAPI_WANT_AddMediaItemToTrack
#define REAPERAPI_WANT_SetMediaItemInfo_Value
#define REAPERAPI_WANT_SetMediaTrackInfo_Value
#define REAPERAPI_WANT_AddTakeToMediaItem
#define REAPERAPI_WANT_PCM_Source_CreateFromFile
#define REAPERAPI_WANT_SetMediaItemTake_Source
#define REAPERAPI_WANT_Undo_BeginBlock
#define REAPERAPI_WANT_Undo_EndBlock
#define REAPERAPI_WANT_UpdateArrange
#define REAPERAPI_WANT_TrackList_AdjustWindows

#define REAPERAPI_IMPLEMENT
#include "reaper_plugin.h"
#include "reaper_plugin_functions.h"

#include "actions.h"
#include "dialog.h"
#include "separator.h"
#include <cstdio>

static HINSTANCE g_hInst;

extern "C" REAPER_PLUGIN_DLL_EXPORT int REAPER_PLUGIN_ENTRYPOINT(
    REAPER_PLUGIN_HINSTANCE hInst, reaper_plugin_info_t* rec)
{
    if (!rec) {
        fprintf(stderr, "[reaper-source-separation] unloading\n");
        actions::unregister_all();
        dialog::close();
        separator::cleanup_model();
        separator::cleanup_temp_files();
        return 0;
    }

    fprintf(stderr, "[reaper-source-separation] loading, caller_version=0x%x, expected=0x%x\n",
            rec->caller_version, REAPER_PLUGIN_VERSION);

    if (rec->caller_version != REAPER_PLUGIN_VERSION) {
        fprintf(stderr, "[reaper-source-separation] version mismatch, aborting\n");
        return 0;
    }

    g_hInst = (HINSTANCE)hInst;

    int api_err = REAPERAPI_LoadAPI(rec->GetFunc);
    if (api_err != 0) {
        fprintf(stderr, "[reaper-source-separation] REAPERAPI_LoadAPI failed (%d functions missing)\n", api_err);
        return 0;
    }

    separator::cleanup_temp_files();
    dialog::init(rec->hwnd_main, g_hInst);

    if (!actions::register_all(rec)) {
        fprintf(stderr, "[reaper-source-separation] action registration failed\n");
        return 0;
    }

    fprintf(stderr, "[reaper-source-separation] loaded OK, command_id=%d\n", actions::command_id());
    return 1;
}
