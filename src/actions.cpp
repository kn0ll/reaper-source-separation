#define REAPERAPI_MINIMAL
#define REAPERAPI_WANT_CountSelectedMediaItems
#define REAPERAPI_WANT_GetSelectedMediaItem
#define REAPERAPI_WANT_GetActiveTake
#define REAPERAPI_WANT_GetMediaItemTake_Source
#define REAPERAPI_WANT_GetMediaSourceFileName
#define REAPERAPI_WANT_GetMediaItemInfo_Value
#define REAPERAPI_WANT_ShowMessageBox
#define REAPERAPI_WANT_GetResourcePath
#define REAPERAPI_WANT_plugin_register
#define REAPERAPI_WANT_GetMediaItemTake_Track
#define REAPERAPI_WANT_GetMediaTrackInfo_Value
#include "reaper_plugin.h"
#include "reaper_plugin_functions.h"

#include "actions.h"
#include "dialog.h"
#include "model_manager.h"
#include "separator.h"
#include <cstring>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

static int g_cmd_id = 0;
static gaccel_register_t g_accel;

static std::string resolve_cache_dir() {
    const char* res = GetResourcePath();
    return (fs::path(res) / "UserPlugins" / "reaper-source-separation" / "models").string();
}

static std::string resolve_local_dir() {
    // Check next to plugin binary for a local models/ dir (dev workflow)
#ifndef _WIN32
    Dl_info info;
    if (dladdr((void*)&resolve_local_dir, &info) && info.dli_fname) {
        fs::path nearby = fs::path(info.dli_fname).parent_path() / "reaper-source-separation" / "models";
        if (fs::exists(nearby)) return nearby.string();
    }
#endif
    return {};
}

static int get_track_index_for_item(MediaItem* item) {
    MediaItem_Take* take = GetActiveTake(item);
    if (!take) return 0;
    MediaTrack* tr = (MediaTrack*)GetMediaItemTake_Track(take);
    if (!tr) return 0;
    return (int)GetMediaTrackInfo_Value(tr, "IP_TRACKNUMBER") - 1;
}

static void on_command() {
    int count = CountSelectedMediaItems(nullptr);
    if (count < 1) {
        ShowMessageBox("Select an audio item first.", "Separate Sources", 0);
        return;
    }

    MediaItem* item = GetSelectedMediaItem(nullptr, 0);
    if (!item) return;

    MediaItem_Take* take = GetActiveTake(item);
    if (!take) {
        ShowMessageBox("No active take in selected item.", "Separate Sources", 0);
        return;
    }

    PCM_source* source = GetMediaItemTake_Source(take);
    if (!source) {
        ShowMessageBox("No audio source in selected take.", "Separate Sources", 0);
        return;
    }

    char filename[4096] = {};
    GetMediaSourceFileName(source, filename, sizeof(filename));
    if (filename[0] == '\0') {
        ShowMessageBox("Selected item has no source file (e.g. MIDI).", "Separate Sources", 0);
        return;
    }

    SeparationRequest req;
    req.source_path = filename;
    req.item_position = GetMediaItemInfo_Value(item, "D_POSITION");
    req.item_length = GetMediaItemInfo_Value(item, "D_LENGTH");
    req.track_index = get_track_index_for_item(item);

    dialog::open(req);
}

static bool hook_command(int cmd, int flag) {
    if (cmd == g_cmd_id) {
        on_command();
        return true;
    }
    return false;
}

static void menu_hook(const char* menuidstr, HMENU menu, int flag) {
    fprintf(stderr, "[reaper-source-separation] menu_hook: menuidstr=\"%s\" flag=%d\n", menuidstr, flag);
    if (flag == 0) {
        InsertMenu(menu, GetMenuItemCount(menu), MF_BYPOSITION | MF_STRING, g_cmd_id, "Separate sources");
    }
}

bool actions::register_all(reaper_plugin_info_t* rec) {
    g_cmd_id = rec->Register("command_id", (void*)"ReaperDemucs_SeparateSources");
    if (!g_cmd_id) return false;

    g_accel.accel.cmd = g_cmd_id;
    g_accel.accel.fVirt = 0;
    g_accel.accel.key = 0;
    g_accel.desc = "Separate sources (Demucs)";
    rec->Register("gaccel", &g_accel);

    rec->Register("hookcommand", (void*)hook_command);
    rec->Register("hookcustommenu", (void*)menu_hook);

    model_manager::init(resolve_cache_dir(), resolve_local_dir());

    return true;
}

void actions::unregister_all() {
    plugin_register("-hookcommand", (void*)hook_command);
    plugin_register("-hookcustommenu", (void*)menu_hook);
}

int actions::command_id() {
    return g_cmd_id;
}
