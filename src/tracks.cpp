#define REAPERAPI_MINIMAL
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
#include "reaper_plugin.h"
#include "reaper_plugin_functions.h"

#include "tracks.h"
#include "log.h"
#include <cctype>
#include <filesystem>

namespace fs = std::filesystem;

static std::string capitalize(const std::string& s) {
    if (s.empty()) return s;
    std::string out = s;
    out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
    return out;
}

void tracks::create_from_result(const SeparationResult& result) {
    Undo_BeginBlock();

    int source_color = 0;
    MediaTrack* source_track = GetTrack(nullptr, result.request.track_index);
    if (source_track)
        source_color = GetTrackColor(source_track);

    std::string parent_name = fs::path(result.request.source_path).stem().string();
    int insert_idx = result.request.track_index + 1;

    // Create folder parent track
    InsertTrackAtIndex(insert_idx, false);
    MediaTrack* folder = GetTrack(nullptr, insert_idx);
    if (folder) {
        std::string folder_name = parent_name + " - Stems";
        GetSetMediaTrackInfo_String(folder, "P_NAME",
            const_cast<char*>(folder_name.c_str()), true);
        SetMediaTrackInfo_Value(folder, "I_FOLDERDEPTH", 1);
        if (source_color)
            SetTrackColor(folder, source_color);
    }
    insert_idx++;

    // Create child stem tracks
    for (size_t i = 0; i < result.stems.size(); ++i) {
        const auto& stem = result.stems[i];

        InsertTrackAtIndex(insert_idx, false);
        MediaTrack* tr = GetTrack(nullptr, insert_idx);
        if (!tr) { insert_idx++; continue; }

        std::string track_name = capitalize(stem.name);
        GetSetMediaTrackInfo_String(tr, "P_NAME",
            const_cast<char*>(track_name.c_str()), true);

        if (source_color)
            SetTrackColor(tr, source_color);

        // Close the folder on the last stem
        if (i == result.stems.size() - 1)
            SetMediaTrackInfo_Value(tr, "I_FOLDERDEPTH", -1);

        MediaItem* item = AddMediaItemToTrack(tr);
        if (!item) { insert_idx++; continue; }

        SetMediaItemInfo_Value(item, "D_POSITION", result.request.item_position);
        SetMediaItemInfo_Value(item, "D_LENGTH", result.request.item_length);

        MediaItem_Take* take = AddTakeToMediaItem(item);
        if (!take) { insert_idx++; continue; }

        PCM_source* src = PCM_Source_CreateFromFile(stem.path.c_str());
        if (src) {
            SetMediaItemTake_Source(take, src);
        }

        insert_idx++;
    }

    TrackList_AdjustWindows(false);
    UpdateArrange();
    Undo_EndBlock("Separate stems", UNDO_STATE_ALL);
    LOG("tracks created: stems=%zu insert_after=%d\n",
        result.stems.size(), result.request.track_index);
}
