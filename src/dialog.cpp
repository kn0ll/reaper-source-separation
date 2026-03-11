#define REAPERAPI_MINIMAL
#define REAPERAPI_WANT_plugin_register
#define REAPERAPI_WANT_ShowMessageBox
#include "reaper_plugin.h"
#include "reaper_plugin_functions.h"

#ifdef _WIN32
#include <commctrl.h>
#endif

#include "dialog.h"
#include "model_manager.h"
#include "resource.h"
#include "tracks.h"
#include <algorithm>
#include <filesystem>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>

namespace fs = std::filesystem;

static HWND g_main_hwnd = nullptr;
static HINSTANCE g_hInst = nullptr;
static HWND g_dialog = nullptr;
static SeparationRequest g_request;

enum class DialogMode { Idle, Downloading, Separating };
static DialogMode g_mode = DialogMode::Idle;
static bool g_cancelling = false;
static std::string g_pending_model_id;

static void populate_models(HWND combo) {
    SendMessage(combo, CB_RESETCONTENT, 0, 0);
    auto& models = model_manager::available_models();
    for (auto& m : models) {
        SendMessage(combo, CB_ADDSTRING, 0, (LPARAM)m.display_name.c_str());
    }
    if (models.empty())
        SendMessage(combo, CB_ADDSTRING, 0, (LPARAM)"(no models defined)");
    SendMessage(combo, CB_SETCURSEL, 0, 0);
}

static const model_manager::ModelInfo* get_selected_model(HWND hwnd) {
    int sel = (int)SendDlgItemMessage(hwnd, IDC_MODEL, CB_GETCURSEL, 0, 0);
    auto& models = model_manager::available_models();
    if (sel < 0 || sel >= (int)models.size()) return nullptr;
    return &models[sel];
}

static void set_controls_enabled(HWND hwnd, bool idle) {
    EnableWindow(GetDlgItem(hwnd, IDC_SEPARATE), idle);
    EnableWindow(GetDlgItem(hwnd, IDC_MODEL), idle);
    EnableWindow(GetDlgItem(hwnd, IDC_CANCEL), !idle);
}

static void timer_callback();

static void reset_to_idle(HWND hwnd) {
    plugin_register("-timer", (void*)timer_callback);
    g_mode = DialogMode::Idle;
    g_cancelling = false;
    SendDlgItemMessage(hwnd, IDC_PROGRESS, PBM_SETPOS, 0, 0);
    SetDlgItemText(hwnd, IDC_STATUS, "");
    set_controls_enabled(hwnd, true);
}

static void update_ui(HWND hwnd) {
    if (g_mode == DialogMode::Downloading) {
        auto ds = model_manager::download_state();

        if (!g_cancelling) {
            float prog = model_manager::download_progress();
            int pct = (int)(prog * 100.0f);
            char buf[256];
            snprintf(buf, sizeof(buf), "[%d%%] Downloading model...", pct);
            SetDlgItemText(hwnd, IDC_STATUS, buf);
            SendDlgItemMessage(hwnd, IDC_PROGRESS, PBM_SETPOS, pct, 0);
        }

        if (ds == model_manager::DownloadState::Done) {
            model_manager::reset_download();
            g_cancelling = false;
            populate_models(GetDlgItem(hwnd, IDC_MODEL));
            auto& models = model_manager::available_models();
            for (int i = 0; i < (int)models.size(); ++i) {
                if (models[i].id == g_pending_model_id)
                    SendDlgItemMessage(hwnd, IDC_MODEL, CB_SETCURSEL, i, 0);
            }
            g_mode = DialogMode::Separating;
            std::string path = model_manager::model_path(g_pending_model_id);
            g_request.model_id = g_pending_model_id;
            g_request.model_path = path;
            separator::start(g_request);
        } else if (ds == model_manager::DownloadState::Idle && g_cancelling) {
            reset_to_idle(hwnd);
        } else if (ds == model_manager::DownloadState::Error) {
            std::string err = model_manager::download_error();
            model_manager::reset_download();
            reset_to_idle(hwnd);
            ShowMessageBox(err.c_str(), "Download Error", 0);
        }
        return;
    }

    if (g_mode == DialogMode::Separating) {
        auto st = separator::state();

        if (!g_cancelling) {
            float prog = separator::progress();
            std::string msg = separator::status_message();
            int pct = (int)(prog * 100.0f);
            char status_buf[512];
            snprintf(status_buf, sizeof(status_buf), "[%d%%] %s", pct, msg.c_str());
            SetDlgItemText(hwnd, IDC_STATUS, status_buf);
            SendDlgItemMessage(hwnd, IDC_PROGRESS, PBM_SETPOS, pct, 0);
        }

        if (st == separator::State::Done) {
            plugin_register("-timer", (void*)timer_callback);
            SeparationResult res = separator::result();
            separator::reset();
            g_mode = DialogMode::Idle;
            g_cancelling = false;
            tracks::create_from_result(res);
            dialog::close();
        } else if (st == separator::State::Error) {
            std::string err = separator::error_message();
            separator::reset();
            reset_to_idle(hwnd);
            if (err != "Cancelled") {
                ShowMessageBox(err.c_str(), "Separation Error", 0);
            }
        }
    }
}

static void timer_callback() {
    if (g_dialog) update_ui(g_dialog);
}

static void start_separation(HWND hwnd) {
    auto* info = get_selected_model(hwnd);
    if (!info) {
        ShowMessageBox("No valid model selected.", "Separate Stems", 0);
        return;
    }

    set_controls_enabled(hwnd, false);
    plugin_register("timer", (void*)timer_callback);

    if (!model_manager::is_available(info->id)) {
        g_mode = DialogMode::Downloading;
        g_pending_model_id = info->id;
        SetDlgItemText(hwnd, IDC_STATUS, "Starting download...");
        model_manager::start_download(info->id);
        return;
    }

    g_mode = DialogMode::Separating;
    g_request.model_id = info->id;
    g_request.model_path = model_manager::model_path(info->id);
    separator::start(g_request);
}

static INT_PTR WINAPI dialog_proc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM /*lParam*/) {
    switch (msg) {
    case WM_INITDIALOG:
        populate_models(GetDlgItem(hwnd, IDC_MODEL));
        SendDlgItemMessage(hwnd, IDC_PROGRESS, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
        SendDlgItemMessage(hwnd, IDC_PROGRESS, PBM_SETPOS, 0, 0);
        EnableWindow(GetDlgItem(hwnd, IDC_CANCEL), FALSE);
        return TRUE;

    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case IDC_SEPARATE:
            start_separation(hwnd);
            return TRUE;
        case IDC_CANCEL:
            if (g_mode == DialogMode::Downloading) {
                model_manager::cancel_download();
            } else if (g_mode == DialogMode::Separating) {
                separator::cancel();
            }
            g_cancelling = true;
            EnableWindow(GetDlgItem(hwnd, IDC_CANCEL), FALSE);
            SetDlgItemText(hwnd, IDC_STATUS, "Cancelling...");
            return TRUE;
        case IDCANCEL:
            if (g_mode == DialogMode::Downloading) {
                model_manager::cancel_download();
                plugin_register("-timer", (void*)timer_callback);
            } else if (g_mode == DialogMode::Separating) {
                separator::cancel();
                plugin_register("-timer", (void*)timer_callback);
            }
            g_mode = DialogMode::Idle;
            g_cancelling = false;
            dialog::close();
            return TRUE;
        }
        break;

    case WM_CLOSE:
        if (g_mode == DialogMode::Downloading) {
            model_manager::cancel_download();
            plugin_register("-timer", (void*)timer_callback);
        } else if (g_mode == DialogMode::Separating) {
            separator::cancel();
            plugin_register("-timer", (void*)timer_callback);
        }
        g_mode = DialogMode::Idle;
        g_cancelling = false;
        dialog::close();
        return TRUE;
    }
    return FALSE;
}

void dialog::init(HWND main_hwnd, HINSTANCE hInst) {
    g_main_hwnd = main_hwnd;
    g_hInst = hInst;
}

void dialog::open(const SeparationRequest& req) {
    if (g_dialog) {
        SetForegroundWindow(g_dialog);
        return;
    }

    g_request = req;
    g_mode = DialogMode::Idle;
    g_cancelling = false;
    separator::reset();
    model_manager::reset_download();

    g_dialog = CreateDialogParam(g_hInst, MAKEINTRESOURCE(IDD_SEPARATE),
                                 g_main_hwnd, dialog_proc, 0);
    if (g_dialog) {
        RECT pr, dr;
        GetWindowRect(g_main_hwnd, &pr);
        GetWindowRect(g_dialog, &dr);
        int x = pr.left + ((pr.right - pr.left) - (dr.right - dr.left)) / 2;
        int y = pr.top + ((pr.bottom - pr.top) - (dr.bottom - dr.top)) / 2;
        SetWindowPos(g_dialog, nullptr, x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER);
        ShowWindow(g_dialog, SW_SHOW);
    }
}

void dialog::close() {
    if (g_dialog) {
        DestroyWindow(g_dialog);
        g_dialog = nullptr;
    }
}
