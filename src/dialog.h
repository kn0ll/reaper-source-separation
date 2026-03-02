#ifndef DIALOG_H
#define DIALOG_H

#include "separator.h"

namespace dialog {

void init(HWND main_hwnd, HINSTANCE hInst);
void open(const SeparationRequest& req);
void close();

} // namespace dialog

#endif
