#ifndef ACTIONS_H
#define ACTIONS_H

#include "reaper_plugin.h"

namespace actions {

bool register_all(reaper_plugin_info_t* rec);
void unregister_all();

int command_id();

} // namespace actions

#endif
