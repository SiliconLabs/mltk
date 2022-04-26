
#ifndef SL_POWER_MANAGER_H
#define SL_POWER_MANAGER_H



#include <stdbool.h>
#include <stdint.h>

#include "sl_status.h"

#ifdef __cplusplus
extern "C" {
#endif



// -----------------------------------------------------------------------------
// Data Types

/// @brief Energy modes
typedef  enum  {
  SL_POWER_MANAGER_EM0 = 0,   ///< Run Mode (Energy Mode 0)
  SL_POWER_MANAGER_EM1,       ///< Sleep Mode (Energy Mode 1)
  SL_POWER_MANAGER_EM2,       ///< Deep Sleep Mode (Energy Mode 2)
  SL_POWER_MANAGER_EM3,       ///< Stop Mode (Energy Mode 3)
  SL_POWER_MANAGER_EM4,       ///< Shutoff Mode (Energy Mode 4)
} sl_power_manager_em_t;



// -----------------------------------------------------------------------------
// Internal Prototypes only to be used by Power Manager module
static inline void sli_power_manager_update_em_requirement(sl_power_manager_em_t em,
                                             bool  add)
{
    
}

static inline void sl_power_manager_sleep(void)
{

}


static inline void sl_power_manager_add_em_requirement(sl_power_manager_em_t em)
{
}

static inline void sl_power_manager_remove_em_requirement(sl_power_manager_em_t em)
{
}


#ifdef __cplusplus
}
#endif

#endif // SL_POWER_MANAGER_H
