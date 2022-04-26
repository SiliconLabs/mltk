#pragma once


#ifndef SL_MIN
#define SL_MIN(a, b) __extension__({ __typeof__(a)_a = (a); __typeof__(b)_b = (b); _a < _b ? _a : _b; })
#endif 
#ifndef SL_MAX
#define SL_MAX(a, b) __extension__({ __typeof__(a)_a = (a); __typeof__(b)_b = (b); _a > _b ? _a : _b; })
#endif 
#ifndef __STATIC_FORCEINLINE
#define __STATIC_FORCEINLINE static inline
#endif
#ifndef __INLINE
#define __INLINE inline
#endif