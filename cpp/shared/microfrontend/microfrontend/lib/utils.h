#pragma once


#if defined(MICROFRONTEND_DLL_EXPORT) && !defined(DLL_EXPORT)
#  if defined(WIN32) || defined(_WIN32)
#    define DLL_EXPORT __declspec(dllexport)
#  else
#    define DLL_EXPORT __attribute__ ((visibility ("default")))
#  endif
#endif 

#ifndef DLL_EXPORT
#define DLL_EXPORT
#endif
