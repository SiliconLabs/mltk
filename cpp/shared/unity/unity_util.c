#include "unity.h"
#include "unity_internals.h"


extern struct UNITY_STORAGE_T Unity;


extern struct UNITY_STORAGE_T* UnityGet()
{
    return &Unity;
}