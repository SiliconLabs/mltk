"""Extends standard ZipFile class to support extracting to paths > 260 chars on Windows"""
import os
import zipfile


        
class ZipFile(zipfile.ZipFile):
    """Simple class that wraps the standard ZipFile class to enable file paths > 260 on windows"""
    def _extract_member(self, member, targetpath, pwd):
        targetpath = targetpath if os.name != 'nt' else winapi_path(targetpath)
        return zipfile.ZipFile._extract_member(self, member, targetpath, pwd)


def winapi_path(dos_path, encoding=None):
    path = os.path.abspath(dos_path)

    if path.startswith("\\\\"):
        path = "\\\\?\\UNC\\" + path[2:]
    else:
        path = "\\\\?\\" + path 

    return path