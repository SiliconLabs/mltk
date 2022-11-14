import os 
import gzip 
import shutil

from .path import create_tempdir


def upload_file(
    src_path: str, 
    api_token:str, 
    dst_path: str = None, 
    dst_subdir:str = None,
    compress=False
) -> str:
    """Upload a file to dropbox and return a downloadable link to it

    This requires a Dropbox "App", for more details see:
    https://www.dropbox.com/developers/reference/getting-started#overview
    
    Args:
        src_path: File path of source file
        api_token: Dropbox API token
        dst_path: File path on dropbox (relative to App folder), if omitted use basename of src_path
        dst_subdir: Subdirectory of file path on dropbox (relative to App folder), only used if dst_path is None
        compress: GZIP compress the source file before uploading, dst file automatically gets a .gz extension
    Returns:
        Public download link to file on Dropbox
    """
    try:
        import dropbox
    except Exception:
        raise RuntimeError('Failed import dropbox Python package, try running: pip install dropbox')

    d = dropbox.Dropbox(api_token)

    if not dst_path:
        dst_subdir = '' if dst_subdir is None else f'{dst_subdir}/'
        dst_path = dst_subdir + os.path.basename(src_path)

    dst_path = f'/{dst_path}'

    if compress:
        with open(src_path, 'rb') as orig_file:
            name = os.path.basename(dst_path)
            tmp_path = f'{create_tempdir("uploads")}/{name}.gz'
            with gzip.open(tmp_path, 'wb') as zipped_file:
                shutil.copyfileobj(orig_file, zipped_file)
            src_path = tmp_path
            dst_path += '.gz'


    with open(src_path, 'rb') as f:
        # upload gives you metadata about the file
        # we want to overwite any previous version of the file
        d.files_upload(f.read(), dst_path, mode=dropbox.files.WriteMode("overwrite"))

    link = d.sharing_create_shared_link(dst_path)

    return link.url.replace('?dl=0', '?dl=1')