"""NIfTI reading from plain folders OR directly from a .zip archive (no extraction needed).

A file inside a zip is addressed as  '/path/to/archive.zip::member/name.nii.gz'.
This keeps the BraTS training zip (13 GB) usable on accounts with a small disk quota.
"""
import gzip
import os
import threading
import zipfile

import nibabel as nib

SEP = "::"
_ZIPS = {}  # one open handle per process AND thread (a shared handle corrupts parallel reads)


def _zip(path):
    key = (os.getpid(), threading.get_ident(), path)
    if key not in _ZIPS:
        _ZIPS[key] = zipfile.ZipFile(path)
    return _ZIPS[key]


def is_zip_path(p):
    return isinstance(p, str) and SEP in p


def read_nifti(path):
    """Return a nibabel image for a normal path or a 'archive.zip::member' path."""
    if not is_zip_path(path):
        return nib.load(path)
    zpath, member = path.split(SEP, 1)
    data = _zip(zpath).read(member)
    if member.endswith(".gz"):
        data = gzip.decompress(data)
    return nib.Nifti1Image.from_bytes(data)


def zip_members(zpath):
    return [n for n in _zip(zpath).namelist()
            if n.endswith((".nii", ".nii.gz")) and "__MACOSX" not in n and not os.path.basename(n).startswith("._")]
