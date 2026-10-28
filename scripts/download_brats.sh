#!/usr/bin/env bash
# Download a BraTS file from Synapse straight onto the server (kept as .zip: the code reads
# cases straight from the zip, which fits a 20 GB home quota). Add 'unzip' as 2nd arg to extract.
# Uses the plain python:3.12 image so synapseclient's dependencies never touch the PyTorch setup.
# Usage (on the server):  bash scripts/download_brats.sh syn12345678
#   syn12345678 = the Synapse ID of the TrainingData zip (click the file name on Synapse to see it).
# You will be asked for a Synapse Personal Access Token (Account Settings -> Personal Access Tokens,
# with View + Download). The token is typed in hidden and is not saved anywhere.
set -e
SYNID=$1
UNZIP=${2:-no}
[ -z "$SYNID" ] && { echo "usage: bash scripts/download_brats.sh <synapse_id>"; exit 1; }
DATA=${BRATS_DATA_DIR:-$HOME/data}
mkdir -p "$DATA"
read -r -s -p "Synapse Personal Access Token: " TOKEN; echo
docker run --rm -it --user "$(id -u):$(id -g)" -e HOME=/tmp -e SYNAPSE_AUTH_TOKEN="$TOKEN" \
  -v "$DATA:$DATA" -w "$DATA" python:3.12-bookworm bash -c "
    pip install -q --user synapseclient &&
    python -c \"
import os, synapseclient, zipfile
syn = synapseclient.login(authToken=os.environ['SYNAPSE_AUTH_TOKEN'], silent=True)
f = syn.get('$SYNID', downloadLocation='$DATA')
print('downloaded:', f.path)
if f.path.endswith('.zip') and '$UNZIP' == 'unzip':
    print('unzipping ...'); zipfile.ZipFile(f.path).extractall('$DATA'); os.remove(f.path)
    print('unzipped and removed the zip')
\""
ls "$DATA"
du -sh "$DATA"
