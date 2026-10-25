#!/bin/bash
# Build the patched OAI USRP module and the derived SRK images:
#   oai-nr-ue-antsdr   - OAI nrUE linked against MicroPhase antsdr_uhd (ANTSDR E316, UHD type "ant")
#   oai-gnb-cuda-txrx  - SRK gNB with the same usrp_lib patch (RX antenna override)
# Needs the SRK images ran-build-cuda, oai-nr-ue-cuda, oai-gnb-cuda (scripts/build-oai-images.sh)
# and antsdr_uhd installed to $ANTSDR_UHD (see README.md).
set -euo pipefail

HERE=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
OAI_DIR=${OAI_DIR:?set OAI_DIR to the SRK OpenAirInterface checkout (sionna-rk/ext/openairinterface5g)}
ANTSDR_UHD=${ANTSDR_UHD:-$HOME/opt/antsdr-uhd}
WORK=${WORK:-$HERE/build}

mkdir -p "$WORK/ue" "$WORK/gnb"
cp "$OAI_DIR/radio/USRP/usrp_lib.cpp" "$WORK/usrp_lib.cpp"
patch -d "$WORK" -p3 usrp_lib.cpp < "$HERE/patches/oai_usrp_lib_antsdr.patch"

build_lib() {  # $1: output dir, $2...: extra cmake args
    local out=$1; shift
    docker run --rm \
        -v "$ANTSDR_UHD":"$ANTSDR_UHD":ro \
        -v "$WORK/usrp_lib.cpp":/oai-ran/radio/USRP/usrp_lib.cpp:ro \
        -v "$out":/out \
        --entrypoint bash ran-build-cuda:latest -c \
        "cd /oai-ran/cmake_targets/ran_build/build && cmake $* . > /out/cmake.log && ninja oai_usrpdevif && cp liboai_usrpdevif.so /out/"
}

# UE: link against antsdr_uhd (UHD 4.1, soname libuhd.so.4.1.0)
build_lib "$WORK/ue" -DUHD_LIBRARIES="$ANTSDR_UHD/lib/libuhd.so" -DUHD_INCLUDE_DIRS="$ANTSDR_UHD/include"
rm -rf "$WORK/ue/antsdr-lib"
cp -a "$ANTSDR_UHD/lib" "$WORK/ue/antsdr-lib"
rm -rf "$WORK/ue/antsdr-lib/uhd/examples" "$WORK/ue/antsdr-lib/cmake" "$WORK/ue/antsdr-lib/pkgconfig"
cp "$HERE/docker/ue/Dockerfile" "$WORK/ue/"
docker build -t oai-nr-ue-antsdr:latest "$WORK/ue"

# gNB: same source, container UHD 4.8
build_lib "$WORK/gnb"
cp "$HERE/docker/gnb/Dockerfile" "$WORK/gnb/"
docker build -t oai-gnb-cuda-txrx:latest "$WORK/gnb"
