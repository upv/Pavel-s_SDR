# OAI nrUE (SRK image) with MicroPhase antsdr_uhd for ANTSDR E316 (UHD "ant" device)
FROM oai-nr-ue-cuda:latest
COPY antsdr-lib/ /opt/antsdr-uhd/lib/
COPY liboai_usrpdevif.so /usr/local/lib/liboai_usrpdevif.so
ENV LD_LIBRARY_PATH=/opt/antsdr-uhd/lib
