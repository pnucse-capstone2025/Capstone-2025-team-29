FROM bluenviron/mediamtx:latest

COPY ./www /www

ENV MTX_LOGLEVEL=debug \
    MTX_RTSP=yes \
    MTX_HLS=yes \
    MTX_WEBRTC=yes \
    MTX_RTMP=yes \
    MTX_SRT=yes

# 포트 노출
# RTSP
EXPOSE 8554/tcp 8554/udp
# Web UI/HLS
EXPOSE 8888/tcp
# WebRTC ICE TCP
EXPOSE 8889/tcp
# RTMP 입력
EXPOSE 1935/tcp
# SRT 입력(예시 포트)
EXPOSE 8890/udp

ENTRYPOINT ["/mediamtx"]
# 기본 실행은 config 없이
CMD []