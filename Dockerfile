# A simple docker image for Gristmill

FROM tschijnmo/drudge:drudge

WORKDIR /home/src
COPY . gristmill
RUN set -ex; \
        cd gristmill; \
        python3 setup.py build; \
        python3 setup.py install;

WORKDIR /home/work
