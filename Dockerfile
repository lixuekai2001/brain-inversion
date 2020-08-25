#FROM multiphenics/multiphenics
FROM ba4b9c0dbca7
USER root
RUN apt-get -qq update && \
    apt-get -qq install libgl1-mesa-dev libglu1-mesa-dev libxcursor-dev libxinerama-dev python3-h5py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    pip3 -q install --upgrade --no-cache-dir pip meshio==4.0.13 jdata pyvista gmsh pygmsh pyyaml &&\
    export PYTHONPATH=/usr/local/lib/python3/dist-packages/gmsh-4.6.0-Linux64-sdk/lib/ &&\
    cat /dev/null > $FENICS_HOME/WELCOME
