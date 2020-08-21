FROM multiphenics/multiphenics
USER root
RUN apt-get -qq update && \
    apt-get -qq install libgl1-mesa-glx gmsh python3-h5py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    pip3 -q install --upgrade --no-cache-dir pip meshio[all] jdata pyvista pygmsh pyyaml &&\
    cat /dev/null > $FENICS_HOME/WELCOME
