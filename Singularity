#Bootstrap: docker
Bootstrap: docker-daemon
#From: multiphenics/multiphenics
From: fenics/openmp:latest

%environment
    PYTHONPATH=/usr/local/lib/python3/dist-packages/gmsh-4.6.0-Linux64-sdk/lib/
    export PYTHONPATH

%post
    chmod -R 777 var/*
    apt-get -q update && 
    apt-get install -q -y libgl1-mesa-dev libglu1-mesa-dev libxcursor-dev libxinerama-dev python3-h5py
    apt-get clean
    pip3 -q install --upgrade --no-cache-dir pip meshio==4.0.13 jdata pyvista gmsh pygmsh pyyaml