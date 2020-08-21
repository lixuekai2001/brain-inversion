Bootstrap: docker
From: multiphenics/multiphenics


%post
    apt-get update && apt-get -y install libgl1-mesa-glx gmsh python3-h5py
    apt-get clean
    pip3 -q install --upgrade --no-cache-dir pip meshio[all] jdata pyvista pygmsh pyyaml