# Builds a Docker image with the necessary libraries for compiling
# FEniCS.  The image is at:
#
#    https://quay.io/repository/fenicsproject/dev-env
#
# Authors:
# Jack S. Hale <jack.hale@uni.lu>
# Lizao Li <lzlarryli@gmail.com>
# Garth N. Wells <gnw20@cam.ac.uk>
# Jan Blechta <blechta@karlin.mff.cuni.cz>

FROM quay.io/fenicsproject/base:latest
MAINTAINER fenics-project <fenics-support@googlegroups.org>

USER root
WORKDIR /tmp

# Environment variables
ENV PETSC_VERSION=3.14.0 \
    PYBIND11_VERSION=2.4.3 \
    MPI4PY_VERSION=3.0.3 \
    PETSC4PY_VERSION=3.14.0 \
    SLEPC4PY_VERSION=3.12.0 \
    TRILINOS_VERSION=12.10.1 \
    MPICH_VERSION=3.3 \
    MPICH_DIR=/opt/mpich \
    NUM_THREADS=40 \
    #OPENBLAS_NUM_THREADS=1 \
    #OPENBLAS_VERBOSE=0 \
    FENICS_PREFIX=$FENICS_HOME/local

#RUN git clone -q --branch=develop git://github.com/xianyi/OpenBLAS.git && \
#    (cd OpenBLAS \
#    && make DYNAMIC_ARCH=1 NO_AFFINITY=1 NUM_THREADS=40 TARGET=SKYLAKEX USE_OPENMP=1 \
#    && make install)


# Non-Python utilities and libraries
RUN apt-get -qq update && \
    apt-get -y --with-new-pkgs \
        -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install curl && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get -y install \
        bison \
        ccache \
        cmake \
        doxygen \
        flex \
        g++ \
        gfortran \
        git \
        git-lfs \
        graphviz \
        libboost-filesystem-dev \
        libboost-iostreams-dev \
        libboost-math-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libboost-timer-dev \
        libeigen3-dev \
        libfreetype6-dev \
        #liblapack-dev \
        #libopenmpi-dev \
        #libmpich-dev \
        #libopenblas-openmp-dev \
        libpcre3-dev \
        libpng-dev \
        #libhdf5-dev \
        #libhdf5-openmpi-dev \
        libhdf5-mpich-dev \
        libgmp-dev \
        libcln-dev \
        libmpfr-dev \
        man \
        #openmpi-bin \
        #mpich \
        nano \
        pkg-config \
        wget \
        bash-completion && \
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#RUN sudo update-alternatives --config libblas.so.3

#RUN cd /tmp && \
#    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
#    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB&& \


    ## all products:
    #sudo wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
    ## just MKL
#    sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' && \
    ## other (TBB, DAAL, MPI, ...) listed on page
    #sudo dpkg --add-architecture i386 && \
#    apt-get update && \
#    apt-get -y install intel-mkl-2020.0-088 && \
    #update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so     libblas.so-x86_64-linux-gnu      /opt/intel/mkl/lib/intel64/libmkl_rt.so 150 && \
    #update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so.3   libblas.so.3-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 150 && \
    #update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so   liblapack.so-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 150 && \
    #update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so.3 liblapack.so.3-x86_64-linux-gnu  /opt/intel/mkl/lib/intel64/libmkl_rt.so 150 && \

#    echo "/opt/intel/lib/intel64"     >  /etc/ld.so.conf.d/mkl.conf && \
#    echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/mkl.conf && \
#    ldconfig && \
#    echo "MKL_THREADING_LAYER=GNU" >> /etc/environment

# Install Python3 based environment
RUN apt-get -qq update && \
    apt-get -y --with-new-pkgs \
        -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
        python3-dev \
        python3-flufl.lock \
        python3-numpy \
        python3-ply \
        python3-pytest \
        python3-scipy \
        python3-tk \
        python3-urllib3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install setuptools
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    pip3 install --no-cache-dir setuptools && \
    rm -rf /tmp/*

## Install MPICH
RUN echo "Installing MPICH..." && \
    mkdir -p /tmp/mpich && \
    mkdir -p /opt && \
    export MPICH_URL="http://www.mpich.org/static/downloads/$MPICH_VERSION/mpich-$MPICH_VERSION.tar.gz" && \
    cd /tmp/mpich && wget -O mpich-$MPICH_VERSION.tar.gz $MPICH_URL && tar xzf mpich-$MPICH_VERSION.tar.gz && \
    # Compile and install
    cd /tmp/mpich/mpich-$MPICH_VERSION && ./configure --prefix=$MPICH_DIR && make install

ENV PATH=$MPICH_DIR/bin:$PATH  \
    LD_LIBRARY_PATH=$MPICH_DIR/lib:$LD_LIBRARY_PATH  \
    MANPATH=$MPICH_DIR/share/man:$MANPATH \
    SINGULARITY_MPICH_DIR=$MPICH_DIR  \
    SINGULARITYENV_APPEND_PATH=$MPICH_DIR/bin  \
    SINGULAIRTYENV_APPEND_LD_LIBRARY_PATH=$MPICH_DIR/lib 


#RUN git clone -q --branch=develop git://github.com/xianyi/OpenBLAS.git && \
#    (cd OpenBLAS \
#    && make PREFIX=/usr/local/openblas DYNAMIC_ARCH=1 NUM_THREADS=40 USE_THREADS=1 USE_OPENMP=1 \
#    && make install && ldconfig)
    
# Install PETSc from source
RUN apt-get -qq update && \
    apt-get -y install \
        python-minimal && \
    wget -nc --quiet https://gitlab.com/petsc/petsc/-/archive/v${PETSC_VERSION}/petsc-v${PETSC_VERSION}.tar.gz -O petsc-${PETSC_VERSION}.tar.gz && \
    mkdir -p petsc-src && tar -xf petsc-${PETSC_VERSION}.tar.gz -C petsc-src --strip-components 1 && \
    cd petsc-src && \
    ./configure --COPTFLAGS="-O2" \
                --CXXOPTFLAGS="-O2" \
                --FOPTFLAGS="-O2" \
                --with-blaslapack-dir=$MKLROOT \
                #--with-blaslapack-dir=/opt/OpenBLAS/lib \
                --with-mpi-dir=$MPICH_DIR \
                --with-fortran-bindings=no \
                #--download-fblaslapack=1 \
                #--download-openblas-make-options="NUM_THREADS=40 USE_THREAD=1 TARGET=SKYLAKEX USE_OPENMP=1" \
                --with-debugging=0 \
                --with-openmp \
                --download-hwloc \
                #--download-openmpi \
                #--download-mpich \
                #--download-blacs \
                #--download-hypre \
                 --download-scalapack \
                --download-metis \
                --download-mumps \
                --download-parmetis \
                --download-ptscotch \
                #--download-scalapack \
                #--download-spai \
                #--download-suitesparse \
                #--download-superlu \
                --prefix=/usr/local/petsc-32 && \
     make && \
     make install && \
     rm -rf /tmp/*

# Install SLEPc from source
# NOTE: Had issues building SLEPc from source tarball generated by bitbucket.
# Website tarballs work fine, however.
#RUN apt-get -qq update && \
#    apt-get -y install \
#    python-minimal && \
#    wget -nc --quiet https://gitlab.com/slepc/slepc/-/archive/v${SLEPC_VERSION}/slepc-v${SLEPC_VERSION}.tar.gz -O slepc-${SLEPC_VERSION}.tar.gz && \
#    mkdir -p slepc-src && tar -xf slepc-${SLEPC_VERSION}.tar.gz -C slepc-src --strip-components 1 && \
#    export PETSC_DIR=/usr/local/petsc-32 && \
#    cd slepc-src && \
#    ./configure --prefix=/usr/local/slepc-32 && \
#    make SLEPC_DIR=$(pwd) && \
#    make install && \
#    rm -rf /tmp/*

# By default use the 32-bit build of SLEPc and PETSc.
ENV PETSC_DIR=/usr/local/petsc-32 \
    SLEPC_DIR=/usr/local/slepc-32


# Install jupyterlab, sympy, mpi4py, petsc4py, slepc4py and pybind11 from source.
RUN pip3 install --no-cache-dir jupyter jupyterlab matplotlib sympy==1.1.1 pkgconfig && \
    pip3 install --no-cache-dir https://github.com/mpi4py/mpi4py/archive/${MPI4PY_VERSION}.tar.gz && \
    pip3 install --no-cache-dir https://bitbucket.org/petsc/petsc4py/downloads/petsc4py-${PETSC4PY_VERSION}.tar.gz && \
#    pip3 install --no-cache-dir https://bitbucket.org/slepc/slepc4py/downloads/slepc4py-${SLEPC4PY_VERSION}.tar.gz && \
    pip3 install --no-cache-dir pybind11==${PYBIND11_VERSION} && \
    pip3 install --no-cache-dir git+https://github.com/mathLab/multiphenics.git && \
    wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz && \
    tar -xf v${PYBIND11_VERSION}.tar.gz && \
    cd pybind11-${PYBIND11_VERSION} && \
    mkdir build && \
    cd build && \
    cmake -DPYBIND11_TEST=False ../ && \
    make && \
    make install && \
    rm -rf /tmp/*

USER root

RUN git clone https://github.com/FEniCS/fiat.git && \
    git clone https://bitbucket.org/fenics-project/dijitso && \
    git clone https://github.com/FEniCS/ufl.git && \
    git clone https://bitbucket.org/fenics-project/ffc && \
    git clone https://bitbucket.org/fenics-project/dolfin && \
    git clone https://bitbucket.org/fenics-project/mshr && \
    cd fiat    && pip3 install . && cd .. && \
    cd dijitso && pip3 install . && cd .. && \
    cd ufl     && pip3 install . && cd .. && \
    cd ffc     && pip3 install . && cd .. && \
    mkdir dolfin/build && cd dolfin/build && cmake .. && make install && cd ../.. && \
    mkdir mshr/build   && cd mshr/build   && cmake .. && make install && cd ../.. && \
    cd dolfin/python && pip3 install . && cd ../.. && \
    cd mshr/python   && pip3 install . && cd ../.. && \
    rm -rf /tmp/*

RUN pip3 install --no-cache-dir pyyaml && \ 
    pip3 install --no-cache-dir git+https://github.com/SVMTK/SVMTK.git
    
USER root
RUN apt-get -qq update && \
    apt-get -qq install libgl1-mesa-dev xvfb libglu1-mesa-dev libxcursor-dev libxinerama-dev python3-h5py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    pip3 -q install --upgrade --no-cache-dir pip meshio==4.0.13 jdata pyvista gmsh pygmsh pyyaml cmocean imageio-ffmpeg && \
    cat /dev/null > $FENICS_HOME/WELCOME

ENV PYTHONPATH=/usr/local/lib/python3/dist-packages/gmsh-4.6.0-Linux64-sdk/lib/

