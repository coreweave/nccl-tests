ARG CUDA_VERSION_MINOR=11.7.1
FROM nvidia/cuda:${CUDA_VERSION_MINOR}-devel-ubuntu22.04

ARG CUDA_VERSION_MAJOR=11.7
ARG TARGET_NCCL_VERSION=2.14.3-1

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update && \
        apt-get -qq install -y --allow-change-held-packages --no-install-recommends \
        build-essential libtool autoconf automake autotools-dev unzip \
        ca-certificates \
        wget curl openssh-server vim \
        iputils-ping net-tools \
        libnuma1 libsubunit0 libpci-dev \
        libpmix-dev \
        datacenter-gpu-manager \
        libnccl2=$TARGET_NCCL_VERSION+cuda${CUDA_VERSION_MAJOR} libnccl-dev=${TARGET_NCCL_VERSION}+cuda${CUDA_VERSION_MAJOR}

# Mellanox OFED (latest)
RUN wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add -
RUN cd /etc/apt/sources.list.d/ && wget https://linux.mellanox.com/public/repo/mlnx_ofed/latest/ubuntu18.04/mellanox_mlnx_ofed.list

RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends \
    ibverbs-utils libibverbs-dev libibumad3 libibumad-dev librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils \
    && rm -rf /var/lib/apt/lists/*
#         mlnx-ofed-hpc-user-only

# IB perftest with GDR
ENV PERFTEST_VERSION=4.5-0.17
ENV PERFTEST_VERSION_HASH=g6f25f23

RUN mkdir /tmp/build && \
    cd /tmp/build && \
    wget -q https://github.com/linux-rdma/perftest/releases/download/v${PERFTEST_VERSION}/perftest-${PERFTEST_VERSION}.${PERFTEST_VERSION_HASH}.tar.gz && \
    tar xvf perftest-${PERFTEST_VERSION}.${PERFTEST_VERSION_HASH}.tar.gz && \
    cd perftest-4.5 && \
    ./configure CUDA_H_PATH=/usr/local/cuda/include/cuda.h && \
    make install && \
    cd /tmp && \
    rm -r /tmp/build

# Build GPU Bandwidthtest from samples
ARG CUDA_SAMPLES_VERSION=11.6
RUN mkdir /tmp/build && \
    cd /tmp/build && \
    curl -sLo master.zip https://github.com/NVIDIA/cuda-samples/archive/refs/tags/v${CUDA_SAMPLES_VERSION}.zip && \
    unzip master.zip && \
    cd cuda-samples-${CUDA_SAMPLES_VERSION}/Samples/1_Utilities/bandwidthTest && \
    make && \
    install bandwidthTest /usr/bin/ && \
    cd /tmp && \
    rm -r /tmp/build

# HPC-X (2.12)
ENV HPCX_VERSION=2.12
RUN cd /tmp && \
    wget -q -O - http://blobstore.s3.ord1.coreweave.com/drivers/hpcx-v${HPCX_VERSION}-gcc-MLNX_OFED_LINUX-5-ubuntu20.04-cuda11-gdrcopy2-nccl${HPCX_VERSION}-x86_64.tbz | tar xjf - && \
    mv hpcx-v${HPCX_VERSION}-gcc-MLNX_OFED_LINUX-5-ubuntu20.04-cuda11-gdrcopy2-nccl${HPCX_VERSION}-x86_64 /hpcx

# GDRCopy userspace components (2.3)
RUN cd /tmp && \
    wget -q https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2011.4/x86/Ubuntu20.04/gdrcopy-tests_2.3-1_amd64.cuda11_4.Ubuntu20_04.deb && \
    wget -q https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2011.4/x86/Ubuntu20.04/libgdrapi_2.3-1_amd64.Ubuntu20_04.deb && \
    dpkg -i *.deb && \
    rm *.deb

# HPC-X Environment variables
#
# The following envs are from the output of the printpaths script. Uncomment the rows below to
# run the script as part of a Docker build. Copy-paste the updated output in here.
# These ENVs need to be updated on new HPC-X install or any path related modifications before 
# this stage in the Dockerfile.
#
#COPY ./printpaths.sh /tmp
#RUN /bin/bash -c '\
#   source /hpcx/hpcx-init.sh && \
#   hpcx_load && \
#   /tmp/printpaths.sh && \
#   rm /tmp/printpaths.sh'

# Begin auto-generated paths
ENV HPCX_DIR=/hpcx
ENV HPCX_UCX_DIR=/hpcx/ucx
ENV HPCX_UCC_DIR=/hpcx/ucc
ENV HPCX_SHARP_DIR=/hpcx/sharp
ENV HPCX_NCCL_RDMA_SHARP_PLUGIN_DIR=/hpcx/nccl_rdma_sharp_plugin
ENV HPCX_HCOLL_DIR=/hpcx/hcoll
ENV HPCX_MPI_DIR=/hpcx/ompi
ENV HPCX_OSHMEM_DIR=/hpcx/ompi
ENV HPCX_MPI_TESTS_DIR=/hpcx/ompi/tests
ENV HPCX_OSU_DIR=/hpcx/ompi/tests/osu-micro-benchmarks-5.8
ENV HPCX_OSU_CUDA_DIR=/hpcx/ompi/tests/osu-micro-benchmarks-5.8-cuda
ENV HPCX_IPM_DIR=/hpcx/ompi/tests/ipm-2.0.6
ENV HPCX_CLUSTERKIT_DIR=/hpcx/clusterkit
ENV OMPI_HOME=/hpcx/ompi
ENV MPI_HOME=/hpcx/ompi
ENV OSHMEM_HOME=/hpcx/ompi
ENV OPAL_PREFIX=/hpcx/ompi
ENV OLD_PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PATH=/hpcx/clusterkit/bin:/hpcx/hcoll/bin:/hpcx/ucc/bin:/hpcx/ucx/bin:/hpcx/ompi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV OLD_LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV LD_LIBRARY_PATH=/hpcx/nccl_rdma_sharp_plugin/lib:/hpcx/ucc/lib/ucc:/hpcx/ucc/lib:/hpcx/ucx/lib/ucx:/hpcx/ucx/lib:/hpcx/sharp/lib:/hpcx/hcoll/lib:/hpcx/ompi/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV OLD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV LIBRARY_PATH=/hpcx/nccl_rdma_sharp_plugin/lib:/hpcx/ompi/lib:/hpcx/sharp/lib:/hpcx/ucc/lib:/hpcx/ucx/lib:/hpcx/hcoll/lib:/hpcx/ompi/lib:/usr/local/cuda/lib64/stubs
ENV OLD_CPATH=
ENV CPATH=/hpcx/ompi/include:/hpcx/ucc/include:/hpcx/ucx/include:/hpcx/sharp/include:/hpcx/hcoll/include:
ENV PKG_CONFIG_PATH=/hpcx/hcoll/lib/pkgconfig:/hpcx/sharp/lib/pkgconfig:/hpcx/ucx/lib/pkgconfig:/hpcx/ompi/lib/pkgconfig:
# End of auto-generated paths

# Rebuild OpenMPI to support SLURM
RUN cd /hpcx/sources/ && rm -r /hpcx/ompi && tar -zxvf openmpi-gitclone.tar.gz && cd openmpi-gitclone && \
    ./configure --prefix=/hpcx/ompi \
           --with-hcoll=/hpcx/hcoll --with-ucx=/hpcx/ucx \
           --with-platform=contrib/platform/mellanox/optimized \
           --with-slurm --with-pmix=/usr/lib/x86_64-linux-gnu/pmix2 --with-hwloc --with-libevent \
           --without-xpmem --with-cuda --with-ucc=/hpcx/ucc && \
           make -j14 && \
           make -j14 install && \
           cd .. && \
           rm -r openmpi-gitclone

# NCCL SHARP PLugin (master)
RUN cd /tmp && \
    wget -q https://github.com/Mellanox/nccl-rdma-sharp-plugins/archive/refs/heads/master.zip && \
    unzip master.zip && \
    cd nccl-rdma-sharp-plugins-master && \
    ./autogen.sh && \
    ./configure --with-cuda=/usr/local/cuda-${CUDA_VERSION_MAJOR} --prefix=/usr && \
    make && \
    make install && \
    rm /hpcx/nccl_rdma_sharp_plugin/lib/* && \
    rm -r /tmp/*

# NCCL Tests
ENV NCCL_TESTS_COMMITISH=d313d20
WORKDIR /opt/nccl-tests
RUN  wget -q -O - https://github.com/NVIDIA/nccl-tests/archive/${NCCL_TESTS_COMMITISH}.tar.gz | tar --strip-components=1 -xzf - \
   && make MPI=1 \
   && ln -s /opt/nccl-tests /opt/nccl_tests

RUN ldconfig

# SSH dependencies for MPI
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config && \
    mkdir /var/run/sshd