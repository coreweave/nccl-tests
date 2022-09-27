FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends \
        build-essential libtool autoconf automake autotools-dev unzip \
        ca-certificates \
        wget \
        iputils-ping net-tools \
        libnuma1 libpmi0-dev libpmi2-0-dev libsubunit0

# Mellanox OFED (latest)
RUN wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add -
RUN cd /etc/apt/sources.list.d/ && wget https://linux.mellanox.com/public/repo/mlnx_ofed/latest/ubuntu18.04/mellanox_mlnx_ofed.list

 RUN apt-get -qq update \
    && apt-get -qq install -y --no-install-recommends \
        mlnx-ofed-hpc-user-only perftest ibverbs-utils libibverbs-dev libibumad3 rdmacm-utils infiniband-diags \
        openssh-server \
    && rm -rf /var/lib/apt/lists/*

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

# NCCL SHARP PLugin (master)
RUN cd /tmp && \
    wget -q https://github.com/Mellanox/nccl-rdma-sharp-plugins/archive/refs/heads/master.zip && \
    unzip master.zip && \
    cd nccl-rdma-sharp-plugins-master && \
    ./autogen.sh && \
    ./configure --with-cuda=/usr/local/cuda-11.6 --prefix=/usr && \
    make && \
    make install && \
    rm /hpcx/nccl_rdma_sharp_plugin/lib/*

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

# Being auto-generated paths
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

# NCCL Tests
ENV NCCL_TESTS_COMMITISH=8274cb4
WORKDIR /opt/nccl_tests
RUN  wget -q -O - https://github.com/NVIDIA/nccl-tests/archive/${NCCL_TESTS_COMMITISH}.tar.gz | tar --strip-components=1 -xzf - \
   && make MPI=1

RUN ldconfig

# SSH dependencies for MPI
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config && \
    mkdir /var/run/sshd