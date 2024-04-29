FROM mcr.microsoft.com/devcontainers/cpp:ubuntu

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="none"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
# Install Eigen and FFTW
# Install Eigen and FFTW
#python3-dev is used to install python.h depenencie of matplotlib
RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    libfftw3-dev \ 
    libsndfile1-dev \
    libtag1-dev\
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libavresample-dev \
    libyaml-cpp-dev \
    libsamplerate0-dev \
     g++ \
    python3 \
    python3-dev \  
    python3-pip \
    python3-numpy \
    && python3 -m pip install --no-cache-dir matplotlib 

    # Install Essentia
RUN apt-get update \
    && apt-get install -y build-essential python3-dev git \
    libfftw3-dev libavcodec-dev libavformat-dev libavresample-dev \
    libsamplerate0-dev libtag1-dev libyaml-dev \
    && mkdir /essentia && cd /essentia && git clone https://github.com/MTG/essentia.git \
    && cd /essentia/essentia && git checkout v2.1_beta5 && python3 waf configure --with-python --with-examples --with-vamp \
    && python3 waf && python3 waf install && ldconfig \
    &&  apt-get remove -y build-essential libyaml-dev libfftw3-dev libavcodec-dev \
        libavformat-dev libavutil-dev libavresample-dev python-dev libsamplerate0-dev \
        libtag1-dev \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && cd / && rm -rf /essentia/essentia

# Clone the matplotlib-cpp repository and copy matplotlibcpp.h to /usr/include
RUN apt-get install -y git \
    && git clone https://github.com/lava/matplotlib-cpp.git /tmp/matplotlib-cpp \
    && cp /tmp/matplotlib-cpp/matplotlibcpp.h /usr/include/ \
    && rm -rf /tmp/matplotlib-cpp \ 
    && apt-get remove -y git \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*