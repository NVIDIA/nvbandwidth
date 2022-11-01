#!/bin/sh
# Utility script that attempts to install
# necessary software components needed to
# build nvbandwidth

apt install git
apt install build-essential
apt install libboost-program-options-dev
apt install cmake
output=$(cmake --version | sed -n 1p | sed 's/[^0-9]*//g')
if [ $output -lt 3200 ]; then
    echo "Upgrade cmake version to 3.20 or above to build nvbandwidth"
    exit 1
fi
cmake .
make
