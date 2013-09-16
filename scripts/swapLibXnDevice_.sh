#!/usr/bin/bash

## Replace
#sudo cp /usr/lib/libXnDeviceSensorV2.so /usr/lib/libXnDeviceSensorV2.so.bak
#sudo ln -sfn ~/3rdparty/SensorKinect/LocalInstall/usr/lib/libXnDeviceSensorV2KM.so /usr/lib/libXnDeviceSensorV2.so

#sudo cp /usr/lib/libXnDeviceFile.so /usr/lib/libXnDeviceFile.so.bak
#sudo ln -sfn ~/3rdparty/SensorKinect/LocalInstall/usr/lib/libXnDeviceFile.so /usr/lib/libXnDeviceFile.so

## Reset
sudo cp /usr/lib/libXnDeviceFile.so.bak /usr/lib/libXnDeviceFile.so
sudo cp /usr/lib/libXnDeviceSensorV2.so.bak /usr/lib/libXnDeviceSensorV2.so






