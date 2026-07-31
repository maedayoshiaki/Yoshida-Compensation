import os
import time
import uuid

import edsdk
from edsdk import (
    CameraCommand,
    ObjectEvent,
    PropID,
    FileCreateDisposition,
    Access,
    SaveTo,
    EdsObject,
    PropertyEvent,
)
from edsdk import *
from edsdk.constants.properties import *


def callback_property(event: PropertyEvent, property_id: PropID, parameter: int) -> int:
    print("event: ", event)
    print("Property changed:", property_id)
    print("Parameter:", parameter)
    return 0


edsdk.InitializeSDK()
cam_list = edsdk.GetCameraList()
nr_cameras = edsdk.GetChildCount(cam_list)

if nr_cameras == 0:
    print("No cameras connected")
    exit(1)

cam = edsdk.GetChildAtIndex(cam_list, 0)
edsdk.OpenSession(cam)
edsdk.SendStatusCommand(cam, CameraStatusCommand.EnterDirectTransfer, 0)
edsdk.SendStatusCommand(cam, CameraStatusCommand.ExitDirectTransfer, 0)

# edsdk.SendCommand(cam, CameraCommand.TakePicture, 0)
# edsdk.SendStatusCommand(cam, CameraStatusCommand.UIUnLock, 0)

edsdk.CloseSession(cam)
edsdk.TerminateSDK()
print("EDS SDK example completed successfully.")
