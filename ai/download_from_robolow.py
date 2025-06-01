from roboflow import Roboflow

rf = Roboflow(api_key="G3bPzT10Q0dtbkvrTzvE")
project = rf.workspace("crumbobly").project("diplom_v3")
version = project.version(1)
dataset = version.download("yolov11")
