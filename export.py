from models.yolov3_tiny import Yolov3Tiny, load_from_ultralytics_state_dict
from models.yolov3_tiny_quant import Yolov3TinyQuant
import torch
import onnxruntime as ort
import argparse

from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup


def main():
    anchors = [[10, 14, 23, 27, 37, 58], [81, 82, 135, 169, 344, 319]]
    num_classes = 80

    model = Yolov3Tiny(num_classes, len(anchors[0]) // 2)
    load_from_ultralytics_state_dict(model, 'weights/yolov3-tiny-state-dict.pt')
    model.to('cpu')
    torch.onnx.export(model, torch.randn(1, 3, 416, 416), 'yolov3-tiny.onnx')


if __name__ == '__main__':
    main()
