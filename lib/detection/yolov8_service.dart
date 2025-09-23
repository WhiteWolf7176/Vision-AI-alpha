import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class Yolov8Service {
  Interpreter? _interpreter;
  static const int _inputSize = 640;
  static const double _confidenceThreshold = 0.25;
  static const double _nmsIouThreshold = 0.45;
  
  // COCO labels
  final List<String> cocoLabels = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
  ];

  Future<void> loadModel() async {
    // Using yolov8m.tflite as per our project setup
    _interpreter ??= await Interpreter.fromAsset('assets/models/yolov8m_float32.tflite');
  }

  // This function for real-time streaming is not used in our current workflow,
  // but is kept here for potential future use.
  Future<List<List<List<List<double>>>>> preProcessCameraImage(CameraImage cameraImage) async {
    final img.Image rgba = _yuv420ToImage(cameraImage);
    final img.Image resized = img.copyResize(
      rgba,
      width: _inputSize,
      height: _inputSize,
      interpolation: img.Interpolation.average,
    );

    final List<List<List<List<double>>>> input = List.generate(
      1,
      (_) => List.generate(
        _inputSize,
        (y) => List.generate(
          _inputSize,
          (x) {
            final pixel = resized.getPixel(x, y);
            final r = pixel.r / 255.0;
            final g = pixel.g / 255.0;
            final b = pixel.b / 255.0;
            return [r, g, b];
          },
        ),
      ),
    );

    return input;
  }

  // This function for real-time streaming is not used in our current workflow.
  Future<List<Map<String, dynamic>>> predict(CameraImage cameraImage) async {
    final interpreter = _interpreter;
    if (interpreter == null) {
      throw StateError('Interpreter not initialized. Call loadModel() first.');
    }

    final input = await preProcessCameraImage(cameraImage);

    final List<List<List<double>>> output = List.generate(
      1,
      (_) => List.generate(84, (_) => List<double>.filled(8400, 0.0)),
    );

    interpreter.run(input, output);

    final List<_Detection> rawDetections = _postProcess(output[0]);
    final List<_Detection> nms = _nonMaxSuppression(rawDetections, _nmsIouThreshold);

    return nms
      .map((d) => {
        'box': [d.x1, d.y1, d.x2, d.y2],
        'confidence': d.score,
        'classId': d.classId,
        'tag': 'object_${d.classId}', // Added placeholder tag
      })
      .toList();
  }

  Future<List<Map<String, dynamic>>> predictFromFile(String imagePath) async {
    try {
      final interpreter = _interpreter;
      if (interpreter == null) {
        throw StateError('Interpreter not initialized. Call loadModel() first.');
      }

      final List<int> bytes = File(imagePath).readAsBytesSync();
      img.Image? decoded = img.decodeImage(Uint8List.fromList(bytes));
      if (decoded == null) {
        return <Map<String, dynamic>>[];
      }
      
      // NEW: Check if the image is portrait (height > width) and rotate it.
      if (decoded.height > decoded.width) {
        decoded = img.copyRotate(decoded, angle: 90);
      }
      
      // MODIFIED: Use the potentially rotated 'decoded' image for resizing.
      final img.Image resized = img.copyResize(
        decoded,
        width: _inputSize,
        height: _inputSize,
        interpolation: img.Interpolation.average,
      );

      final List<List<List<List<double>>>> input = List.generate(
        1,
        (_) => List.generate(
          _inputSize,
          (y) => List.generate(
            _inputSize,
            (x) {
              final pixel = resized.getPixel(x, y);
              final r = pixel.r / 255.0;
              final g = pixel.g / 255.0;
              final b = pixel.b / 255.0;
              return [r, g, b];
            },
          ),
        ),
      );

      final List<List<List<double>>> output = List.generate(
        1,
        (_) => List.generate(84, (_) => List<double>.filled(8400, 0.0)),
      );

      interpreter.run(input, output);

      final List<_Detection> rawDetections = _postProcess(output[0]);
      final List<_Detection> nms = _nonMaxSuppression(rawDetections, _nmsIouThreshold);

      return nms
        .map((d) => {
          'box': [d.x1, d.y1, d.x2, d.y2],
          'confidence': d.score,
          'classId': d.classId,
          'tag': cocoLabels[d.classId], // coco labels output
        })
        .toList();
    } catch (e, st) {
      print('predictFromFile error: $e\n$st');
      return <Map<String, dynamic>>[];
    }
  }

  img.Image _yuv420ToImage(CameraImage image) {
    if (image.format.group != ImageFormatGroup.yuv420) {
      throw UnsupportedError('Only YUV420 camera image format is supported');
    }

    final int width = image.width;
    final int height = image.height;

    final Plane planeY = image.planes[0];
    final Plane planeU = image.planes[1];
    final Plane planeV = image.planes[2];

    final Uint8List bytesY = planeY.bytes;
    final Uint8List bytesU = planeU.bytes;
    final Uint8List bytesV = planeV.bytes;

    final int strideY = planeY.bytesPerRow;
    final int strideU = planeU.bytesPerRow;
    final int pixelStrideU = planeU.bytesPerPixel ?? 1;

    final img.Image out = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex = (y ~/ 2) * strideU + (x ~/ 2) * pixelStrideU;

        final int yp = bytesY[y * strideY + x] & 0xFF;
        final int up = bytesU[uvIndex] & 0xFF;
        final int vp = bytesV[uvIndex] & 0xFF;

        final double yVal = yp.toDouble();
        final double uVal = up.toDouble() - 128.0;
        final double vVal = vp.toDouble() - 128.0;

        // Convert YUV -> RGB (BT.601)
        double r = yVal + 1.402 * vVal;
        double g = yVal - 0.344136 * uVal - 0.714136 * vVal;
        double b = yVal + 1.772 * uVal;

        int ri = r.clamp(0.0, 255.0).toInt();
        int gi = g.clamp(0.0, 255.0).toInt();
        int bi = b.clamp(0.0, 255.0).toInt();

        out.setPixelRgba(x, y, ri, gi, bi, 255);
      }
    }

    return out;
  }

  List<_Detection> _postProcess(List<List<double>> output84x8400) {
    // This seems to have a logic error. The output shape is [1, 84, 8400].
    // Let's transpose it first for easier processing.
    final transposedOutput = List.generate(
        8400, (i) => List.generate(84, (j) => output84x8400[j][i]));

    final int numAnchors = 8400;
    final int numClasses = 80;

    final List<_Detection> detections = <_Detection>[];

    for (int i = 0; i < numAnchors; i++) {
        final double cx = transposedOutput[i][0];
        final double cy = transposedOutput[i][1];
        final double w = transposedOutput[i][2];
        final double h = transposedOutput[i][3];

        // Find best class
        int bestClassId = -1;
        double bestClassScore = 0.0;
        for (int c = 0; c < numClasses; c++) {
            final double score = transposedOutput[i][4 + c]; // Class scores start at index 4
            if (score > bestClassScore) {
                bestClassScore = score;
                bestClassId = c;
            }
        }
        
        // Use the single confidence score if it exists, otherwise multiply
        // The standard YOLOv8 output has class scores, not a separate objectness score
        if (bestClassScore > _confidenceThreshold) {
            final double x1 = (cx - w / 2.0);
            final double y1 = (cy - h / 2.0);
            final double x2 = (cx + w / 2.0);
            final double y2 = (cy + h / 2.0);

            detections.add(_Detection(
                x1: x1,
                y1: y1,
                x2: x2,
                y2: y2,
                score: bestClassScore,
                classId: bestClassId,
            ));
        }
    }
    return detections;
  }

  List<_Detection> _nonMaxSuppression(List<_Detection> detections, double iouThreshold) {
    final List<_Detection> result = <_Detection>[];
    final Map<int, List<_Detection>> byClass = <int, List<_Detection>>{};
    for (final d in detections) {
      byClass.putIfAbsent(d.classId, () => <_Detection>[]).add(d);
    }

    for (final entry in byClass.entries) {
      final List<_Detection> dets = entry.value..sort((a, b) => b.score.compareTo(a.score));
      final List<bool> removed = List<bool>.filled(dets.length, false);

      for (int i = 0; i < dets.length; i++) {
        if (removed[i]) continue;
        final _Detection a = dets[i];
        result.add(a);
        for (int j = i + 1; j < dets.length; j++) {
          if (removed[j]) continue;
          final _Detection b = dets[j];
          final double overlap = _iou(a, b);
          if (overlap > iouThreshold) {
            removed[j] = true;
          }
        }
      }
    }
    return result;
  }

  double _iou(_Detection a, _Detection b) {
    final double x1 = math.max(a.x1, b.x1);
    final double y1 = math.max(a.y1, b.y1);
    final double x2 = math.min(a.x2, b.x2);
    final double y2 = math.min(a.y2, b.y2);

    final double w = math.max(0.0, x2 - x1);
    final double h = math.max(0.0, y2 - y1);
    final double inter = w * h;

    final double areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    final double areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    final double union = areaA + areaB - inter + 1e-6;
    return inter / union;
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}

class _Detection {
  final double x1;
  final double y1;
  final double x2;
  final double y2;
  final double score;
  final int classId;

  _Detection({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.score,
    required this.classId,
  });
}