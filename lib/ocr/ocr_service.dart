import 'dart:io';
import 'package:flutter/services.dart' show rootBundle, ByteData;
import 'package:path_provider/path_provider.dart';
import 'package:flutter_tesseract_ocr/flutter_tesseract_ocr.dart';

class OcrService {
  /// This function copies the Tesseract language file from your app's assets
  /// to a folder where the Tesseract engine can access it.
  Future<void> initialize() async {
    final Directory appDocumentsDirectory = await getApplicationDocumentsDirectory();
    final String tessdataPath = '${appDocumentsDirectory.path}/tessdata';
    final String trainedDataPath = '$tessdataPath/eng.traineddata';

    if (!await Directory(tessdataPath).exists()) {
      await Directory(tessdataPath).create(recursive: true);
    }

    if (!await File(trainedDataPath).exists()) {
      final ByteData data = await rootBundle.load('assets/tessdata/eng.traineddata');
      final List<int> bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
      await File(trainedDataPath).writeAsBytes(bytes);
      print("Tesseract data prepared at $trainedDataPath");
    }
  }

  /// This function runs OCR, but with a special argument `tessdata-dir`.
  /// This argument tells the Tesseract engine the exact folder to find its
  /// language data in, completely bypassing its faulty automatic setup.
  Future<String> recognizeTextFromImage(String imagePath) async {
    final Directory appDocumentsDirectory = await getApplicationDocumentsDirectory();
    final String tessdataPath = appDocumentsDirectory.path;

    return await FlutterTesseractOcr.extractText(
      imagePath,
      language: 'eng',
      args: {
        'tessdata-dir': tessdataPath,
      },
    );
  }
}