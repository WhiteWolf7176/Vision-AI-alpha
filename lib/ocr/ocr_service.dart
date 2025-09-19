import 'dart:io';
import 'package:flutter/services.dart' show rootBundle, ByteData;
import 'package:path_provider/path_provider.dart';
import 'package:flutter_tesseract_ocr/flutter_tesseract_ocr.dart';

class OcrService {
	static const String _language = 'eng';

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
		}
		print("Tesseract data prepared at $trainedDataPath");
	}

	Future<String> recognizeTextFromImage(String imagePath) async {
		return await FlutterTesseractOcr.extractText(
			imagePath,
			language: _language,
		);
	}
} 