import 'package:flutter_tesseract_ocr/flutter_tesseract_ocr.dart';

class OcrService {
	static const String _language = 'eng';

	Future<String> recognizeTextFromImage(String imagePath) async {
		return await FlutterTesseractOcr.extractText(
			imagePath,
			language: _language,
		);
	}
} 