import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';

class OcrService {
	Future<String> recognizeTextFromImage(String imagePath) async {
		final InputImage inputImage = InputImage.fromFilePath(imagePath);
		final TextRecognizer textRecognizer = TextRecognizer(script: TextRecognitionScript.latin);
		try {
			final RecognizedText recognizedText = await textRecognizer.processImage(inputImage);
			return recognizedText.text;
		} catch (e) {
			return '';
		} finally {
			await textRecognizer.close();
		}
	}
}