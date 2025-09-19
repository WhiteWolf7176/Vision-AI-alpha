import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:visionai/detection/yolov8_service.dart';
import 'package:visionai/ocr/ocr_service.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:flutter_tts/flutter_tts.dart';

class CameraScreen extends StatefulWidget {
	const CameraScreen({super.key});

	@override
	State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
	CameraController? _cameraController;
	bool _isInitializing = true;
	String? _errorMessage;

	final Yolov8Service _yolov8Service = Yolov8Service();
	final OcrService _ocrService = OcrService();
	final FlutterTts flutterTts = FlutterTts();
	bool isProcessing = false;
	String? processingResult;

	late final stt.SpeechToText _speech;
	bool _isListening = false;
	bool _speechAvailable = false;

	@override
	void initState() {
		super.initState();
		_speech = stt.SpeechToText();
		_ocrService.initialize();
		_initializeCameraFlow();
	}

	Future<void> _initializeCameraFlow() async {
		try {
			final hasPermissions = await _requestPermissions();
			if (!hasPermissions) {
				setState(() {
					_isInitializing = false;
					_errorMessage = 'Camera and microphone permissions are required.';
				});
				return;
			}

			// Initialize speech
			_speechAvailable = await _speech.initialize(
				onStatus: (status) {
					if (!mounted) return;
					setState(() {
						_isListening = status == 'listening';
					});
				},
				onError: (error) {
					if (!mounted) return;
					setState(() {
						_isListening = false;
					});
				},
			);

			// Load model in advance
			await _yolov8Service.loadModel();

			final cameras = await availableCameras();
			final backCameras = cameras.where((c) => c.lensDirection == CameraLensDirection.back).toList();
			if (backCameras.isEmpty) {
				setState(() {
					_isInitializing = false;
					_errorMessage = 'No back camera found on this device.';
				});
				return;
			}

			final selectedCamera = backCameras.first;
			final controller = CameraController(
				selectedCamera,
				ResolutionPreset.high,
				enableAudio: true,
				imageFormatGroup: ImageFormatGroup.yuv420,
			);

			await controller.initialize();
			if (!mounted) return;
			setState(() {
				_cameraController = controller;
				_isInitializing = false;
				_errorMessage = null;
			});
		} catch (e) {
			setState(() {
				_isInitializing = false;
				_errorMessage = 'Failed to initialize camera: $e';
			});
		}
	}

	Future<bool> _requestPermissions() async {
		final statuses = await [
			Permission.camera,
			Permission.microphone,
		].request();

		final cameraGranted = statuses[Permission.camera]?.isGranted ?? false;
		final micGranted = statuses[Permission.microphone]?.isGranted ?? false;
		return cameraGranted && micGranted;
	}

	Future<void> _startListening() async {
		if (!_speechAvailable) {
			ScaffoldMessenger.of(context).showSnackBar(
				const SnackBar(content: Text('Speech not available on this device')), 
			);
			return;
		}
		if (_isListening) {
			await _speech.stop();
			setState(() {
				_isListening = false;
			});
			return;
		}

		setState(() {
			_isListening = true;
		});

		await _speech.listen(
			onResult: (result) async {
				final recognized = (result.recognizedWords).toLowerCase();
				if (recognized.contains('scan book') || recognized.contains('read text') || recognized.contains('capture') || recognized.contains("tell me what's ahead") || recognized.contains('tell me what\'s ahead')) {
					await _speech.stop();
					setState(() {
						_isListening = false;
					});
					await captureAndProcessImage();
				}
			},
			localeId: 'en_US',
			onSoundLevelChange: null,
			listenFor: const Duration(seconds: 5),
		);
	}

	Future<void> captureAndProcessImage() async {
		final controller = _cameraController;
		if (controller == null || !controller.value.isInitialized) return;
		if (isProcessing) return;
		setState(() {
			isProcessing = true;
			processingResult = 'Processing...';
		});
		try {
			await controller.setFlashMode(FlashMode.off);
			final XFile file = await controller.takePicture();

			print("--- Starting Capture and Process ---");
			final String imagePath = file.path;
			print("1. Picture taken at: $imagePath");
			final List<Map<String, dynamic>>? yoloResults = await _yolov8Service.predictFromFile(imagePath);
			final List<Map<String, dynamic>> safeYoloResults = yoloResults ?? <Map<String, dynamic>>[];
			print("2. YOLO service returned: $yoloResults");
			print("   Safeguarded YOLO results: $safeYoloResults");
			final String? ocrText = await _ocrService.recognizeTextFromImage(imagePath);
			final String safeOcrText = (ocrText?.trim() ?? '');
			print("3. OCR service returned: '$ocrText'");
			print("   Safeguarded OCR text: '$safeOcrText'");
			final StringBuffer resultBuffer = StringBuffer();
			if (safeYoloResults.isNotEmpty) {
				resultBuffer.write("I see ");
				final String objectNames = safeYoloResults.map((r) => (r['tag'] ?? r['label'] ?? r['class'] ?? r['classId'] ?? 'unknown').toString()).join(', ');
				resultBuffer.write(objectNames);
				resultBuffer.write('. ');
			}
			if (safeOcrText.isNotEmpty) {
				resultBuffer.write("The text says: $safeOcrText");
			}
			String finalResultString = resultBuffer.toString();
			if (finalResultString.isEmpty) {
				finalResultString = "Nothing detected. Please try again.";
			}
			print("4. Final result string: '$finalResultString'");
			print("--- Process Finished ---");
			if (!mounted) return;
			setState(() {
				processingResult = finalResultString;
			});
			await flutterTts.speak(finalResultString);
		} catch (e, stackTrace) {
			print('!!! AN ERROR OCCURRED DURING PROCESSING: $e');
			print('Stack Trace: $stackTrace');
			if (mounted) {
				setState(() {
					processingResult = "Processing failed. Please try again.";
				});
				flutterTts.speak("Processing failed. Please try again.");
			}
		} finally {
			if (mounted) {
				setState(() {
					isProcessing = false;
				});
			}
		}
	}

	@override
	void dispose() {
		_cameraController?.dispose();
		_yolov8Service.dispose();
		super.dispose();
	}

	@override
	Widget build(BuildContext context) {
		return Scaffold(
			backgroundColor: Colors.black,
			body: Stack(
				fit: StackFit.expand,
				children: [
					// Camera preview as full-screen background
					if (_isInitializing)
						const Center(child: CircularProgressIndicator())
					else if (_errorMessage != null)
						Center(
							child: Padding(
								padding: const EdgeInsets.all(24.0),
								child: Column(
									mainAxisSize: MainAxisSize.min,
									children: [
										Icon(Icons.error_outline, color: Colors.white70, size: 48),
										const SizedBox(height: 12),
										Text(
											_errorMessage!,
											style: const TextStyle(color: Colors.white70),
											textAlign: TextAlign.center,
										),
									],
								),
							),
						)
					else if (_cameraController != null && _cameraController!.value.isInitialized)
						Stack(
							fit: StackFit.expand,
							children: [
								CameraPreview(_cameraController!),
							],
						)
					else
						const SizedBox.shrink(),

					// Result overlay at bottom
					if (processingResult != null)
						Positioned(
							left: 0,
							right: 0,
							bottom: 0,
							child: SafeArea(
								minimum: const EdgeInsets.only(bottom: 16),
								child: GestureDetector(
									onTap: () => setState(() => processingResult = null),
									child: Container(
										margin: const EdgeInsets.symmetric(horizontal: 16),
										padding: const EdgeInsets.all(16),
										decoration: BoxDecoration(
											color: Colors.black87,
											borderRadius: BorderRadius.circular(16),
										),
										child: Text(
											processingResult!,
											style: const TextStyle(color: Colors.white),
											textAlign: TextAlign.center,
										),
									),
								),
							),
						),

					// Bottom control bar overlay
					Align(
						alignment: Alignment.bottomCenter,
						child: SafeArea(
							minimum: const EdgeInsets.only(bottom: 16),
							child: Container(
								margin: const EdgeInsets.symmetric(horizontal: 16),
								padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
								decoration: BoxDecoration(
									color: Colors.black54,
									borderRadius: BorderRadius.circular(32),
								),
								child: Row(
									mainAxisAlignment: MainAxisAlignment.spaceBetween,
									children: [
										IconButton(
											icon: const Icon(Icons.history, color: Colors.white),
											onPressed: () {
												// Placeholder: history action
											},
										),
										IconButton(
											iconSize: 72,
											icon: const Icon(Icons.radio_button_unchecked, color: Colors.white),
											onPressed: captureAndProcessImage,
										),
										IconButton(
											icon: Icon(Icons.mic, color: _isListening ? Colors.redAccent : Colors.white),
											onPressed: _startListening,
										),
									],
								),
							),
						),
					),
				],
			),
		);
	}
} 