import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:visionai/detection/yolov8_service.dart';
import 'package:visionai/ocr/ocr_service.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:flutter_tts/flutter_tts.dart';
import 'dart:ui';

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
	bool isResultExpanded = true;

	late final stt.SpeechToText _speech;
	bool _isListening = false;
	bool _speechAvailable = false;

	@override
	void initState() {
		super.initState();
		_speech = stt.SpeechToText();
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

// Format detections for TTS
	String formatDetectionsForTTS(List<Map<String, dynamic>> detections) {
		if (detections.isEmpty) {
			return "No objects detected.";
		}

		// Get a list of unique object names
		final objectNames = detections.map((r) => r['tag'] as String).toSet().toList();

		if (objectNames.length == 1) {
			return "There is a ${objectNames.first}.";
		} 
		else if (objectNames.length == 2) {
			return "There are a ${objectNames.first} and a ${objectNames.last}.";
		} 
		else {
			// For 3 or more items, join with commas and add "and" before the last one.
			final allButLast = objectNames.sublist(0, objectNames.length - 1).join(', a ');
			final last = objectNames.last;
			return "There are a $allButLast, and a $last.";
		}
	}

	Future<void> captureAndProcessImage() async {
		final controller = _cameraController;
		if (controller == null || !controller.value.isInitialized) return;
		if (isProcessing) return;

		try {
			setState(() {
				isProcessing = true;
			});

			final XFile file = await controller.takePicture();
			final String imagePath = file.path;

			final List<Map<String, dynamic>>? yoloResults = await _yolov8Service.predictFromFile(imagePath);
			final List<Map<String, dynamic>> safeYoloResults = yoloResults ?? [];

			String safeOcrText = '';
			try {
				final String? ocrText = await _ocrService.recognizeTextFromImage(imagePath);
				safeOcrText = ocrText?.trim() ?? '';
			} catch (_) {
				// ignore OCR errors, proceed with empty text
			}

			final StringBuffer resultBuffer = StringBuffer();
			resultBuffer.write(formatDetectionsForTTS(safeYoloResults));
			resultBuffer.write(' ');
			if (safeOcrText.isNotEmpty) {
				resultBuffer.write("The text says: $safeOcrText");
			}
			final String finalResultString = resultBuffer.toString();

			if (!mounted) return;
			setState(() {
				processingResult = finalResultString.isEmpty ? "Nothing detected. Please try again." : finalResultString;
			});

			await flutterTts.speak(processingResult!);
		} catch (e, stackTrace) {
			// This outer catch is for other unexpected errors (like taking a picture failing)
			print('!!! AN UNEXPECTED ERROR OCCURRED: $e');
			if (mounted) {
				setState(() {
					processingResult = "An unexpected error occurred.";
				});
				flutterTts.speak("An unexpected error occurred.");
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

					// Add processing overlay spinner while keeping preview live
					if (isProcessing)
						Container(
							color: Colors.black.withOpacity(0.5),
							child: const Center(child: CircularProgressIndicator()),
						),
					// Result and controls combined at bottom to avoid overlap
					Positioned(
						left: 0,
						right: 0,
						bottom: 0,
						child: SafeArea(
							minimum: const EdgeInsets.only(bottom: 16),
							child: Column(
								mainAxisSize: MainAxisSize.min,
								children: [
									if (processingResult != null)
										Padding(
											padding: const EdgeInsets.symmetric(horizontal: 16),
																										child: ClipRRect(
																borderRadius: BorderRadius.circular(24),
												child: BackdropFilter(
													filter: ImageFilter.blur(sigmaX: 12, sigmaY: 12),
													child: Container(
														padding: const EdgeInsets.all(12),
														decoration: BoxDecoration(
															color: Colors.white.withOpacity(0.12),
															borderRadius: BorderRadius.circular(24),
														),
														child: Column(
															mainAxisSize: MainAxisSize.min,
															children: [
																SizedBox(
																	height: isResultExpanded ? 200.0 : 50.0,
																	child: SingleChildScrollView(
																		child: Text(
																			processingResult!,
																			style: const TextStyle(
																				fontSize: 16,
																				color: Colors.white,
																				fontWeight: FontWeight.w500,
																			),
																			textAlign: TextAlign.center,
																		),
																	),
																),
																Align(
																	alignment: Alignment.centerRight,
																	child: IconButton(
																		icon: Icon(
																			isResultExpanded ? Icons.expand_more : Icons.expand_less,
																			color: Colors.white,
																		),
																		onPressed: () {
																		setState(() {
																			isResultExpanded = !isResultExpanded;
																		});
																	},
																	),
																),
															],
														),
													),
												),
											),
										),

															const SizedBox(height: 12),

															Container(
																margin: const EdgeInsets.symmetric(horizontal: 16),
																child: ClipRRect(
																	borderRadius: BorderRadius.circular(24),
																	child: BackdropFilter(
																		filter: ImageFilter.blur(sigmaX: 12, sigmaY: 12),
																		child: Container(
																			padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
																			decoration: BoxDecoration(
																				color: Colors.white.withOpacity(0.12),
																				borderRadius: BorderRadius.circular(24),
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
                              ),
								],
							),
						),
					),
				],
			),
		);
	}
} 