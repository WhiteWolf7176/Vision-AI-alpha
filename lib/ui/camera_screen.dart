import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:visionai/detection/yolov8_service.dart';
import 'package:visionai/ocr/ocr_service.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:flutter_tts/flutter_tts.dart';
import 'dart:ui';
import 'package:visionai/services/gemini_service.dart';

class CameraScreen extends StatefulWidget {
	const CameraScreen({super.key});

	@override
	State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> with TickerProviderStateMixin {
	CameraController? _cameraController;
	bool _isInitializing = true;
	String? _errorMessage;

	final Yolov8Service _yolov8Service = Yolov8Service();
	final OcrService _ocrService = OcrService();
	final FlutterTts flutterTts = FlutterTts();
	bool isProcessing = false;
	String? processingResult;
	bool isResultExpanded = true;

  late final GeminiService geminiService;

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
      geminiService = GeminiService();
      await geminiService.initialize();

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
				ResolutionPreset.max,
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
				if (recognized.contains('scan book') || recognized.contains('read text') || recognized.contains('capture') || recognized.contains("tell me what's ahead") || recognized.contains('I need assisstance')) {
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
        processingResult = null;
			});

			flutterTts.speak("Processing");

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

      // --- STAGE 2: Detailed Cloud Analysis (Gemini) ---
      print("Starting detailed analysis with AI...");
      final geminiResult = await geminiService.describeImage(imagePath);
      
      if (!mounted) return;
      setState(() {
        // Update the UI with the richer description
        processingResult = geminiResult;
      });
      // Speak the new, better result
      await flutterTts.speak(geminiResult);


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
						SizedBox.expand(
              child: FittedBox(
                fit: BoxFit.cover,
                child: SizedBox(
                  width: _cameraController!.value.previewSize!.height,
                  height: _cameraController!.value.previewSize!.width,
                  child: CameraPreview(_cameraController!),
                ),
              ),
            ),
					//else
						//const SizedBox.shrink(),

					// Add processing overlay spinner while keeping preview live
					if (isProcessing)
						Container(
							color: Colors.black.withOpacity(0.5),
							child: const Center(child: CircularProgressIndicator()),
						),
					// Result and controls combined at bottom to avoid overlap
					Positioned(
            bottom: 32.0,
            left: 20,
            right: 20,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                if (processingResult != null && !isProcessing)
                  // MODIFIED: Wrapped the box in a GestureDetector for a subtle toggle.
                   GestureDetector(
                    onTap: () => setState(() => isResultExpanded = !isResultExpanded),
                    // MODIFIED: Replaced AnimatedContainer with AnimatedSize for a smoother effect.
                    child: _buildGlassmorphicBox(
                      child: AnimatedSize(
                        duration: const Duration(milliseconds: 400),
                        curve: Curves.easeInOutCubic,
                        child: Container(
                          constraints: BoxConstraints(
                            maxHeight: isResultExpanded ? 200 : 60,
                          ),
                          padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 16.0),
                          child: Stack(
                            children: [
                              SingleChildScrollView(
                                child: Text(
                                  processingResult!,
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 16,
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                              ),
                              // NEW: Added a subtle, animated icon as a visual cue.
                              Positioned(
                                bottom: 0,
                                right: 0,
                                child: AnimatedOpacity(
                                  duration: const Duration(milliseconds: 300),
                                  opacity: isResultExpanded ? 1.0 : 0.0,
                                  child: Icon(
                                    isResultExpanded ? Icons.keyboard_arrow_up : Icons.keyboard_arrow_down,
                                    color: Colors.white38,
                                    size: 20,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                
                if (processingResult != null && !isProcessing)
                  const SizedBox(height: 12),

                _buildGlassmorphicBox(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8.0),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                         IconButton(
                           icon: const Icon(Icons.history, color: Colors.white, size: 30),
                           onPressed: () {},
                         ),
                        GestureDetector(
                          onTap: captureAndProcessImage,
                          child: Container(
                            height: 80,
                            width: 80,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              border: Border.all(color: Colors.white, width: 4),
                            ),
                          ),
                        ),
                        IconButton(
                          icon: Icon(
                            Icons.mic,
                            color:
                                _isListening ? Colors.redAccent : Colors.white,
                          ),
                          onPressed: _startListening,
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildGlassmorphicBox({required Widget child}) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(24.0),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10.0, sigmaY: 10.0),
        child: Container(
          decoration: BoxDecoration(
            // MODIFIED: Replaced solid color border with a gradient for a glassy sheen.
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Colors.white.withOpacity(0.4),
                Colors.white.withOpacity(0.1),
              ],
            ),
            borderRadius: BorderRadius.circular(24.0),
            border: Border.all(
              color: Colors.white.withOpacity(0.2),
            ),
          ),
          child: child,
        ),
      ),
    );
  }
}