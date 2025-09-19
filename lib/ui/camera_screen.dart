import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:visionai/detection/yolov8_service.dart';
import 'package:visionai/ocr/ocr_service.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

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
	bool _isProcessingFrame = false;
	bool _isStreaming = false;
	List<Map<String, dynamic>> _detections = <Map<String, dynamic>>[];

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

			await _startImageStream();
		} catch (e) {
			setState(() {
				_isInitializing = false;
				_errorMessage = 'Failed to initialize camera: $e';
			});
		}
	}

	Future<void> _startImageStream() async {
		final controller = _cameraController;
		if (controller == null || !controller.value.isInitialized) return;
		if (_isStreaming || controller.value.isStreamingImages) return;
		await controller.startImageStream((CameraImage image) async {
			if (_isProcessingFrame) return;
			_isProcessingFrame = true;
			try {
				final results = await _yolov8Service.predict(image);
				if (!mounted) return;
				setState(() {
					_detections = results;
				});
			} catch (_) {
				// swallow frame errors to keep stream alive
			} finally {
				_isProcessingFrame = false;
			}
		});
		_isStreaming = true;
	}

	Future<void> _stopImageStream() async {
		final controller = _cameraController;
		if (controller == null) return;
		if (!_isStreaming) return;
		try {
			await controller.stopImageStream();
		} finally {
			_isStreaming = false;
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
				if (recognized.contains('scan book') || recognized.contains('read text')) {
					await _speech.stop();
					setState(() {
						_isListening = false;
					});
					await _onCapturePressed();
				}
			},
			localeId: 'en_US',
			onSoundLevelChange: null,
			listenFor: const Duration(seconds: 5),
		);
	}

	@override
	void dispose() {
		_stopImageStream();
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
								CustomPaint(
									painter: _DetectionsPainter(
										detections: _detections,
										screenSize: MediaQuery.of(context).size,
									),
								),
							],
						)
					else
						const SizedBox.shrink(),

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
											onPressed: _onCapturePressed,
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

	Future<void> _onCapturePressed() async {
		final controller = _cameraController;
		if (controller == null || !controller.value.isInitialized) return;
		try {
			if (controller.value.isStreamingImages) {
				await _stopImageStream();
			}
			await controller.setFlashMode(FlashMode.off);
			final XFile file = await controller.takePicture();

			final String text = await _ocrService.recognizeTextFromImage(file.path);
			if (!mounted) return;
			await showDialog<void>(
				context: context,
				builder: (context) {
					return AlertDialog(
						title: const Text('Recognized Text'),
						content: SingleChildScrollView(child: Text(text.isEmpty ? '(No text found)' : text)),
						actions: [
							TextButton(
								onPressed: () => Navigator.of(context).pop(),
								child: const Text('Close'),
							),
						],
					);
				},
			);
		} catch (e) {
			if (!mounted) return;
			ScaffoldMessenger.of(context).showSnackBar(
				SnackBar(content: Text('Capture failed: $e')),
			);
		} finally {
			// Ensure the stream restarts even if an error occurs above
			if (!controller.value.isStreamingImages) {
				await _startImageStream();
			}
		}
	}
}

class _DetectionsPainter extends CustomPainter {
	final List<Map<String, dynamic>> detections;
	final Size screenSize;

	_DetectionsPainter({
		required this.detections,
		required this.screenSize,
	});

	@override
	void paint(Canvas canvas, Size size) {
		final paint = Paint()
			..style = PaintingStyle.stroke
			..strokeWidth = 2.0
			..color = Colors.greenAccent;

		for (final d in detections) {
			final List box = d['box'] as List;
			if (box.length != 4) continue;
			double x1 = (box[0] as num).toDouble();
			double y1 = (box[1] as num).toDouble();
			double x2 = (box[2] as num).toDouble();
			double y2 = (box[3] as num).toDouble();

			// The model coordinates are in 640x640 space. Scale to current canvas size.
			const double modelW = 640.0;
			const double modelH = 640.0;
			final double scaleX = size.width / modelW;
			final double scaleY = size.height / modelH;

			final rect = Rect.fromLTRB(x1 * scaleX, y1 * scaleY, x2 * scaleX, y2 * scaleY);
			canvas.drawRect(rect, paint);
		}
	}

	@override
	bool shouldRepaint(covariant _DetectionsPainter oldDelegate) {
		return oldDelegate.detections != detections || oldDelegate.screenSize != screenSize;
	}
} 