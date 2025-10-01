import 'dart:io';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:google_generative_ai/google_generative_ai.dart';

class GeminiService {
  late final GenerativeModel _model;
  bool _isInitialized = false;

  Future<void> initialize() async {
    // Load the API key from the .env file
    await dotenv.load(fileName: ".env");
    final apiKey = dotenv.env['GEMINI_API_KEY'];
    if (apiKey == null) {
      print('Failed to load API key.');
      return;
    }
    
    // Configure the model
    _model = GenerativeModel(model: 'gemini-flash-latest', apiKey: apiKey);
    _isInitialized = true;
  }

  Future<String> describeImage(String imagePath) async {
    if (!_isInitialized) {
      return "Error: Gemini service not initialized.";
    }

    try {
      final imageBytes = await File(imagePath).readAsBytes();
      
      // This is the prompt that instructs the AI.
      const prompt =
          "You are an expert accessibility assistant for visually impaired users. "
          "Describe everything in this image in a clear, concise, and helpful way. "
          "Focus on the most important objects, text, and the overall scene context."
          "Do not go into too much detail, just give a brief description of the image."
          "The output should be in a single paragraph";
      
      final content = [
        Content.multi([
          TextPart(prompt),
          DataPart('image/jpeg', imageBytes),
        ])
      ];

      final response = await _model.generateContent(content);
      
      return response.text ?? "I'm sorry, I couldn't describe this image.";
      } on GenerativeAIException catch (e) {
        // NEW: Specifically catch Gemini exceptions
        print("Gemini API Error: ${e.message}");
        // NEW: Check if the error is due to the server being busy
        if (e.message.contains('overloaded') || e.message.contains('503')) {
          return "The AI model is currently busy. Please try again in a moment.";
        }
        return "An error occurred while analyzing the image.";
      } catch (e) {
        // Catch any other general errors
        print("Error describing image with Gemini: $e");
        return "An error occurred while analyzing the image.";
    }
  }
}