import 'package:shared_preferences/shared_preferences.dart';

class SettingsService {
  // Define keys for each setting to avoid typos.
  static const String offlineModeKey = 'offline_mode';
  static const String manageLightingKey = 'manage_lighting';
  static const String speechSpeedKey = 'speech_speed';
  static const String voiceKey = 'voice_selection';

  // --- Loading Methods ---

  Future<bool> getOfflineMode() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(offlineModeKey) ?? false; // Default to false
  }

  Future<bool> getManageLighting() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(manageLightingKey) ?? true; // Default to true
  }

  Future<double> getSpeechSpeed() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getDouble(speechSpeedKey) ?? 1.0; // Default to normal speed
  }

  Future<Map<String, String>?> getVoice() async {
    final prefs = await SharedPreferences.getInstance();
    final String? voiceName = prefs.getString('${voiceKey}_name');
    final String? voiceLocale = prefs.getString('${voiceKey}_locale');
    if (voiceName != null && voiceLocale != null) {
      return {'name': voiceName, 'locale': voiceLocale};
    }
    return null;
  }

  // --- Saving Methods ---

  Future<void> saveOfflineMode(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(offlineModeKey, value);
  }

  Future<void> saveManageLighting(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(manageLightingKey, value);
  }

  Future<void> saveSpeechSpeed(double value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setDouble(speechSpeedKey, value);
  }

  Future<void> saveVoice(Map<String, String> voice) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('${voiceKey}_name', voice['name']!);
    await prefs.setString('${voiceKey}_locale', voice['locale']!);
  }
}