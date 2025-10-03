import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:visionai/services/settings_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final SettingsService _settingsService = SettingsService();
  final FlutterTts _flutterTts = FlutterTts();

  bool _isOfflineMode = false;
  bool _manageLighting = true;
  double _speechSpeed = 1.0;
  
  List<Map<String, String>> _voices = [];
  // MODIFIED: The state now holds the unique composite key (e.g., "en-us-voice#en-US")
  String? _selectedVoiceKey;

  @override
  void initState() {
    super.initState();
    _loadSettings();
    _getVoices();
  }

  // NEW HELPER: Creates a unique key from a voice map.
  String _getVoiceKey(Map<String, String> voice) {
    return '${voice['name']}#${voice['locale']}';
  }

  Future<void> _loadSettings() async {
    _isOfflineMode = await _settingsService.getOfflineMode();
    _manageLighting = await _settingsService.getManageLighting();
    _speechSpeed = await _settingsService.getSpeechSpeed();
    
    // MODIFIED: Load the saved voice and construct the composite key.
    final voiceMap = await _settingsService.getVoice();
    if (voiceMap != null) {
      _selectedVoiceKey = _getVoiceKey(voiceMap);
    }
    
    setState(() {});
  }

  Future<void> _getVoices() async {
    final dynamic voices = await _flutterTts.getVoices;
    
    print("----------- RAW VOICE DATA FROM DEVICE -----------");
    print(voices);
    print("-------------------------------------------------");
    
    if (voices != null) {
      final filteredVoices = (voices as List)
          .where((v) => (v['locale'] as String).toLowerCase().contains('en'))
          .map((v) => {'name': v['name'] as String, 'locale': v['locale'] as String})
          .toList();
      
      setState(() {
        _voices = List<Map<String, String>>.from(filteredVoices);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
      ),
      backgroundColor: Colors.grey[900],
      body: ListView(
        children: [
          // ... other settings (SwitchListTiles, Slider) remain the same ...
          SwitchListTile(
            title: const Text('Offline Mode', style: TextStyle(color: Colors.white)),
            subtitle: const Text('Use only on-device models (faster, less accurate)', style: TextStyle(color: Colors.white70)),
            value: _isOfflineMode,
            onChanged: (bool value) {
              setState(() => _isOfflineMode = value);
              _settingsService.saveOfflineMode(value);
            },
            secondary: const Icon(Icons.wifi_off, color: Colors.white),
          ),
          SwitchListTile(
            title: const Text('Manage Lighting', style: TextStyle(color: Colors.white)),
            subtitle: const Text('Automatically use flash in low light', style: TextStyle(color: Colors.white70)),
            value: _manageLighting,
            onChanged: (bool value) {
              setState(() => _manageLighting = value);
              _settingsService.saveManageLighting(value);
            },
            secondary: const Icon(Icons.highlight, color: Colors.white),
          ),
          ListTile(
            leading: const Icon(Icons.speed, color: Colors.white),
            title: const Text('Speech Speed', style: TextStyle(color: Colors.white)),
            subtitle: Slider(
              value: _speechSpeed,
              min: 0.5,
              max: 1.0,
              divisions: 10,
              label: _speechSpeed.toStringAsFixed(1),
              onChanged: (double value) {
                setState(() => _speechSpeed = value);
              },
              onChangeEnd: (double value) {
                _settingsService.saveSpeechSpeed(value);
              },
            ),
          ),
          
          // MODIFIED: The entire DropdownButton is updated to use the unique composite key.
          if (_voices.isNotEmpty)
            ListTile(
              leading: const Icon(Icons.record_voice_over, color: Colors.white),
              title: const Text('Voice', style: TextStyle(color: Colors.white)),
              trailing: DropdownButton<String>(
                value: _selectedVoiceKey,
                hint: const Text('Default', style: TextStyle(color: Colors.white70)),
                dropdownColor: Colors.grey[800],
                onChanged: (String? newKey) {
                  if (newKey != null) {
                    setState(() {
                      _selectedVoiceKey = newKey;
                      // Find the full voice map that matches the selected key and save it.
                      final selectedVoiceMap = _voices.firstWhere((v) => _getVoiceKey(v) == newKey);
                      _settingsService.saveVoice(selectedVoiceMap);
                    });
                  }
                },
                items: _voices.map<DropdownMenuItem<String>>((Map<String, String> voice) {
                  final uniqueKey = _getVoiceKey(voice);
                  return DropdownMenuItem<String>(
                    value: uniqueKey, // Value is the unique composite key.
                    child: Text(
                      voice['name']!.length > 25 ? '${voice['name']!.substring(0, 22)}...' : voice['name']!,
                      style: const TextStyle(color: Colors.white),
                    ),
                  );
                }).toList(),
              ),
            ),
            
          const Divider(color: Colors.white24),
          ListTile(
            title: const Text('About Us', style: TextStyle(color: Colors.white)),
            leading: const Icon(Icons.info_outline, color: Colors.white),
            onTap: () {
              // TODO: Navigate to an "About Us" page
            },
          ),
        ],
      ),
    );
  }
}