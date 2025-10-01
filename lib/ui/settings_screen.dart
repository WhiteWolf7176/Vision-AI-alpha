import 'package:flutter/material.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  // Placeholder state variables for the settings
  bool _isOfflineMode = false;
  bool _manageLighting = true;
  double _speechSpeed = 1.0;

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
          SwitchListTile(
            title: const Text('Offline Mode', style: TextStyle(color: Colors.white)),
            subtitle: const Text('Use only on-device models', style: TextStyle(color: Colors.white70)),
            value: _isOfflineMode,
            onChanged: (bool value) {
              setState(() {
                _isOfflineMode = value;
                // TODO: Save this setting
              });
            },
            secondary: const Icon(Icons.wifi_off, color: Colors.white),
          ),
          SwitchListTile(
            title: const Text('Manage Lighting', style: TextStyle(color: Colors.white)),
            subtitle: const Text('Automatically use flash in low light', style: TextStyle(color: Colors.white70)),
            value: _manageLighting,
            onChanged: (bool value) {
              setState(() {
                _manageLighting = value;
                // TODO: Save this setting
              });
            },
            secondary: const Icon(Icons.highlight, color: Colors.white),
          ),
          ListTile(
            title: const Text('Speech Speed', style: TextStyle(color: Colors.white)),
            subtitle: Slider(
              value: _speechSpeed,
              min: 0.5,
              max: 2.0,
              divisions: 3,
              label: _speechSpeed.toStringAsFixed(1),
              onChanged: (double value) {
                setState(() {
                  _speechSpeed = value;
                  // TODO: Save this setting and update TTS engine
                });
              },
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