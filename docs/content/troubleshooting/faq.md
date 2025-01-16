# Frequently Asked Questions

## General Questions

### Q: What Python versions are supported?
A: AgentNeo supports Python 3.9 and above.

### Q: Can I use AgentNeo with any LLM provider?
A: Yes, AgentNeo supports various providers including OpenAI, Anthropic, and others through LiteLLM.

### Q: How do I update AgentNeo?
A: Use pip to update to the latest version:
```bash
pip install --upgrade agentneo
```

## Privacy & Security

### Q: Is my data safe with AgentNeo? What about privacy?
A: Yes, your data is completely safe. AgentNeo operates entirely on your local system and does not transmit any data externally. All data, including traces and logs, are stored locally on your machine. We do not collect any telemetry data or usage statistics - your privacy is our priority.

The only external communications are the API calls you explicitly configure to your chosen LLM provider (like OpenAI or Anthropic). These calls are made using your own API keys and follow the privacy policies of those providers.

### Q: Where is my data stored?
A: All data is stored locally on your system:
- Trace data: `{agentneo-package-path}/ui/dist/trace_data.db`
- Logs: In your system's standard logging directory
- Configuration: In your project directory

You have complete control over this data and can delete it at any time.

// ... existing questions ...

## Privacy & Security

### Q: Is my data safe with AgentNeo? What about privacy?
A: Yes, your data is completely safe. AgentNeo operates entirely on your local system and does not transmit any data externally. All data, including traces and logs, are stored locally on your machine. We do not collect any telemetry data or usage statistics - your privacy is our priority.

The only external communications are the API calls you explicitly configure to your chosen LLM provider (like OpenAI or Anthropic). These calls are made using your own API keys and follow the privacy policies of those providers.

### Q: Where is my data stored?
A: All data is stored locally on your system:
- Trace data: `{agentneo-package-path}/ui/dist/trace_data.db`
- Logs: In your system's standard logging directory
- Configuration: In your project directory

You have complete control over this data and can delete it at any time.

## Integration Questions

### Q: Can I use AgentNeo with other Agentic Framworks?
A: Yes, see the [Developer Guide](../developer-guide/advanced-usage.md) for more information.

## Customization Questions

### Q: How do I open the dashboard on a different port?
A: You can specify a custom port when launching the dashboard:
```python
neo_session.launch_dashboard(port=8080)
```

### Q: How do I enable the dark theme?
A: The dark theme can be enabled through the settings menu by selecting the Dark Theme option or by using the System Theme setting that switches automatically based on your system preferences.

### Q: Can I switch back to the light theme?
A: Yes, you can toggle between the Light and Dark themes at any time in the settings. Simply select Light Theme from the options.

### Q: Does the dark theme affect all parts of the application?
A: Yes, the dark theme applies to the entire UI, including navigation, background, and text, to ensure a consistent experience in low-light environments.