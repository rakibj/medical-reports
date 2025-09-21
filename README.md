# Report Communication System

## Description

The Report Communication System is a robust application designed to handle the import, classification, and chatting of various reports. It leverages advanced AI models for text classification and generation, allowing users to interact with and derive insights from their reports seamlessly.

## Features

- **Import Reports**: Users can import new reports for analysis.
- **Text Classification**: Classify the content of reports using NLP models.
- **Chat Interface**: Communicate with the system to get responses and insights based on the reports.
- **Report Management**: Efficiently manage and store reports for future reference.

## Project Structure

```plaintext
.
├── app/                  # Main application folder
│   ├── runner/           # Contains the runner scripts for the app
│   │   ├── app.py        # Main entry point for the application
│   │   └── workflow.py   # Workflow management for processing reports
│   ├── src/              # Source code for core functionality
│   │   ├── chat_ai.py    # Chat interface logic
│   │   ├── cloud_storage.py # Logic for cloud storage integration
│   │   ├── ocr_processor.py # OCR processing for extracting text from images
│   │   ├── report_repository.py # Database integration for report storage
│   │   ├── report_service.py    # Business logic for report management
│   │   ├── text_embedder.py      # Text embedding logic for enhanced text processing
│   │   └── utils/              # Utility functions
│   │       └── files.py         # Helper functions related to file handling
├── basics/             # Basic application scripts (if needed)
│   ├── .gradio/        # Gradio related setup if applicable
│   ├── langgraph_basic_tool_memory.ipynb
│   ├── langgraph_mcp_agentworkflow.ipynb
│   ├── langgraph_stroutput.ipynb
│   └── memory.db       # SQLite database for storing memory
├── resources/          # Resources such as sample reports and images
│   └── sample_report.jpg
├── pyproject.toml      # Python project dependencies and configuration
└── .gitignore          # Git ignore file
```

## Installation

Follow these steps to set up the project locally:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```bash
   cd report-communication-system
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the application, run the following command:

```bash
python app/runner/app.py
```

Open your browser and go to `http://127.0.0.1:5000` to access the application.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create your feature branch:
   ```bash
   git checkout -b feature-branch-name
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch-name
   ```
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.