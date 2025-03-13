# Image Processing with Vision LLM

This repository contains a Python script that processes images using a Vision Language Model (VLLM). It supports two backends—**Ollama** and **OpenAI**—to generate image descriptions, captions, keywords, and other analyses based on pre-defined prompts.

## Repository Structure

```
/src
  ├── process_image.py    # Main script for processing images (named process_images.py in the repo)
  └── image_prompts.py    # Contains a dictionary of prompts for various processing modes

requirements.txt          # List of Python dependencies required for the project
```

## Overview

The main script reads images from a specified directory, processes each image by:
- Preprocessing (resizing and JPEG encoding)
- Sending the image along with prompts (from `image_prompts.py`) to the selected Vision LLM backend
- Saving the output in both text and JSON formats

The script supports downscaling images, repeating the prompt processing, and even optional translation (currently commented out) of the results.

## Setup & Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/michal-hradis/image-descriptions.git
   cd image-descriptions
   ```

2. **Create a Virtual Environment (Optional but recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script from the command line with the required arguments. For example:

```bash
python src/process_image.py \
  --image-dir path/to/images \
  --backend ollama \
  --url http://your-ollama-endpoint \
  --model your-model-name \
  --prompt orbis_01 \
  --resolution 256 \
  --output-path path/to/output \
  --repeat 1
```

### Command-Line Arguments

- **--image-dir**: Directory containing the images to process.
- **--backend**: Specify which backend to use (`ollama` or `openai`).
- **--url**: (Required for Ollama) API endpoint URL for the Ollama backend.
- **--api-key**: (Required for OpenAI) Your OpenAI API key.
- **--model**: Name of the model to use.
- **--resolution**: Resolution to which images will be resized (default is 256).
- **--output-path**: Directory where output files (TXT and JSON) will be saved (default is current directory).
- **--prompt**: Key to choose the prompt from the prompt dictionary in `image_prompts.py` (default is `aisee_01`).
- **--repeat**: Number of times to process each prompt (default is 1).
- **--only-downscale**: Flag to only downscale the image without further processing.
- **--translate-cz**: Flag to translate the output to Czech (this feature is available but currently commented out).

## Files Description

- **process_image.py** (or *process_images.py*):
  - Contains the main script logic for image processing.
  - Handles argument parsing, image pre-processing, invoking the Vision LLM backend, and writing the output.
  - Defines two classes for processing via the Ollama and OpenAI backends.

- **image_prompts.py**:
  - Stores a dictionary (`all_prompts`) with different sets of prompts.
  - Prompts are divided into modes (e.g., `orbis_01`, `aisee_01`) that provide instructions for generating keywords, captions, and descriptions.

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the BSD 2-Clause License.
