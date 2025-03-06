import glob
import argparse
import cv2
import json
import os

from ollama import Client
from openai import OpenAI

# Dictionary of prompts for different processing modes
all_prompts =  {
    "orbis_01":
        {
            "keywords": (
                'Write list of keywords that describe the image. The keywords could describe for example the type '
                'of the image itself, objects, actions, location, names. Write only a list of the keywords  '
                'separated by commas.'
            ),
            "caption": (
                'Write five possible captions for the image which could for example be used in a book, magazine, '
                'webpage, newspaper or social media post. Write only a list of the captions with each caption on '
                'a separate text line.'
            ),
            "description": (
                'Describe the image in a few sentences. Describe the image itsef (photo, drawing, graphs), '
                'possibly it\'s style and historic period. Describe the image content, objects, actions, '
                'location, names.'
            ),
        },
    "aisee_01":
        {
            "person": (
                'Describe the person in the image using as a plain list of attributes. '
                'List the person\'s general appearance, age, ethnicity, hairstyle, body type, ... '
                'Provide just comma separated list of attributes. Do not write any additional text or comments.'
            ),
            "clothing": (
                'Describe each piece of clothing the person is wearing, all accessories, any carried items, '
                'and object the person is interacting with such as a phone, book, or computer, bicicles, '
                'suitcases, stroller and similar. Provide just s plain list of attributes such as: '
                'Provide just comma separated list of items. Do not write any additional text or comments.'
            ),
            "search_queries": (
                'Write a list of 10 search queries that a policeman could used to find this specific person '
                'from the image in a large database. Make the queries distinctive and specific for this person. '
                'Focus only on the person, clothing, accessories, and objects in the '
                'person\'s possession. Write only a list of the search queries separated by commas.'
                'Do not write any additional text or comments.'
            ),
        }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Process images with a Vision LLM using either Ollama or OpenAI.")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory with images to process.")
    parser.add_argument("--backend", type=str, required=True, choices=["ollama", "openai"],
                        help="Backend to use: 'ollama' or 'openai'.")
    parser.add_argument("--url", type=str, help="URL for the Ollama API endpoint (required for ollama backend).")
    parser.add_argument("--api-key", type=str, help="API key for the OpenAI API (required for openai backend).")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Resolution to resize the images to before processing.")
    parser.add_argument("--output-path", type=str, default="./", help="Path to the output directory.")
    parser.add_argument("--prompt", type=str, default="aisee_01", help="Prompt key to use from the prompt dictionary.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat each prompt.")
    parser.add_argument("--only-downscale", action="store_true", help="Only downscale the image without processing.")
    parser.add_argument("--translate-cz", action="store_true", help="Translate the output to Czech.")
    return parser.parse_args()


def prepare_image(image_path, resolution, only_downscale):
    """Load and preprocess the image (resize and encode as JPEG)."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}. Skipping.")
        return None

    # Downscale if required
    if only_downscale and (image.shape[0] > resolution or image.shape[1] > resolution):
        scale = min(resolution / image.shape[0], resolution / image.shape[1])
        image = cv2.resize(image, (int(scale * image.shape[1]), int(scale * image.shape[0])), interpolation=cv2.INTER_AREA)
    elif not only_downscale:
        scale = min(resolution / image.shape[0], resolution / image.shape[1])
        image = cv2.resize(image, (int(scale * image.shape[1]), int(scale * image.shape[0])))

    # Skip images that are too small.
    if image.shape[0] < 48 or image.shape[1] < 48:
        print(f"Image {image_path} is too small. Skipping.")
        return None

    # Encode the image as JPEG bytes.
    encoded_image = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tobytes()
    return encoded_image


class VLLMProcessor:
    """Abstract class for a Vision LLM processor."""
    def process_image(self, model, prompt, base64_encoded_image, options, system_prompt=None):
        raise NotImplementedError("Subclasses must implement this method.")


class OllamaProcessor(VLLMProcessor):
    def __init__(self, host):
        self.client = Client(host=host)

    def process_image(self, model, prompt, base64_encoded_image, options, system_prompt=None):
        messages = [{
            "role": "user",
            "content": prompt,
            "images": [base64_encoded_image]
        }]
        response = self.client.chat(
            model=model,
            messages=messages,
            options=options
        )
        return response.message.content


class OpenAIProcessor(VLLMProcessor):
    def __init__(self, api_key):
        self.openai = OpenAI(api_key=api_key)

    def process_image(self, model, prompt, base64_encoded_image, options, system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image",
                    "image": {"url": f"data:image/jpeg;base64,{base64_encoded_image}"}
                }
                ]
        })

        response = self.openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=options.get("num_predict", 256)
        )

        # Return the content from the first message choice.
        return response["choices"][0]["message"]["content"]


def process_image(processor, encoded_image, prompts, model, repeat, translate_cz):
    """Send each prompt (and optional translation) to the processor for the image."""
    results = {}
    for key, prompt in prompts.items():
        for i in range(repeat):
            # Primary processing prompt
            options = {"num_predict": 256}
            content = processor.process_image(model, prompt, encoded_image, options)
            results[f"{key}_{i:02d}"] = content
            print(f"{key}_{i:02d}: {content}")

            # Optionally translate the response to Czech
            #if translate_cz:
            #    messages_translate = [
            #        {
            #            "role": "user",
            #            "content": prompt,
            #            "images": [encoded_image]
            #        },
            #        {
            #            "role": "system",
            #            "content": content
            #        },
            #        {
            #            "role": "user",
            #            "content": "Přelož předchozí odpověď do češtiny."
            #        }
            #    ]
            #    options_translate = {"num_predict": 1024}
            #    content_translate = processor.chat(model, messages_translate, options_translate)
            #    results[f"{key}_CZ"] = content_translate
            #    print(f"{key}_CZ: {content_translate}")
    return results


def output_results(results, output_path, image_path):
    """Save the results to a text file and a JSON file."""
    basename = os.path.basename(image_path)
    basename = os.path.splitext(basename)[0]

    # Write a text file.
    txt_file = os.path.join(output_path, f"{basename}.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        for key, value in results.items():
            f.write(f"\n-------------\n{key}:\n{value}\n\n")

    # Write a JSON file.
    json_file = os.path.join(output_path, f"{basename}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # Create the appropriate processor based on the chosen backend.
    if args.backend == "ollama":
        if not args.url:
            print("Error: --url is required when backend is 'ollama'.")
            return
        processor = OllamaProcessor(args.url)
    elif args.backend == "openai":
        if not args.api_key:
            print("Error: --api-key is required when backend is 'openai'.")
            return
        processor = OpenAIProcessor(args.api_key)
    else:
        print("Unsupported backend.")
        return

    if args.prompt not in all_prompts:
        print(f"Prompt {args.prompt} not found.")
        return

    prompts = all_prompts[args.prompt]

    # Process each image in the provided directory.
    image_files = glob.glob(os.path.join(args.image_dir, "*.*"))
    for image_path in image_files:
        encoded_image = prepare_image(image_path, args.resolution, args.only_downscale)
        if encoded_image is None:
            continue

        results = process_image(processor, encoded_image, prompts, args.model, args.repeat, args.translate_cz)
        output_results(results, args.output_path, image_path)
        print("-" * 40)


if __name__ == "__main__":
    main()
