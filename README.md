# Frigchat

Frigchat is a project that utilizes a dataset obtained from a personal Discord group chat. The dataset was exported using DiscordChatExporter.

## Dataset

The dataset consists of chat logs from a personal Discord group chat. The logs were exported using the DiscordChatExporter tool. The dataset is available on Hugging Face: [eekay/kissy](https://huggingface.co/datasets/eekay/kissy).

## Tools Used

- **DiscordChatExporter**: Used to export the chat logs from Discord. This tool allows you to download your Discord chat history in a readable format.
- **Transformers**: A library from Hugging Face used for natural language processing tasks.
- **Hugging Face Datasets**: Used for handling and processing the dataset.
- **Axolotl**: Used for training the model on a single GPU on RunPod.
- **RunPod**: Finetuned the lora on a single GPU runpod, and use serverless to serve the model.

## Usage

1. **Export Chat Logs**: Export your Discord chat logs using DiscordChatExporter.
   - Download and install DiscordChatExporter from its [GitHub repository](https://github.com/Tyrrrz/DiscordChatExporter).
   - Follow the instructions to export your chat logs.
2. **Process Data**: Use Python scripts to process the exported dataset.
   - Install the required Python libraries: `transformers`, `datasets`, `axolotl`, and `runpod`.
   - Run the `tokenize_chat.py` script to tokenize and prepare the dataset.
3. **Train Model**: Use the `frig_tune.py` script to fine-tune the model on the dataset using Axolotl.
   - The model was trained on a single GPU on RunPod.
   - The fine-tuned model is available on Hugging Face: [eekay/mistral7B_kissy](https://huggingface.co/eekay/mistral7B_kissy).
4. **Inference**: Use the `inference.py` script to generate text completions based on the chat data.
5. **Run Inference on RunPod**: Use the `call.py` and `endpoint.py` scripts to run the model inference on RunPod.

## License

This project is for personal use only. Please ensure you have permission to use any data that you export from Discord.
