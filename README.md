# Walter

## What does this bot do?

You can talk to Walter by just speaking to him and he will respond with speech. This project uses [Ollama](https://ollama.com/) for speech generation. [Whisper](https://openai.com/index/whisper/) for transcription and [TTS](https://github.com/coqui-ai/TTS) for talking. **Warning this project is experimental and should not be used for production. `I made this as a weekend project`**

## How to setup?
First make sure you have the following things installed:

- [Python 3.12 (other versions untested, but might work)](https://www.python.org/)
- [Ollama](https://ollama.com/)

### Automatic setup

If you are on Mac or Linux run

```bash
./start.sh
```

Then you might get a 401 error. That's because you need to modify the `config.json` the discord token to your's and then run

```bash
./start.sh
```
again.

If you're on windows run

```pwsh
.\start.ps1
```
In powershell

Then you might get a 401 error. That's because you need to modify the `config.json` the discord token to your's and then run `start.ps1` again

```pwsh
.\start.ps1
```


For NVidia GPU support on Windows you might need to install Pytorch through the following command

```bash
.venv\Scripts\Activate.ps1; pip uninstall torch torchvision torchaudio; pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache
```

### Manual setup
1. Create a venv

```
python3 -m venv .venv
```

2. Activate the venv (On Linux and Mac run):
```bash
source .venv/bin/activate
```

(On Windows run):

```pwsh
.venv\Scripts\Activate.ps1
```

3. Install dependencies
```bash
pip3 install wheel && pip3 install -r requirements.txt
```

4. Create config json

```bash
cp example.config.json config.json
```
And modify the Discord token

Then run

```bash
python3 main.py
```

## Commands

If you say `WIPE WALTER MEMORY` or `TRUNCATE WALTER` it will get rid of his memory

## Configuration

All the configuration you can easily change in inside of `config.json`

* `ollamaModel` - The model you want to use for generation. [Here you can find all the models](https://ollama.com/library). You can download a llama3.2 with ```ollama pull llama3.2```.

* `ollamaUrl` - The base url of Ollama. If you are running this locally, you don't have to change anything.

* `whisperModel` - The model of voice detection. You can change it by changing the name. [Here is a reference to the available models](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages).

* `maxStoredMessages` - The max amount of messages stored in `conversation.json`. To remove the limit set the value to 0.

* `keepAudioFiles` - It will delete the wav files in `recordings/` if set to `false`. If you want to keep the files set it to `true`.

* `gracePeriodInMS` - This is the delay before it stops recording when it's silent. The higher the grace period the longer it takes to process the audio. Unit: Milliseconds.

* `respondsTo` - This is all the words or sentences the AI will respond to.

* `alwaysUseDefaultMic` - This is for Windows and Mac users and even Linux users if you mic configuration is set properly. Otherwise set this to `false` and it will prompt you to select your mic.