# Walter

## What does this bot do?

You can talk to Walter by just speaking to him and he will respond with speech

## How to setup?
First make sure you have the following things installed:

- [Python 3.10 or 3.12 (other version untested)](https://www.python.org/)
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

Then you might get a 401 error. That's because you need to modify the `config.json` the discord token to your's and then run

```pwsh
.\start.ps1
```
again.

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
pip3 install -r requirements.txt
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