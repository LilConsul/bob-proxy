# bob-proxy

A local HMAC-signing proxy for Bob's LiteLLM backend. Lets any OpenAI or Anthropic-compatible client use Bob's models without custom signing or token management.

## Requirements

- Docker & Docker Compose

## Setup

```bash
cp .env.sample .env
# Edit .env and set BOB_API_KEY=sk-your-key
docker compose up -d
```

Proxy listens on `http://localhost:8989`.

To view logs:
```bash
docker compose logs -f
```

To stop:
```bash
docker compose down
```

## Client configuration

### OpenCode / any OpenAI-compatible client

```json
{
  "baseURL": "http://localhost:8989/v1",
  "apiKey": "proxy"
}
```

### JetBrains AI Assistant

In **Settings → Tools → AI Assistant → Providers & API keys**:

| Field | Value |
|---|---|
| Provider | OpenAI-compatible |
| URL | `http://localhost:8989/v1` |
| API Key | any value |

Then under **Model Assignment**, pick any model from the list (e.g. `sonnet-4.5` for Core features).

### Claude Code (Anthropic mode)

```bash
ANTHROPIC_BASE_URL=http://localhost:8989/anthropic ANTHROPIC_API_KEY=proxy CLAUDE_CODE_AUTO_MODE_SERVER=0 claude --model premium
```

### Claude Code (Bedrock mode)

**Windows** — set once, then just run `claude`:
```powershell
[System.Environment]::SetEnvironmentVariable("CLAUDE_CODE_USE_BEDROCK","1","User")
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_BEDROCK_BASE_URL","http://localhost:8989/bedrock","User")
[System.Environment]::SetEnvironmentVariable("CLAUDE_CODE_SKIP_BEDROCK_AUTH","1","User")
[System.Environment]::SetEnvironmentVariable("CLAUDE_CODE_DISABLE_BEDROCK_CONTENT_TYPE_GUARD","1","User")
[System.Environment]::SetEnvironmentVariable("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC","1","User")
[System.Environment]::SetEnvironmentVariable("AWS_ACCESS_KEY_ID","dummy","User")
[System.Environment]::SetEnvironmentVariable("AWS_SECRET_ACCESS_KEY","dummy","User")
[System.Environment]::SetEnvironmentVariable("AWS_REGION","us-east-1","User")
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_DEFAULT_MODEL","premium","User")
[System.Environment]::SetEnvironmentVariable("CLAUDE_CODE_AUTO_MODE_SERVER","0","User")
```

**Linux/macOS** — inline per session:
```bash
CLAUDE_CODE_USE_BEDROCK=1 \
ANTHROPIC_BEDROCK_BASE_URL=http://localhost:8989/bedrock \
CLAUDE_CODE_SKIP_BEDROCK_AUTH=1 \
CLAUDE_CODE_DISABLE_BEDROCK_CONTENT_TYPE_GUARD=1 \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
AWS_ACCESS_KEY_ID=dummy \
AWS_SECRET_ACCESS_KEY=dummy \
AWS_REGION=us-east-1 \
ANTHROPIC_DEFAULT_MODEL=premium \
CLAUDE_CODE_AUTO_MODE_SERVER=0 \
claude
```

> **`CLAUDE_CODE_AUTO_MODE_SERVER=0`** — suppresses the _"this session isn't eligible for auto mode's no-charge classifier requests"_ notice. This proxy translates between Anthropic and OpenAI formats, so server-side classifier checks are never available here. Set this once and it won't appear again.

### GitHub Copilot (VS Code)

```json
{
  "github.copilot.chat.models": [
    {
      "name": "Bob",
      "vendor": "customoai",
      "apiKey": "${input:chat.lm.secret.<your-secret-id>}",
      "models": [
        {
          "id": "premium",
          "name": "Bob Premium",
          "url": "http://localhost:8989/v1",
          "toolCalling": true,
          "vision": true,
          "maxInputTokens": 128000,
          "maxOutputTokens": 16000
        }
      ]
    }
  ]
}
```

### CUGA

```
AGENT_SETTING_CONFIG="settings.openai.toml"
OPENAI_API_KEY=any-value
OPENAI_BASE_URL=http://localhost:8989/v1
MODEL_NAME=premium
```

## Available models

| Model | Description |
|---|---|
| `premium` | Default — best available model |
| `premium-fallback` | Fallback if premium is unavailable |
| `premium-shell` | Optimised for shell/terminal use |
| `premium-ide` | Optimised for IDE use |
| `sonnet-4.5` | Claude Sonnet 4.5 |
| `sonnet-4.5-global-ibm-only` | Claude Sonnet 4.5 (IBM-only global) |
| `sonnet-4.6` | Claude Sonnet 4.6 |
| `haiku-4.5` | Claude Haiku 4.5 |
| `explorer` | Explorer model |
| `mistral-medium-3.1` | Mistral Medium 3.1 |
| `granite-8b-code-instruct` | Granite 8B Code |
| `codestral-2508` | Codestral 2508 |
| `openai/gpt-oss-20b` | GPT OSS 20B |
| `gpt-2026-5.4` | GPT 2026 5.4 |
