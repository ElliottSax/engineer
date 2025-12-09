# Engineer Repository

A collection of engineering projects and research, focusing on autonomous AI systems and automation tools.

**Featured Project:** Hybrid AI Video Generation System (combining insights from Wondershare Filmora + elliottsax/once)

## Projects

### 1. Enhanced Video Automation System
**Location:** `/video-automation/`

An advanced automated video generation pipeline that combines insights from **reverse engineering Wondershare Filmora** with the **"once" automated video system**. Transforms text topics into high-quality educational explainer videos using multiple AI models.

**Key Features:**
- **Multi-Model AI System** (inspired by Filmora):
  - Google Veo 3.0/3.1 for text/image-to-video with auto-audio
  - OpenAI Sora 2 for high-quality image-to-video
  - Kelin model for cost-effective generation
  - Dynamic model selection based on scene importance
- **Automated script generation** using GPT-4 (from "once")
- **AI-powered image generation** (DALL-E 3, SDXL)
- **Professional narration** with ElevenLabs TTS
- **React-based video composition** with Remotion
- **GPU-accelerated rendering** with FFmpeg

**Performance Metrics:**
- Pipeline Time: 30-45 minutes per 5-8 minute video
- Cost: $2-3 (economy) to $12-15 (premium) per video
- Automation Rate: 90-95% success rate
- Output: 720p-1080p YouTube-ready videos with auto-generated audio

## Repository Structure

```
engineer/
├── video-automation/          # Automated YouTube video generation system
│   ├── docs/                 # Technical documentation and analysis
│   ├── src/                  # Python source code for pipeline
│   ├── remotion/            # React-based video composition
│   └── config/              # Configuration and settings
└── README.md                # This file
```

## Getting Started

Each project contains its own README with specific setup instructions. Navigate to the project directory for detailed documentation.

### Quick Start for Video Automation

```bash
cd video-automation
# Follow the setup instructions in video-automation/README.md
```

## Technologies Used

- **Languages:** Python (88.4%), TypeScript (7.9%), Shell (3.1%)
- **AI/ML Models:**
  - **Text-to-Video:** Google Veo 3.0/3.1, Kelin (proprietary)
  - **Image-to-Video:** OpenAI Sora 2, Google Veo 3.1, Standard models
  - **Text Generation:** GPT-4
  - **Image Generation:** DALL-E 3, Stable Diffusion XL
  - **Speech Synthesis:** ElevenLabs TTS
  - **Speech Recognition:** OpenAI Whisper
- **Media Processing:** FFmpeg (GPU-accelerated), Remotion, spaCy NLP
- **Infrastructure:** PostgreSQL, Redis, Docker, Temporal.io

## Documentation

### Reverse Engineering Reports
- [Filmora AI System Reverse Engineering](video-automation/docs/FILMORA_REVERSE_ENGINEERING.md) - Analysis of Wondershare Filmora's multi-model AI architecture
- [Once System Analysis](video-automation/docs/REVERSE_ENGINEERING_REPORT.md) - Original automated video pipeline analysis

### Technical Guides
- [Production Guide](video-automation/docs/PRODUCTION_GUIDE_V2.md) - Complete system architecture and production deployment
- [Video Automation README](video-automation/README.md) - Setup and usage instructions

### Key Innovations from Combining Both Systems
1. **JSON-based AI model configuration** (from Filmora) + **Python orchestration** (from Once)
2. **Multi-model routing** with dynamic selection based on scene importance
3. **Veo 3.0 auto-audio generation** combined with **ElevenLabs narration**
4. **Cost optimization** through intelligent model selection (economy/standard/premium tiers)
5. **Enhanced quality** with Sora 2 for image-to-video transitions

## Contributing

This repository contains research and development projects. For contributions, please:
1. Review the project-specific documentation
2. Follow the existing code style and patterns
3. Test your changes thoroughly
4. Submit detailed pull requests

## License

See individual project directories for specific licensing information.

## Contact

Repository maintained by Elliott Sax
GitHub: [@elliottsax](https://github.com/elliottsax)