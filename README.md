# Engineer Repository

A collection of engineering projects and research, focusing on autonomous AI systems and automation tools.

## Projects

### 1. Video Automation System
**Location:** `/video-automation/`

An advanced automated video generation pipeline that transforms text topics into high-quality educational explainer videos. This system leverages modern AI APIs, media processing, and React-based video composition to create YouTube-ready content.

**Key Features:**
- Automated script generation using GPT-4
- AI-powered image generation (DALL-E 3, SDXL)
- Professional narration with ElevenLabs TTS
- React-based video composition with Remotion
- GPU-accelerated rendering with FFmpeg

**Performance Metrics:**
- Pipeline Time: 30-45 minutes per 5-8 minute video
- Cost: $2-12 per video
- Automation Rate: 90-95% success rate
- Output: 1080p YouTube-ready videos

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
- **AI/ML:** GPT-4, DALL-E 3, Stable Diffusion XL, ElevenLabs TTS
- **Media Processing:** FFmpeg, Remotion, Whisper
- **Infrastructure:** PostgreSQL, Redis, Docker

## Documentation

- [Video Automation Reverse Engineering Report](video-automation/docs/REVERSE_ENGINEERING_REPORT.md)
- [Production Guide](video-automation/docs/PRODUCTION_GUIDE_V2.md)
- [Video Automation README](video-automation/README.md)

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