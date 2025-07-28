# Final Year Legal AI Project

This project uses AI to analyze legal cases involving government employees (e.g., police) and predict possible outcomes before filing a case.

## Components

- **Whisper-Tamil**: Converts Tamil audio to text
- **IndicBERT**: Embeds Tamil legal text for processing
- **InLegalBERT**: Analyzes and summarizes English legal text

## Files

- `whisper_test.py`: Audio-to-text test
- `indicbert_test.py`: Tamil embedding test
- `inlegalbert_test.py`: Legal text understanding test
- `requirements.txt`: All dependencies
- `sample_audio/`: Example audio files

## Goal

To build a pipeline that helps citizens understand legal outcomes before taking action.

---

### ✅ 3. Git Commands to Push to GitHub

Open terminal in this folder and run:

```bash
git add .
git commit -m "Initial commit: tested all models"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
