# Railway Deployment Guide

## Overview
This guide helps you deploy the AI Interview API to Railway.

## Prerequisites
1. Railway account
2. Google API key for Gemini
3. ElevenLabs API key (optional, for better TTS)

## Environment Variables
Set these in your Railway project:

```
GOOGLE_API_KEY=your_google_api_key_here
ELEVEN_API_KEY=your_elevenlabs_api_key_here (optional)
```

## Deployment Steps

1. **Connect your repository to Railway**
   - Go to Railway dashboard
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose this repository

2. **Set environment variables**
   - In your Railway project settings
   - Add the environment variables listed above

3. **Deploy**
   - Railway will automatically detect the Dockerfile
   - The build process will install dependencies
   - The app will start using the Procfile

## Troubleshooting

### Build Timeout Issues
If the build times out:

1. **Check logs**: Look for specific error messages
2. **Reduce dependencies**: The app loads heavy models (Whisper, sentence transformers)
3. **Use lazy loading**: Models are now loaded only when needed
4. **Increase build timeout**: Railway allows up to 45 minutes

### Common Issues

1. **Missing API keys**: Ensure GOOGLE_API_KEY is set
2. **Port issues**: Railway sets PORT automatically
3. **Memory issues**: The app uses significant memory for ML models

### Health Check
The app includes a health check endpoint at `/health` that Railway can use to verify the app is running.

### Performance Tips
- Models are loaded lazily to reduce startup time
- Use the health check endpoint to monitor app status
- Consider using Railway's auto-scaling features

## Local Testing
To test locally before deploying:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GOOGLE_API_KEY=your_key_here
export ELEVEN_API_KEY=your_key_here

# Run the app
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /start-interview/` - Start interview session
- `POST /answer/` - Submit answer
- `POST /tts/` - Text-to-speech
- `POST /stt/` - Speech-to-text
- `POST /signup/` - User registration
- `POST /login/` - User login

## Monitoring
- Check Railway logs for errors
- Monitor memory usage
- Use the health check endpoint
- Set up alerts for downtime 