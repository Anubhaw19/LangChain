from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled


video_id = "Cbqtxys2qPM"

try:
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id, languages=['en'])

    # Extract plain text
    transcript_text = " ".join([snippet.text for snippet in fetched_transcript.snippets])

    print(transcript_text)

except TranscriptsDisabled:
    print("No captions available for this video.")
