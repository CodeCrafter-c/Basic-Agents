from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

def load_transcript(video_id: str):
    ytt_api = YouTubeTranscriptApi()

    video_transcript = None
    video_lang_code = None

    try:
        transcript_list = ytt_api.list(video_id)

        #  Direct English
        try:
            transcript = transcript_list.find_transcript(['en'])
            video_transcript = transcript.fetch()
            video_lang_code = "en"

        except:
            #  Try translation
            for transcript in transcript_list:
                if transcript.is_translatable:
                    translation_codes = [
                        lang['language_code']
                        for lang in transcript.translation_languages
                    ]

                    if 'en' in translation_codes:
                        video_transcript = transcript.translate('en').fetch()
                        video_lang_code = "en"
                        break

            # 3️ Fallback
            if video_transcript is None:
                transcript = list(transcript_list)[0]
                video_transcript = transcript.fetch()
                video_lang_code = transcript.language_code

    except TranscriptsDisabled:
        print("No captions available for this video")
        return None, None

    clean_text = " ".join(chunk.text for chunk in video_transcript)

    return clean_text, video_lang_code