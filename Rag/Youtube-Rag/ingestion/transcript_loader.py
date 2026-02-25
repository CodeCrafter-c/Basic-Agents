from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.documents import Document


def load_transcript(video_id: str, window_size: int = 60):

    ytt_api = YouTubeTranscriptApi()

    video_transcript = None
    video_lang_code = None

    try:
        transcript_list = ytt_api.list(video_id)

        #   direct English transcript
        try:
            transcript = transcript_list.find_transcript(['en'])
            video_transcript = transcript.fetch()
            video_lang_code = "en"

        except:
            #  translation to English
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

            # Fallback:  original language
            if video_transcript is None:
                transcript = list(transcript_list)[0]
                video_transcript = transcript.fetch()
                video_lang_code = transcript.language_code

    except TranscriptsDisabled:
        print("No captions available for this video")
        return None, None

    #grouping
    documents = []
    current_window_start = 0
    current_text = []

    for chunk in video_transcript:
        chunk_start = chunk.start
        chunk_text = chunk.text

        
        if chunk_start >= current_window_start + window_size:
            if current_text:
                documents.append(
                    Document(
                        page_content=" ".join(current_text),
                        metadata={
                            "start": current_window_start,
                            "window_size": window_size
                        }
                    )
                )

            
            current_window_start += window_size
            current_text = []

        current_text.append(chunk_text)

    if current_text:
        documents.append(
            Document(
                page_content=" ".join(current_text),
                metadata={
                    "start": current_window_start,
                    "window_size": window_size
                }
            )
        )
    return documents, video_lang_code


