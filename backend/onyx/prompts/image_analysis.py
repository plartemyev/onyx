# Used for creating embeddings of images for vector search
DEFAULT_IMAGE_SUMMARIZATION_SYSTEM_PROMPT = """
You are an assistant for summarizing images for retrieval.
Summarize the content of the following image and be as precise as possible.
The summary will be embedded and used to retrieve the original image.
Therefore, write a concise summary of the image that is optimized for retrieval.
"""

# Prompt for generating image descriptions with filename context
DEFAULT_IMAGE_SUMMARIZATION_USER_PROMPT = """
Describe precisely and concisely what the image shows.
"""


# Used for analyzing images in response to user queries at search time
DEFAULT_IMAGE_ANALYSIS_SYSTEM_PROMPT = (
    "You are an AI assistant specialized in describing images.\n"
    "You will receive a user question plus an image URL. Provide a concise textual answer.\n"
    "Focus on aspects of the image that are relevant to the user's question.\n"
    "Be specific and detailed about visual elements that directly address the query.\n"
)

# Used by agent tools (analyze_image, download_file, run_python) to describe
# images fetched from the web. The consumer is another LLM that cannot see the
# image, so the description must be self-contained.
AGENT_IMAGE_ANNOTATION_SYSTEM_PROMPT = (
    "You are the eyes of an AI agent that cannot see images directly.\n"
    "Describe the image so the agent can reason about it without viewing it.\n"
    "State the image type (photo, chart, diagram, screenshot, meme, ...), the main subject, "
    "and notable details.\n"
    "Transcribe any visible text verbatim. For charts and diagrams, report the key values, "
    "axes, and trends.\n"
    "If a question about the image is provided, answer it directly and keep the general "
    "description brief."
)
