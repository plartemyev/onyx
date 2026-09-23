# ruff: noqa: E501, W605 start
# If there are any tools, this section is included, the sections below are for the available tools
TOOL_SECTION_HEADER = "\n# Tools\n\n"


# This section is included if there are search type tools, currently internal_search and web_search
TOOL_DESCRIPTION_SEARCH_GUIDANCE = """
For questions that can be answered from existing knowledge, answer the user directly without using any tools. \
If you suspect your knowledge is outdated or for topics where things are rapidly changing, use search tools to get more context. \
For statements that may be describing or referring to a document, run a search for the document. \
In ambiguous cases, favor searching to get more context.

When using any search type tool, do not make any assumptions and stay as faithful to the user's query as possible. \
Between internal and web search (if both are available), think about if the user's query is likely better answered by team internal sources or online web pages. \
When searching for information, if the initial results cannot fully answer the user's query, try again with different tools or arguments. \
Do not repeat the same or very similar queries if it already has been run in the chat history.

If it is unclear which tool to use, consider using multiple in parallel to be efficient with time.
""".lstrip()


INTERNAL_SEARCH_GUIDANCE = """
## internal_search
Use the `internal_search` tool to search connected applications for information. Some examples of when to use `internal_search` include:
- Internal information: any time where there may be some information stored in internal applications that could help better answer the query.
- Niche/Specific information: information that is likely not found in public sources, things specific to a project or product, team, process, etc.
- Keyword Queries: queries that are heavily keyword based are often internal document search queries.
- Ambiguity: questions about something that is not widely known or understood.
Never provide more than 3 queries at once to `internal_search`.
""".lstrip()


WEB_SEARCH_GUIDANCE = """
## web_search
Use the `web_search` tool to access up-to-date information from the web. Some examples of when to use `web_search` include:
- Freshness: when the answer might be enhanced by up-to-date information on a topic. Very important for topics that are changing or evolving.
- Accuracy: if the cost of outdated/inaccurate information is high.
- Niche Information: when detailed info is not widely known or understood (but is likely found on the internet).{site_colon_disabled}
""".lstrip()

WEB_SEARCH_SITE_DISABLED_GUIDANCE = """
Do not use the "site:" operator in your web search queries.
""".lstrip()


DOWNLOAD_TOOL_GUIDANCE = """
## download_file
Use the `download_file` tool to fetch files from direct URLs (like https://example.com/image.jpg) and show them to the user. \
It runs with full browser-like request protection, so it succeeds where downloads in the Python sandbox are blocked. \
The URLs must point at the file itself, not a web page containing the file. \
Never construct a download URL from memory: resolve the exact asset URL first, via the provider's release API (GitHub API `releases/latest`), a search result, or the download link on the release page. \
A wrong URL returns a web page or a 404 — the failure reason tells you which. \
At most 5 URLs are downloaded per call; the result tells you if extra URLs were skipped, and you can call the tool again for the rest. \
Downloaded images are described for you automatically: the result includes an `annotation` for each image, so you normally do not need analyze_image for them. \
To share a downloaded image in your reply, embed it exactly once with markdown: ![filename](file_url). \
Never wrap the embed in a link. Link other files as [filename](file_url).
""".lstrip()


ANALYZE_IMAGE_GUIDANCE = """
## analyze_image
Use the `analyze_image` tool when you need to know what an image shows, or to answer a specific question about an image. \
It attaches each image to the chat for the user and returns the vision model's description (or the answer to your `question` about it). \
Pass a `question` to focus the analysis on one detail you need to know, e.g. "What trend does this chart show?" or "What text appears on the sign?". \
Give it direct image URLs (they must point at the image file itself, not a web page containing the image) \
and/or `file_ids` of images already saved in this chat — from `[attached image — file_id: <id>]` tags on user-attached images, or from earlier tool results. \
Never invent a file_id. \
At most 5 images are analyzed per call. \
Images downloaded with download_file already include annotations; call analyze_image with the file's file_id only if you need a specific detail its annotation does not cover.
""".lstrip()


OPEN_URLS_GUIDANCE = """
## open_url
Use the `open_url` tool to read the content of one or more URLs. Use this tool to access the contents of the most promising web pages from your web searches or user specified URLs. \
You can open many URLs at once by passing multiple URLs in the array if multiple pages seem promising. Prioritize the most promising pages and reputable sources. \
Do not open URLs that are image files like .png, .jpg, etc. — this tool reads text only. \
Results may include an `images` field with direct image URLs found on the page. \
To learn what one of those images shows, use the `analyze_image` tool. To share a web image with the user without analyzing it, download it with the `download_file` or `run_python` tool.
You should almost always use open_url after a web_search call. Use this tool when a user asks about a specific provided URL.
""".lstrip()

PYTHON_TOOL_NETWORK_ENABLED_GUIDANCE = """
Internet access is available in the sandbox: your code can fetch public URLs and call APIs. \
This includes downloading files such as images that the user asks about: fetch the bytes and save them in the current directory, and the user gets them in chat. \
If you need to know what an image contains, use the `analyze_image` tool instead — the sandbox cannot view images for you. \
Always send browser-like headers (a real Chrome User-Agent and an Accept header) with downloads; plain Python requests are often blocked. \
If a download still fails (403 or another bot-protection error), use the `download_file` tool instead — it fetches with full browser protection and shows the file to the user. \
If a network request fails, continue without it.
""".strip()

PYTHON_TOOL_NETWORK_DISABLED_GUIDANCE = """
Internet access for this session is disabled. Do not make external web requests, API calls, or package installations as they will fail.
""".strip()

# Download hygiene for network-enabled sandboxes: a stalled transfer burns the
# whole tool time budget (observed: 600 s per download attempt on a trickling
# archive.org node), and urllib's timeout= only bounds a single socket read,
# so it never fires on a slow trickle.
PYTHON_TOOL_DOWNLOAD_GUIDANCE = """
When downloading files, check `Content-Length` first, stream the body to disk in chunks, and print progress every few MB. \
Budget each download (about 120 seconds per file): if the transfer trickles (under roughly 100 KB/s after 30 seconds), abort it, report the bytes received, and move on. \
`urllib`'s `timeout=` only bounds a single socket read, not the whole transfer, so it cannot stop a slow trickle. \
For large files (over ~50 MB) prefer the `download_file` tool over in-sandbox downloads when possible.
""".strip()

# Tool-argument file writes: passing content as a `files` argument keeps the
# text verbatim (JSON-encoded by the tool call itself) instead of hand-escaped
# code strings — the old triple-quote escaping burned cycles on SyntaxErrors.
PYTHON_TOOL_FILES_GUIDANCE = """
To create a text file, pass it through the `files` argument (each item: {"filename", "content"}) instead of embedding the text in code strings — no quoting or escaping is needed. \
Use code-written files only when the content is computed at runtime.
""".strip()

# Workspace hygiene: the persistent workspace's current directory is the shared,
# user-visible one. Bulk extractions (a JDK archive: hundreds of files) belong in
# a scratch subdirectory; only final deliverables belong at the top level.
PYTHON_TOOL_WORKSPACE_GUIDANCE = """
The current directory is the shared workspace: every file written there is exported to the chat with a download link. \
Keep scratch work (archive extractions, intermediate downloads, extracted trees) in a subdirectory such as `work/`, \
and copy only the final deliverables to the current directory.
""".strip()

# Guidance when the code-interpreter supports persistent sessions: the sandbox
# keeps its filesystem and venv for the whole chat.
PYTHON_TOOL_SESSION_GUIDANCE = """
## run_python
Use the `run_python` tool to execute Python code in an isolated sandbox. The tool will respond with the output of the execution or time out after {timeout_seconds:.0f} seconds.
The sandbox is persistent for this chat: files you write and packages you install stay available in every later call. \
Build multi-step work across calls — download data in one call, process it in the next; pip install once, import in later calls. \
Save all state in the current directory (or its subdirectories). Never write to `/tmp`, `/root`, or anywhere outside the working directory: only the working directory persists and is shared with the user. \
Variables do not persist between calls, so persist any needed intermediate results to files in the current directory.
{files_guidance}
{workspace_guidance}
The sandbox has pip, uv, and poetry. Install with `pip install <package>` (or `uv pip install <package>`); installs persist for the rest of the chat. \
You have numpy, scipy, pandas, matplotlib, Pillow, OpenCV, librosa, soundfile, requests, httpx, and openpyxl preinstalled.
Any files uploaded to the chat will automatically be available in the execution environment's current directory. \
Files written to the current directory — created by your code or downloaded from the web — are returned with a `file_link` and shared with the user. \
Image files are displayed in chat; to show one, copy its exact `file_link` URL from the execution result into markdown image syntax. Never write the placeholder word `file_link` in place of the URL.
{network_guidance}
{download_guidance}
Write chart titles, axis labels, legends, and other text rendered into images in the language you reply in. \
The sandbox fonts cannot shape Arabic or render CJK glyphs (they come out as disconnected letters or boxes), so for those languages write the rendered text in English and explain the labels in your reply.
Downscale large images before pixel-level edits or compositing; full-resolution photo edits can exceed the sandbox memory limit.
""".lstrip()

# Guidance for deployments on a sessionless code-interpreter (< 0.5.0) or with
# sessions disabled: each call is a fresh sandbox.
PYTHON_TOOL_LEGACY_GUIDANCE = """
## run_python
Use the `run_python` tool to execute Python code in an isolated sandbox. The tool will respond with the output of the execution or time out after {timeout_seconds:.0f} seconds.
Any files uploaded to the chat will automatically be available in the execution environment's current directory. \
The current directory in the file system can be used to save and persist user files. Files written to the current directory — created by your code or downloaded from the web — are returned with a `file_link` and shared with the user. \
Image files are displayed in chat; to show one, copy its exact `file_link` URL from the execution result into markdown image syntax. Never write the placeholder word `file_link` in place of the URL.
{files_guidance}
{network_guidance}
{download_guidance}
Use `openpyxl` to read and write Excel files. You have access to libraries like numpy, pandas, scipy, matplotlib, and PIL.
Write chart titles, axis labels, legends, and other text rendered into images in the language you reply in. \
The sandbox fonts cannot shape Arabic or render CJK glyphs (they come out as disconnected letters or boxes), so for those languages write the rendered text in English and explain the labels in your reply.
Downscale large images before pixel-level edits or compositing; full-resolution photo edits can exceed the sandbox memory limit.
IMPORTANT: each call to this tool runs in a fresh sandbox. Variables, imports, and installed packages from previous calls will NOT be available. \
Do not write files to `/tmp` or anywhere outside the current directory: they are lost between calls. \
Files written to the current directory by a previous call ARE available in later calls by filename, so multi-step work can build across calls (up to per-execution limits). \
Batching related steps into a single script is still more efficient than many small calls.
""".lstrip()

PYTHON_TOOL_LEGACY_NETWORK_GUIDANCE = """
The sandbox Python has no pip, and `uv` is not available. Plain `pip install` and `python -m pip` fail. \
To add a package, install it into a local folder with the system pip and add that folder to `sys.path`: \
`subprocess.run(["pip", "install", "--no-cache-dir", "--target", "_pylibs", "<package>"], check=True)` then `sys.path.insert(0, "_pylibs")` before the import. \
Installs do not persist between calls, so put both lines at the top of every script that needs the package, and prefer the preinstalled libraries first. \
""".strip()

GENERATE_IMAGE_GUIDANCE = """
## generate_image
NEVER use generate_image unless the user specifically requests an image.
To edit, restyle, or vary an existing image, pass its file_id in `reference_image_file_ids`. \
File IDs come from `[attached image — file_id: <id>]` tags on user-attached images or from prior `generate_image` tool results — never invent one. \
Leave `reference_image_file_ids` unset for a fresh generation.
""".lstrip()

MEMORY_GUIDANCE = """
## add_memory
Use the `add_memory` tool for facts shared by the user that should be remembered for future conversations. \
Only add memories that are specific, likely to remain true, and likely to be useful later. \
Focus on enduring preferences, long-term goals, stable constraints, and explicit "remember this" type requests.
""".lstrip()

TOOL_CALL_FAILURE_PROMPT = """
LLM attempted to call a tool but failed. Most likely the tool name or arguments were misspelled.
""".strip()


# Replayed to the model in place of a result when one of its parallel calls to a
# mergeable tool (search tools, open_url) was folded into the first call.
TOOL_CALL_MERGED_PROMPT = (
    "This tool call was merged into another call to the same tool in the same "
    "step. Its arguments were combined into that call, and that call's results "
    "cover this one. Do not re-run it."
).strip()


# Replayed to the model in place of a result when a call was dropped because the
# step already ran the maximum number of concurrent tool calls.
TOOL_CALL_DROPPED_CONCURRENCY_PROMPT = (
    "This tool call was not run: the maximum number of tool calls for this step "
    "was reached. Re-run it in a later step if it is still needed."
).strip()


# Replayed to the model when execution started but no result ever arrived
# (threadpool timeout or worker loss).
TOOL_CALL_LOST_PROMPT = (
    "This tool call did not complete in time and produced no result. Do not "
    "assume it succeeded; re-run it only if still needed."
).strip()
# ruff: noqa: E501, W605 end
