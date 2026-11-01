"""Constants for search tool implementations."""

# Query Expansion and Fusion Weights
# Taking an opinionated stance on the weights, no chance users can do a good job customizing this.
# The dedicated rephrased/extracted semantic query is likely the best for hybrid search
LLM_SEMANTIC_QUERY_WEIGHT = 1.3
# The keyword expansions provide more breadth through a different search ranking function
# This one is likely to produce the most different results.
LLM_KEYWORD_QUERY_WEIGHT = 1.0
# This is also lower because it is the LLM generated query without the custom instructions specifically for this purpose.
LLM_NON_CUSTOM_QUERY_WEIGHT = 0.7
# This is much lower weight because it is likely pretty similar to the LLM semantic query but just worse quality.
ORIGINAL_QUERY_WEIGHT = 0.5

# Hybrid Search Configuration
# This may in the future just use an entirely keyword search. Currently it is a hybrid search with a keyword first phase.
KEYWORD_QUERY_HYBRID_ALPHA = 0.2

# Reciprocal Rank Fusion
RRF_K_VALUE = 50

# RRF ranks without reading content, so relevant documents often land just
# below the selection cut; a wider budget lets the selection LLM rescue them.
SELECTION_TOKEN_BUDGET_MULTIPLIER = 2

# The document-selection prompt is trimmed to a fixed content budget derived
# from chunk counts, but the secondary model running the selection may have a
# much smaller window than that budget assumes. Prompt skeleton, per-section
# JSON metadata, and the model's output reserve are subtracted from its input
# window before clamping; the floor keeps a tiny window usable at all.
SELECTION_PROMPT_RESERVE_TOKENS = 4096
SELECTION_MIN_CONTENT_BUDGET_TOKENS = 1024

# History headroom kept for the query-expansion prompts and their output when
# the history is trimmed to the secondary model's window. Messages carry a
# rough per-message overhead on top of the char-based token estimate.
EXPANSION_HISTORY_RESERVE_TOKENS = 2048
ESTIMATED_TOKENS_PER_MESSAGE_OVERHEAD = 8

# Context Expansion
FULL_DOC_NUM_CHUNKS_AROUND = 5

# If a document is quite relevant and has many returned sections, likely it's enough to use the chunks around
# the highest scoring section to detect relevance. This allows more other docs to be evaluated in the step.
# This avoids documents with good titles or generally strong matches to flood out the rest of the search results.
# If there are multiple indepedent sections from the doc, this won't truncate it, only if they're connected.
MAX_CHUNKS_FOR_RELEVANCE = 3
