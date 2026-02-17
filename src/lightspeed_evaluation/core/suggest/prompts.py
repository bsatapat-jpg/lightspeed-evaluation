"""Prompts for metric suggestion feature."""

# Available metrics catalog with descriptions and requirements
AVAILABLE_METRICS = """
## TURN-LEVEL METRICS (Applied per conversation turn)

### Ragas Framework Metrics
These metrics evaluate response quality using LLM-based assessment.

1. **ragas:response_relevancy**
   - Description: Evaluates how relevant the response is to the user's question
   - Use when: You need to check if responses directly address user queries
   - Requires: query, response
   - Default threshold: 0.8

2. **ragas:faithfulness**
   - Description: Evaluates how faithful the response is to the provided context
   - Use when: Testing RAG applications where responses should be grounded in retrieved context
   - Requires: query, response, contexts
   - Default threshold: 0.8

3. **ragas:context_recall**
   - Description: Evaluates if all facts needed for the answer were retrieved
   - Use when: Testing retrieval quality in RAG systems
   - Requires: query, response, contexts, expected_response
   - Default threshold: 0.8

4. **ragas:context_precision_with_reference**
   - Description: Evaluates precision of retrieved context against a reference answer
   - Use when: You have ground truth and want to measure retrieval precision
   - Requires: query, response, contexts, expected_response
   - Default threshold: 0.7

5. **ragas:context_precision_without_reference**
   - Description: Evaluates context precision without needing reference answers
   - Use when: Testing retrieval precision without ground truth
   - Requires: query, response, contexts
   - Default threshold: 0.7

6. **ragas:context_relevance**
   - Description: Evaluates if retrieved context is relevant to the user query
   - Use when: Checking retrieval relevance in RAG systems
   - Requires: query, contexts
   - Default threshold: 0.7

### Custom Framework Metrics
Custom metrics for specific evaluation needs.

7. **custom:keywords_eval**
   - Description: Boolean evaluation checking if ALL expected keywords appear in response
   - Use when: Testing for specific required terms, technical keywords, or phrases
   - Requires: response, expected_keywords (list of keyword sets)
   - Returns: 0 or 1 (boolean)

8. **custom:answer_correctness**
   - Description: LLM-based evaluation of response correctness against expected answer
   - Use when: Comparing response semantic correctness with expected responses
   - Requires: query, response, expected_response
   - Default threshold: 0.75

9. **custom:intent_eval**
   - Description: LLM-based evaluation of intent alignment
   - Use when: Checking if response matches the expected intent/purpose
   - Requires: query, response, expected_intent
   - Returns: 0 or 1 (boolean)

10. **custom:tool_eval**
    - Description: Evaluates tool/function calls comparing expected vs actual
    - Use when: Testing AI agents that use tools, function calling, or MCP servers
    - Requires: expected_tool_calls, actual tool_calls from API
    - Configuration options:
      - ordered: true/false (whether tool call order matters)
      - full_match: true/false (exact match vs subset matching)

### Script-Based Metrics

11. **script:action_eval**
    - Description: Executes validation scripts for infrastructure/environment testing
    - Use when: Need to verify real-world actions (e.g., Kubernetes deployments, API calls)
    - Requires: verify_script path in turn data

### NLP-Based Metrics (Non-LLM)
Text comparison metrics that don't require LLM calls.

12. **nlp:bleu**
    - Description: BLEU score measuring n-gram overlap between response and expected
    - Use when: Quick text similarity check, translation quality
    - Requires: response, expected_response
    - Configuration: max_ngram (1-4, default 4)
    - Default threshold: 0.5
    - Note: Measures exact text overlap, not semantic meaning

13. **nlp:rouge**
    - Description: ROUGE score measuring recall-oriented n-gram overlap
    - Use when: Summarization evaluation, text coverage assessment
    - Requires: response, expected_response
    - Configuration: rouge_type (rouge1, rouge2, rougeL, rougeLsum)
    - Default threshold: 0.3
    - Note: Measures text overlap, not semantic meaning

14. **nlp:semantic_similarity_distance**
    - Description: String distance metrics (Levenshtein, Jaro, etc.)
    - Use when: Exact string matching scenarios
    - Requires: response, expected_response
    - Configuration: distance_measure (levenshtein, hamming, jaro, jaro_winkler)
    - Default threshold: 0.7
    - Warning: NOT recommended for LLM outputs - use custom:answer_correctness instead

### GEval Custom Metrics (Configuration-Driven)

15. **geval:<custom_name>**
    - Description: Fully customizable evaluation criteria using DeepEval's GEval
    - Use when: Standard metrics don't fit your use case
    - Create by specifying:
      - criteria: Description of what to evaluate
      - evaluation_params: Which data fields to use (query, response, expected_response, contexts)
      - evaluation_steps: Step-by-step evaluation guidance
      - threshold: Pass/fail threshold
    - Example use cases: domain-specific evaluation, custom quality criteria

## CONVERSATION-LEVEL METRICS (Applied to entire conversation)

### DeepEval Framework Metrics

16. **deepeval:conversation_completeness**
    - Description: Evaluates how completely the conversation addresses user intentions
    - Use when: Testing multi-turn conversations for task completion
    - Requires: Full conversation history
    - Default threshold: 0.8

17. **deepeval:conversation_relevancy**
    - Description: Evaluates conversation relevance to topic/context
    - Use when: Testing multi-turn conversation coherence
    - Requires: Full conversation history
    - Default threshold: 0.7

18. **deepeval:knowledge_retention**
    - Description: Evaluates how well the model retains information across turns
    - Use when: Testing memory/context maintenance in long conversations
    - Requires: At least 2 turns
    - Default threshold: 0.7

### GEval Conversation Metrics

19. **geval:<custom_name>** (conversation-level)
    - Description: Customizable conversation-level evaluation
    - Use when: Need to evaluate conversation-wide properties
"""

METRIC_SUGGESTION_PROMPT = """You are an expert in evaluation metrics for LLM/GenAI applications.

Analyze the user's use case and suggest the most appropriate metrics from our evaluation framework.

## User's Use Case:
{user_prompt}

## Available Metrics:
{available_metrics}

## Your Task:
1. Understand the user's evaluation goals, what they're testing, and expected outcomes
2. Select appropriate metrics from the available options
3. If standard metrics don't fully cover the use case, suggest custom GEval metrics with specific criteria
4. Provide reasoning for each suggestion

## Response Format (JSON):
{{
    "analysis": {{
        "use_case_summary": "Brief summary of what the user wants to evaluate",
        "evaluation_type": "turn_level|conversation_level|both",
        "key_requirements": ["list of key evaluation requirements"]
    }},
    "suggested_metrics": {{
        "turn_level": [
            {{
                "metric": "framework:metric_name",
                "reason": "Why this metric is appropriate",
                "threshold": 0.8,
                "required_fields": ["query", "response", "etc"],
                "configuration": {{}}
            }}
        ],
        "conversation_level": [
            {{
                "metric": "framework:metric_name",
                "reason": "Why this metric is appropriate",
                "threshold": 0.7,
                "required_fields": [],
                "configuration": {{}}
            }}
        ]
    }},
    "custom_metrics": [
        {{
            "metric": "geval:custom_metric_name",
            "level": "turn_level|conversation_level",
            "criteria": "Detailed evaluation criteria description",
            "evaluation_params": ["query", "response"],
            "evaluation_steps": [
                "Step 1: ...",
                "Step 2: ..."
            ],
            "threshold": 0.7,
            "reason": "Why a custom metric is needed"
        }}
    ],
    "data_requirements": {{
        "required_fields": ["query", "response", "expected_response", "contexts", "etc"],
        "optional_fields": ["expected_keywords", "expected_intent", "expected_tool_calls"],
        "notes": "Any special data preparation notes"
    }},
    "recommendations": "Additional recommendations for the evaluation setup"
}}

Provide ONLY the JSON response, no additional text."""
