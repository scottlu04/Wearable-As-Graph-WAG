QUERY_PROMPT_BASE = """
-Goal-
Given the user's question, the goal is to answer the question based on retrived context and data. This process includes the following steps:
1. Extract the following information from the user's question:
    - Key Entities: 
    - Time period: Today's date is {today_date}. and the default time period is past 7 days if not specified.
2. Context Search: 
    - Retrieve relevant context and data

3. Answer the user's question based on the retrieved context and data. 
    - If the question is not answerable, please say so.    
output format:
Answer: [answer]
"""

QUERY_PROMPT_BASE = """
## CORE OBJECTIVE
Analyze health queries through structured function calls to external knowledge retrival APIs, synthesizing results into evidence-based responses through systematic analysis of retrived context.

## EXECUTION FRAMEWORK

1. **QUERY DECOMPOSITION**
   - **Key Entities**: Identify health metrics (e.g., HRV, heart rate)
   - **Temporal Scope**:
     - Default: Past 7 days
     - Explicitly stated periods override default

2. **KNOWLEDGE RETRIEVAL**
    - Primary Entity Matching: Fetch data for core health metric.
    - Contextual Filtering: Apply time-based constraints.

3. **ANALYSIS**
    - Cross-reference data with medical best practices.
    - Highlight trends, anomalies, or gaps.

4. **RESPONSE GENERATION**
   - Requirements:
     - Ground all claims in evidence
     - Acknowledge data limitations
     - For unanswerable queries: Specify missing data

## OUTPUT FORMAT
Answer: Concise response with integrated insights.
"""



QUERY_PROMPT_KGRAG = """
-Goal-
Given the user's question, the goal is to answer the question by using the knowledge graph built. This process includes the following steps:
1. Extract the following information from the user's question:
    - Key Entities: 
    - Time period: Today's date is {today_date}. and the default time period is past 7 days if not specified.
    - Openness score: Define an openness score (0.0-1.0, where 1.0 is most open-ended) for the question, which determines the extent to which the system should search for related nodes or entities beyond the primary matched entity. Here's a breakdown:
        - Higher Scores (e.g., 0.8-1.0):
            Indicate that the question is very open-ended. The system should explore a broader range of related nodes, considering multiple connected factors or variables.
            - examples: 
                "I noticed my heart rate was elevated yesterday. Do you know what might have caused it?" -> 0.9
        - Moderate Scores (around 0.4-0.7):
            Suggest a balanced approach. The system focuses on the primary entity while also allowing for some exploration of related factors without drifting too far.
            - examples: 
                "How does my current week's heart rate compare to last week?" -> 0.5
        - Lower Scores (e.g., 0.0-0.3):
            Signal that the question is narrowly focused. The search remains tightly bound to the primary entity with minimal expansion into related nodes.
            - examples: 
                "What is the optimal range for HRV?" -> 0.0
                "What is my max HRV and heart rate today?" -> 0.0
        - examples: 
            Do you think I am stressed? -> low openness score, mostly closed-ended, as it invites a yes/no or binary response
            I am feeling stressed, do you have an idea why? -> high openness score, inviting various interpretations

2. Graph Search: 
    - Find matched entities and relationships
    - Expand search based on openness score
    - Retrieve relevant context and data

3. Analyze Interconnections:
    - Given the question as the context.
    - Explore relationships between mutimodal data from both matched and related entities.
    - Extract insights like which modalities are consistent or inconsisent.

4. Answer the user's question based on the retrieved context and data:
    - Ensure accuracy and relevance based on the knowledge graph.
    - If the question is not answerable, please say so.
    - If applicable, derive insights from multiple entities and their connections.

output format:
Answer: [answer]
"""


QUERY_PROMPT_KGRAG_v2 = """
-Goal-
Given the user's question, the goal is to answer the question by using the knowledge graph built. This process includes the following steps:
1. Extract the following information from the user's question:
    - Key Entities: 
    - Time period: Today's date is {today_date}. and the default time period is past 7 days if not specified.
    - Openness score: Define an openness score (0.0-1.0, where 1.0 is most open-ended) for the question, which determines the extent to which the system should search for related nodes or entities beyond the primary matched entity. Here's a breakdown:
        - Higher Scores (e.g., 0.8-1.0):
            Indicate that the question is very open-ended. The system should explore a broader range of related nodes, considering multiple connected factors or variables.
            - examples: 
                "I noticed my heart rate was elevated yesterday. Do you know what might have caused it?" -> 0.9
        - Moderate Scores (around 0.4-0.7):
            Suggest a balanced approach. The system focuses on the primary entity while also allowing for some exploration of related factors without drifting too far.
            - examples: 
                "How does my current week's heart rate compare to last week?" -> 0.5
        - Lower Scores (e.g., 0.0-0.3):
            Signal that the question is narrowly focused. The search remains tightly bound to the primary entity with minimal expansion into related nodes.
            - examples: 
                "What is the optimal range for HRV?" -> 0.0
                "What is my max HRV and heart rate today?" -> 0.0
        - examples: 
            Do you think I am stressed? -> low openness score, mostly closed-ended, as it invites a yes/no or binary response
            I am feeling stressed, do you have an idea why? -> high openness score, inviting various interpretations

2. Graph Search: 
    - Find matched entities and relationships
    - Expand search based on openness score
    - Retrieve relevant context and data

3. Answer the user's question based on the retrieved context and data. 
    - Explore the inter-connections between data of the matched entities and the related entities to find more insights.
    - If applicable, utilize the context and data from the related nodes to answer the question.
    - Ensure accuracy and relevance based on the knowledge graph.
    - If the question is not answerable, please say so.
    - If applicable, derive insights from multiple entities and their connections.

output format:
Answer: [answer]
"""





QUERY_PROMPT_KGRAG_v3 = """
## CORE OBJECTIVE
Analyze health queries through structured function calls to external graph traversal APIs, synthesizing results into evidence-based responses through systematic analysis of entities, relationships, and multimodal connections.

## EXECUTION FRAMEWORK

1. **QUERY DECOMPOSITION**
   - **Key Entities**: Identify primary subjects/measurements (e.g., HRV, heart rate)
   - **Temporal Scope**:
     - Default: Past 7 days
     - Explicitly stated periods override default
   - **Openness Score** (0.0-1.0):
     | Score Range  | Search Strategy                | Examples                      |
     |--------------|---------------------------------|-------------------------------|
     | 0.0-0.3      | Narrow focus on exact matches  | "Optimal HRV range?" (0.0)    |
     | 0.4-0.7      | Balanced entity+relationships  | "HR comparison week/week?" (0.5) |
     | 0.8-1.0      | Broad multimodal exploration   | "Why elevated heart rate?" (0.9) |

2. **GRAPH TRAVERSAL**
   - Primary entity matching
   - Relationship expansion proportional to openness score
   - Contextual data retrieval with temporal filtering

3. **MULTIMODAL ANALYSIS**
   - Cross-reference data types:
     * Physiological (HRV, HR)
     * Environmental (sleep, activity)
     * Subjective (user notes)
   - Identify:
     - Consistent corroborating evidence
     - Conflicting indicators
     - Temporal patterns

4. **RESPONSE GENERATION**
   - Requirements:
     - Ground all claims in evidence
     - Acknowledge data limitations
     - For unanswerable queries: Specify missing data
   - Prioritize:
     - Direct correlations > inferred relationships
     - User-specific context > general knowledge

## OUTPUT FORMAT
Answer: Concise response with integrated insights.
"""


## OUTPUT FORMAT
"""
Answer: [Concise response with integrated insights. For complex findings, use:
- "Key Patterns:"
- "Notable Connections:"
- "Recommendations:"]
"""


QUERY_PROMPT_v2 = """
-Goal-
Given the user's question, the goal is to answer the question by using the knowledge graph built. This process includes the following steps:
1. Entity Extraction: Extract key entities and time period in the user's question. Today's date is {today_date}. and the default time period is past 7 days.
2. Openness Score: Define an openness score (0.0-1.0, where 1.0 is most open-ended) for the question, which will guide the graph search process.
3. Graph Search: Search the knowledge graph for the key entities to retrieve the context and data of related entities and their relationships. 
4. Answer Generation: Answer the user's question based on the context.

Guideline:
Ensure accuracy and relevance based on the knowledge graph.
If applicable, derive insights from multiple entities and their connections.
output format:
Answer: [answer]
"""

instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""

QUERY_PROMPT_v3 = """
-Goal-
Given the user's question, the goal is to answer the question by using the knowledge graph built. This process includes the following steps:
(1) Entity Extraction: Extract key entities and time period in the user's question. Today's date is {today_date}. and the default time period is past 7 days.
(2) Graph Search: Search the knowledge graph for the key entities.
(3) Answer Generation: Answer the user's question based on the context.
output format:
Question: question
Entity Extraction:
entities: [entity1, entity2, ...]
time: [start_time, end_time]
Graph Search: 
context: [Context]
Answer Generation:
answer: [answer]
"""
