EVAL_PROMPT_CONTEXT_new = """
You are an expert in clinical evaluation and human-centered AI systems. Your task is to evaluate and compare the quality of retrieved contextual information used to answer a user's health-related query based on their wearable data.

Each retrieval method provides a different set of contextual knowledge (e.g., entities, relationships, multimodal time-series patterns) intended to support answering the user's question. Your goal is to assess how well each method performs on three key criteria and then provide an overall ranking from best to worst.

---

**Input Format**
{
  "query": "<User's health-related question>",
  "methods": {
    "method_1": {
      "retrieved_content": "<context retrieved by method 1>"
    },
    "method_2": {
      "retrieved_content": "<context retrieved by method 2>"
    },
    ...
  }
}

---

**Evaluation Criteria**

1. **Relevance**  
   - Does the content directly address the user's query and underlying intent?  
   - Are the retrieved entities, metrics, and relationships the right ones to help answer the question?

2. **Insightfulness**  
   - Does the content go beyond surface-level definitions or isolated metrics?  
   - Does it surface helpful insights, patterns, or interpretations—especially those that link across **modalities** (e.g., sleep + activity + mood)?

3. **Temporal Interconnectedness**  
   - Does the context reflect **time-varying patterns** across modalities (e.g., sequences, trends, or disruptions)?  
   - Are temporal relationships between modalities identified (e.g., "drop in activity precedes poor sleep, which aligns with lower mood")?  
   - Is there evidence of **cross-modal and time-aware reasoning** (e.g., interactions over days or weeks, lag effects, feedback loops)?

---

**Output Format**
Return your evaluation as a structured JSON object:

{
  "per_criterion_ranking": {
    "Relevance": ["method_1", "method_2", ...],
    "Insightfulness": ["method_2", "method_1", ...],
    "Temporal Interconnectedness": ["method_3", "method_1", ...]
  },
  "justification": "<Short explanation for the ranking>",
}
"""



EVAL_PROMPT_CONTEXT = """
You are an expert in clinical evaluation and human-centered AI systems. Your task is to evaluate and compare the quality of retrieved contextual information used to answer a user's health-related query based on their wearable data.

Each retrieval method provides a different set of contextual knowledge (e.g., entities, relationships, time-series metrics) intended to support answering the user's question. Your goal is to **rank the methods from best to worst** based on how effectively their retrieved content supports generating an accurate, insightful, and user-centered answer.

---

**Input Format**
{
  "query": "<User's health-related question>",
  "methods": {
    "method_1": {
      "retrieved_content": "<context retrieved by method 1>"
    },
    "method_2": {
      "retrieved_content": "<context retrieved by method 2>"
    },
    ...
  }
}

---

**Evaluation Criteria**

1. **Relevance**  
   - Does the content directly address the user's query and underlying intent?  
   - Are the entities and relationships retrieved the right ones to help answer the question?

2. **Insightfulness**  
   - Does the content go beyond surface-level definitions or data?  
   - Does it surface helpful insights, connections, or interpretations—especially those that link across modalities (e.g., linking sleep patterns with activity or mood)?

3. **Interconnectedness**  
   - How well are the primary entities linked to related variables or modalities?  
   - Are the relationships between metrics synergistic, contradictory, or disjointed?  
   - Does the content reveal any surprising, holistic, or clinically meaningful patterns across data types?

---

**Output Format**
Return your evaluation as a structured JSON object:

{
  "ranking": ["method_1", "method_2", ...],
  "justification": "<Short explanation for the ranking>",
}
"""


EVAL_PROMPT_CONTEXT_V3 = """
You are an expert in clinical evaluation and human-centered AI systems. Your task is to evaluate and compare the quality of retrieved contextual information used to answer a user's health-related query based on their wearable data.

Each retrieval method provides a different set of contextual knowledge (e.g., entities, relationships, multimodal time-series patterns) intended to support generating a useful, personalized answer. Your goal is to **rank the methods from best to worst** based on how well their retrieved content supports insightful, temporally aware, and multimodal understanding.

---

**Input Format**
{
  "query": "<User's health-related question>",
  "methods": {
    "method_1": {
      "retrieved_content": "<context retrieved by method 1>"
    },
    "method_2": {
      "retrieved_content": "<context retrieved by method 2>"
    },
    ...
  }
}

---

**Evaluation Criteria**

1. **Relevance**  
   - Does the retrieved content directly address the user’s query and its underlying intent?
   - Are the chosen modalities and variables appropriate for answering the question?

2. **Insightfulness**  
   - Does the context offer meaningful or novel clinical insights rather than just surface-level reporting?
   - Does it reveal **cross-modal patterns or interactions** (e.g., low activity + irregular sleep → mood issues)?

3. **Multimodal Temporal Reasoning**  
   - Does the context reflect **time-varying trends, cycles, or anomalies** within and across modalities?
   - Are there **temporally structured relationships** across signals (e.g., “reduced REM sleep precedes lower mood two days later”)?
   - Does the method capture **short-term dynamics** (e.g., daily fluctuations) and **long-term patterns** (e.g., gradual decline)?
   - Are interactions between modalities **synchronized**, **lagged**, or **causally inferred**?

---

**Your Task**

Use the criteria above to **rank the retrieval methods** from best to worst.

- Highlight how well each method makes use of **multimodal time-series characteristics**, such as dynamic interactions between sleep, activity, heart rate, and mood over time.
- Address whether the context enables **explanations**, **comparisons**, or **predictions** based on multimodal and temporal signals.
- Point out any limitations, missed opportunities, or irrelevant content.

---

**Output Format**
Return your evaluation as a structured JSON object:

{
  "ranking": ["method_X", "method_Y", "method_Z"],
  "justification": {
    "method_X": "Ranked highest for capturing both short-term mood dips and long-term REM sleep deficits, and linking them to stress and reduced physical activity. Strong cross-modal temporal reasoning.",
    "method_Y": "Relevant context and some insightful relationships, but limited temporal reasoning. Did not account for time lags between variables.",
    "method_Z": "Surface-level correlations without temporal structure or modality interaction. Mostly static and unimodal content."
  }
}
"""

EVAL_PROMPT_RANK = """

You are an expert in clinical evaluation and human-centered AI systems. Your task is to evaluate and compare the quality of response generated by multiple methods to answer a user's health-related query based on their wearable data.

Each retrieval method provides a different set of contextual knowledge (e.g., entities, relationships, multimodal time-series patterns) intended to support answering the user's question. Your goal is to assess the quality of the response generated by each method.

-Input Format-
{
  "query": "<user's health question>",
  "methods": {
    "method_1": {
      "response": "<generated answer>"
    },
    "method_2": {
      "response": "<generated answer>"
    },
    ...
  }
}

-Evaluation Criteria-
Rank the methods from 1 (best) to N (worst) for each of the following dimensions:

1. **Insightfulness(most important)**: Does the response offer meaningful, actionable insights beyond the obvious?
2. **Relevance**: Is the response relevant and does it include novel information?
3. **Groundedness**: Are factual claims well-supported by the provided content or trusted sources?
4. **Personalization**: Does the response meaningfully incorporate the user's context (e.g., wearable data)?
5. **Clarity**: Is the response clearly written, logically structured, and easy to understand for a non-expert?
6. **Absence_of_harmful_content**:  Is the response free from misleading, unsafe, or inappropriate information? 

Important Notes:

Do not assign the same rank to multiple methods unless they are truly indistinguishable in that dimension.

Rank relative to each other within the batch, not by absolute standards.

Lower rank numbers are better (1 = best performance for that criterion).


-Output Format-
Return a dictionary with evaluation scores per method:

{
  "method_1": {
    "Overall_quality": <1-N>,
    "Insightfulness": <1-N>,
    "Relevance": <1-N>,
    "Groundedness": <1-N>,
    "Personalization": <1-N>,
    "Clarity": <1-N>,  
    "Absence_of_harmful_content": <0 or 1>
  },
  ...
}
"""

EVAL_PROMPT_SINGLE_RANK = """

You are an expert in clinical evaluation and human-centered AI systems. Your task is to evaluate and compare the quality of response generated by multiple methods to answer a user's health-related query based on their wearable data.

Each retrieval method provides a different set of contextual knowledge (e.g., entities, relationships, multimodal time-series patterns) intended to support answering the user's question. Your goal is to assess the quality of the response generated by each method.

-Input Format-
{
  "query": "<user's health question>",
  "methods": {
    "method_1": {
      "response": "<generated answer>"
    },
    "method_2": {
      "response": "<generated answer>"
    },
    ...
  }
}

-Evaluation Criteria-
Rank the methods based on the following dimensions:
1. **Insightfulness(most important)**: Does the response offer meaningful, actionable insights beyond the obvious?
2. **Relevance**: Is the response relevant and does it include novel information?
3. **Groundedness**: Are factual claims well-supported by the provided content or trusted sources?
4. **Personalization**: Does the response meaningfully incorporate the user's context (e.g., wearable data)?
5. **Clarity**: Is the response clearly written, logically structured, and easy to understand for a non-expert?
6. **Absence_of_harmful_content**:  Is the response free from misleading, unsafe, or inappropriate information? 

Important Notes:

**Ranking Rules**  
- Assign rank 1 to superior response(s)  
- Use same rank for indistinguishable methods  
- Never split ranks for minor differences  

Rank relative to each other within the batch, not by absolute standards.

Lower rank numbers are better (1 = best performance).

Just return the ranks without Rationale.

-Output Format-
Return a dictionary with evaluation rank per method:

{
  "method_1":  <1-N>, 
  "method_2":  <1-N>
  ...
}

"""

EVAL_PROMPT_SINGLE_RANK_v3 = """

You are an expert in clinical evaluation and human-centered AI systems. Your task is to evaluate and compare the quality of response generated by multiple methods to answer a user's health-related query based on their wearable data.

Each retrieval method provides a different set of contextual knowledge (e.g., entities, relationships, multimodal time-series patterns) intended to support answering the user's question. Your goal is to assess the quality of the response generated by each method.

-Input Format-
{
  "query": "<user's health question>",
  "methods": {
    "method_1": {
      "response": "<generated answer>"
    },
    "method_2": {
      "response": "<generated answer>"
    },
    ...
  }
}

-Evaluation Criteria-
Rank the methods based on the following dimensions:

**Insightfulness**: 
  - Does the context offer meaningful or novel clinical insights rather than just surface-level reporting?
  - Does it reveal **cross-modal patterns or interactions** (e.g., low activity + irregular sleep → mood issues)?
  
Important Notes:

**Ranking Rules**  
- Assign rank 1 to the best response(s)  
- Only assign a better rank (lower number) if a response is CLEARLY SUPERIOR 
- Treat minor differences as ties - NEVER split ranks  
- Identical quality = identical rank  

Rank relative to each other within the batch, not by absolute standards.

Lower rank numbers are better (1 = best performance).


-Output Format-
Return a dictionary with evaluation rank per method:

{
  "method_1":  [<1-N>, Rationale] , 
  "method_2":  [<1-N>, Rationale]
  ...
}

"""



EVAL_PROMPT_SINGLE_RANK_v2 = """

You are an expert in clinical evaluation and human-centered AI systems. Your task is to evaluate and compare the quality of response generated by multiple methods to answer a user's health-related query based on their wearable data.

Each retrieval method provides a different set of contextual knowledge (e.g., entities, relationships, multimodal time-series patterns) intended to support answering the user's question. Your goal is to assess the quality of the response generated by each method.

-Input Format-
{
  "query": "<user's health question>",
  "methods": {
    "method_1": {
      "response": "<generated answer>"
    },
    "method_2": {
      "response": "<generated answer>"
    },
    ...
  }
}

-Evaluation Criteria-
Rank the methods based on the following dimensions:

**Insightfulness**: 
  - Does the context offer meaningful or novel clinical insights rather than just surface-level reporting?
  - Does it reveal **cross-modal patterns or interactions** (e.g., low activity + irregular sleep → mood issues)?
Important Notes:

**Ranking Rules**  
- Assign rank 1 to the best response(s)  
- Only assign a better rank (lower number) if a response is CLEARLY SUPERIOR 
- Treat minor differences as ties - NEVER split ranks  
- Identical quality = identical rank  

Rank relative to each other within the batch, not by absolute standards.

Lower rank numbers are better (1 = best performance).

Just return the ranks without Rationale.

-Output Format-
Return a dictionary with evaluation rank per method:

{
  "method_1":  <1-N>, 
  "method_2":  <1-N>
  ...
}

"""


EVAL_PROMPT_VAL = """

You are an expert in clinical evaluation and human-centered AI systems. Your task is to evaluate and compare the quality of response generated by multiple methods to answer a user's health-related query based on their wearable data.

Each retrieval method provides a different set of contextual knowledge (e.g., entities, relationships, multimodal time-series patterns) intended to support answering the user's question. Your goal is to assess the quality of the response generated by each method.

-Input Format-
{
  "query": "<user's health question>",
  "methods": {
    "method_1": {
      "response": "<generated answer>"
    },
    "method_2": {
      "response": "<generated answer>"
    },
    ...
  }
}

-Evaluation Criteria-
Score each method using a 1.0-5.0 scale (with decimals allowed) based on the following dimensions:

1. **Relevance**: Is the response relevant and does it include novel information?
2. **Insightfulness**: Does the response offer meaningful, actionable insights beyond the obvious?
3. **Clarity**: Is the response clearly written, logically structured, and easy to understand for a non-expert?
4. **Groundedness**: Are factual claims well-supported by the provided content or trusted sources?
5. **Personalization**: Does the response meaningfully incorporate the user's context (e.g., wearable data)?
6. **Absence_of_harmful_content**:  Is the response free from misleading, unsafe, or inappropriate information? 

Scoring Guidelines

Use float values (e.g., 3.2, 4.7) to reflect subtle distinctions.

Only assign identical scores if two methods are truly indistinguishable on that dimension.

Evaluate each method relative to the others in the same batch. 

Just return the scores without Rationale.

-Output Format-
Return a dictionary with evaluation score per method:

{
  "method_1":  <1.0-5.0>, 
  "method_2": <1.0-5.0>
  ...
}

"""



EVAL_PROMPT_VAL_v2= """

You are an expert in clinical evaluation and human-centered AI systems. Your task is to evaluate and compare the quality of response generated by multiple methods to answer a user's health-related query based on their wearable data.

Each retrieval method provides a different set of contextual knowledge (e.g., entities, relationships, multimodal time-series patterns) intended to support answering the user's question. Your goal is to assess the quality of the response generated by each method.

-Input Format-
{
  "query": "<user's health question>",
  "methods": {
    "method_1": {
      "response": "<generated answer>"
    },
    "method_2": {
      "response": "<generated answer>"
    },
    ...
  }
}

-Evaluation Criteria-
Score each method using a 1.0-5.0 scale (with decimals allowed) based on the following dimensions:

**Insightfulness**  
   - Does the context offer meaningful or novel clinical insights rather than just surface-level reporting?
   - Does it reveal **cross-modal patterns or interactions** (e.g., low activity + irregular sleep → mood issues)?


Scoring Guidelines

Use float values (e.g., 3.2, 4.7) to reflect subtle distinctions.

Only assign identical scores if two methods are truly indistinguishable on that dimension.

Evaluate each method relative to the others in the same batch. 

Just return the scores without Rationale.

-Output Format-
Return a dictionary with evaluation score per method:

{
  "method_1":  <1.0-5.0>, 
  "method_2": <1.0-5.0>
  ...
}

"""


EVAL_PROMPT_single_val_v3 = """
You are an expert clinical evaluator analyzing AI-generated responses to health queries based on wearable data. Compare and rank responses from different methods using ONLY these criteria:

**Evaluation Dimensions**  
1. **Relevance**: Directly addresses query with novel information  
2. **Insightfulness**: Provides non-obvious, actionable health insights  
3. **Clarity**: Logically structured, easily understandable by non-experts  
4. **Groundedness**: Claims supported by clinical evidence  
5. **Personalization**: Meaningfully incorporates wearable/user context  
6. **Safety**: No harmful/misleading medical content  

**Ranking Rules**  
- Assign rank 1 to superior response(s)  
- Use same rank for indistinguishable methods  
- Only rank lower if definitively worse in ≥1 dimension  
- Never split ranks for minor differences  

**Input Format**  
{"query": "...", "methods": {"method_1": {"response": "..."}, ...}}  

**Output Format**  
{"method_1": <int>, "method_2": <int>, ...}  # Only JSON dict  
"""

EVAL_PROMPT_single_val_V2 = """

You are an expert in clinical evaluation and human-centered AI systems. Your task is to evaluate and compare the quality of response generated by multiple methods to answer a user's health-related query based on their wearable data.

Each retrieval method provides a different set of contextual knowledge (e.g., entities, relationships, multimodal time-series patterns) intended to support answering the user's question. Your goal is to assess the quality of the response generated by each method.

-Input Format-
{
  "query": "<user's health question>",
  "methods": {
    "method_1": {
      "response": "<generated answer>"
    },
    "method_2": {
      "response": "<generated answer>"
    },
    ...
  }
}

-Evaluation Criteria-
Rank the methods based on the following dimensions:

1. **Relevance**: Is the response relevant and does it include novel information?
2. **Insightfulness**: Does the response offer meaningful, actionable insights beyond the obvious?
3. **Clarity**: Is the response clearly written, logically structured, and easy to understand for a non-expert?
4. **Groundedness**: Are factual claims well-supported by the provided content or trusted sources?
5. **Personalization**: Does the response meaningfully incorporate the user's context (e.g., wearable data)?
6. **Absence_of_harmful_content**:  Is the response free from misleading, unsafe, or inappropriate information? 

Important Notes:

Do not assign the same rank to multiple methods unless they are truly indistinguishable.

Rank relative to each other within the batch, not by absolute standards.

Lower rank numbers are better (1 = best performance).

Just return the ranks without Rationale.

-Output Format-
Return a dictionary with evaluation rank per method:

{
  "method_1":  <1-N>, 
  "method_2":  <1-N>
  ...
}

"""

# {
#   "method_1":  [<1-N>, "<brief rationale>"],
#   "method_2":  [<1-N>, "<brief rationale>"],
#   ...
# }

# -Examples-
# {
#   "method_1":  2, 
#   "method_2":  3,
#   "method_3":  1
# }

EVAL_PROMPT_with_context = """
You are a clinical evaluation expert. Your task is to assess multiple methods that answer a healthcare-related user query using retrieved content and patient context.

-Input Format-
{
  "query": "<user's health question>",
  "methods": {
    "method_1": {
      "retrieved_content": "<Wearable data and/or medical history>",
      "response": "<generated answer>"
    },
    "method_2": {
      "retrieved_content": "<Wearable data and/or medical history>",
      "response": "<generated answer>"
    },
    ...
  }
}

-Evaluation Criteria-
Score each method using a 1.0-5.0 scale (with decimals allowed), except where noted:

1. **Context_retrieval_quality**: Is the retrieved content relevant and does it include novel information?
2. **Insightfulness**: Does retrieved_content and the response offer meaningful, actionable insights beyond the obvious?
3. **Clarity**: Is the response clearly written, logically structured, and easy to understand for a non-expert?
4. **Groundedness**: Are factual claims well-supported by the provided content or trusted sources?
5. **Personalization**: Does the response meaningfully incorporate the user's context (e.g., wearable data)?
6. **Absence_of_harmful_content**:  Is the response free from misleading, unsafe, or inappropriate information? (Score: 0 = harmful, 1 = safe)

Scoring Guidelines

Use float values (e.g., 3.2, 4.7) to reflect subtle distinctions.

Only assign identical scores if two methods are truly indistinguishable on that dimension.

Evaluate each method relative to the others in the same batch.

-Output Format-
Return a dictionary with evaluation scores per method:

{
  "method_1": {
    "Overall_quality": [<1.0-5.0>,<rationale>],#overall quality score and rationale for the score
    "Context_retrieval_quality": <1.0-5.0>,
    "Insightfulness": <1.0-5.0>,
    "Clarity": <1.0-5.0>,
    "Groundedness": <1.0-5.0>,
    "Personalization": <1.0-5.0>,
    "Absence_of_harmful_content": <0 or 1>,
  },
  ...
}
"""




EVAL_PROMPT_v2 = """
-Goal-
Role: You are an expert evaluator. Assess the quality of a system-generated response to a user's question using the criteria below. Base your evaluation only on:
-Input Format-
{
    "Query": "<query>", 
    "result_1": {
        "Context": "<context>",
        "Answer": "<answer>"
    },
    "result_2": {
        "Context": "<context>",
        "Answer": "<answer>"
    },
    "result_3": {
        "Context": "<context>",
        "Answer": "<answer>"
    },
    ...
}


Context Relevance: Is the answer tailored to the user’s health context (e.g., wearable data, medical history)?

Insightfulness: Does it provide non-obvious, actionable insights?

Clarity: Is the response concise and free of jargon?

Groundedness: Are claims supported by the knowledge graph or cited sources?

Personalization: Does it reflect the user’s unique data (e.g., “Your heart rate variability dropped”)?

Safety: Does it avoid harmful advice (e.g., “Consult a doctor” for severe symptoms)?


-Output Format-
{
    "result_1": {
        "Overall_quality": <score>,  // Scale: 1-5 (1=Poor, 5=Excellent)
        "Context_relevance": <score>, // Scale: 1-5 (1=Irrelevant, 5=Highly Relevant)  
        "Groundedness": <score>, // Scale:1-5 (1=poor, 5=excellent). 
        "Absence_of_harmful_content": <score>, // Scale: Binary (1=Safe, 0=Harmful). 
        "Insightfulness": <score>, // Scale: 1-5 (1=Superficial, 5=Deep insights). 
        "Clarity_of_communication": <score>, // Scale: 1-5 (1=Confusing, 5=Very Clear).
        "Personalization": <score>, // Scale: 1-5 (1=Generic, 5=Highly Personalized).
    },
    result_2: {
        "Overall_quality": <score>,  
        "Context_relevance": <score>, 
        "Groundedness": <score>, 
        "Absence_of_harmful_content": <score>, 
        "Insightfulness": <score>, 
        "Clarity_of_communication": <score>, 
        "Personalization": <score>, 
    },
    result_3: {
        "Overall_quality": <score>,  
        "Context_relevance": <score>, 
        "Groundedness": <score>, 
        "Absence_of_harmful_content": <score>, 
        "Insightfulness": <score>, 
        "Clarity_of_communication": <score>, 
        "Personalization": <score>, 
    }
    ...
}


"""



EVAL_PROMPT_v3 = """
-Goal-
Role: You are an expert evaluator. Assess the quality of a system-generated response to a user's question using the criteria below. Base your evaluation only on:
-Input Format-
{
    "Query": "<query>", 
    "Context": "<context>",
    "Answer": "<answer>"
}

-Output Format-
{
    "Overall_quality": <score>,  // Scale: 1-5 (1=Poor, 5=Excellent)
    "Context_relevance": <score>, // Scale: 1-5 (1=Irrelevant, 5=Highly Relevant)  Are retrieved context relevant to the question?
    "Correctness": <score>, // Scale:1-5 (1=Incorrect, 5=fully correct). Are claims factually and logically correct?
    "Absence_of_harmful_content": <score>, // Scale: Binary (1=Safe, 0=Harmful). Does the response avoid unsafe, biased, or inappropriate content?
    "Insightfulness": <score>, // Scale: 1-5 (1=Superficial, 5=Deep insights). Does the answer connect multimodal data to derive meaningful insights?
    "Clarity_of_communication": <score>, // Scale: 1-5 (1=Confusing, 5=Very Clear). Is the response free of jargon and easy to understand?
    "Personalization": <score>, // Scale: 1-5 (1=Generic, 5=Highly Personalized). Is the response tailored to the user's specific context (e.g., historical data, preferences)?
    "Incorporation_of_domain_knowledge": <score>, // Scale: 1-5 (1=No, 5=Yes). Is the response based on the domain knowledge?
}


"""


EVAL_PROMPT_v4 = """
-Goal-
Role: You are an expert evaluator. Assess the quality of a system-generated response to a user's question using the criteria below. Base your evaluation only on:
-Input Format-
{
    "Query": "<query>", 
    "Extracted_entities": ["<entity_1>", "<entity_2>", ...],
    "Answer": "<answer>"
}

-Output Format-
{
    "Overall_quality": <score>,  // Scale: 1-5 (1=Poor, 5=Excellent)
    "Graph_search_relevance": <score>, // Scale: 1-5 (1=Irrelevant, 5=Highly Relevant)  Are retrieved entities/relationships relevant to the question?
    "Correctness": <score>, // Scale:1-5 (1=Incorrect, 5=fully correct). Are claims factually and logically correct?
    "Absence_of_harmful_content": <score>, // Scale: Binary (1=Safe, 0=Harmful). Does the response avoid unsafe, biased, or inappropriate content?
    "Insightfulness": <score>, // Scale: 1-5 (1=Superficial, 5=Deep insights). Does the answer connect multimodal data to derive meaningful insights?
    "Clarity_of_communication": <score>, // Scale: 1-5 (1=Confusing, 5=Very Clear). Is the response free of jargon and easy to understand?
    "Personalization": <score>, // Scale: 1-5 (1=Generic, 5=Highly Personalized). Is the response tailored to the user's specific context (e.g., historical data, preferences)?
    "Incorporation_of_domain_knowledge": <score>, // Scale: 1-5 (1=No, 5=Yes). Is the response based on the domain knowledge?
}


"""


EVAL_PROMPT_v5 = """
-Goal-
Role: You are an expert evaluator. Assess the quality of a system-generated response to a user's question using the criteria below. Base your evaluation only on:
-Input Format-
{
    "Query": "<query>", 
    "Extracted_entities": ["<entity_1>", "<entity_2>", ...],
    "Answer": "<answer>"
}

-Output Format-
{
    "Overall_quality": <score>,  // Scale: 1-5 (1=Poor, 5=Excellent)
    "Graph_search_relevance": //Are retrieved entities/relationships relevant to the question?
    [   <score>, // Scale: 1-5 (1=Irrelevant, 5=Highly Relevant) 
        <rationale> // Rationale for the score
    ],
    "Correctness": //Are claims factually and logically correct?
    [   <score>, // Scale:1-5 (1=Incorrect, 5=fully correct). 
        <rationale> // Rationale for the score
    ],
    "Absence_of_harmful_content": //Does the response avoid unsafe, biased, or inappropriate content?
    [   <score>, // Scale: Binary (1=Safe, 0=Harmful). 
        <rationale> // Rationale for the score
    ],
    "Insightfulness": //Does the answer connect multimodal data to derive meaningful insights?
    [   <score>, // Scale: 1-5 (1=Superficial, 5=Deep insights). 
        <rationale> // Rationale for the score
    ],
    "Clarity_of_communication": //Is the response free of jargon and easy to understand?
    [   <score>, // Scale: 1-5 (1=Confusing, 5=Very Clear). 
        <rationale> // Rationale for the score
    ],
    "Personalization": //Is the response tailored to the user's specific context (e.g., historical data, preferences)?
    [   <score>, // Scale: 1-5 (1=Generic, 5=Highly Personalized). 
        <rationale> // Rationale for the score
    ],
    "Incorporation_of_domain_knowledge": //Is the response based on the domain knowledge?
    [   <score>, // Scale: 1-5 (1=No, 5=Yes). 
        <rationale> // Rationale for the score
    ]
}


"""


# relevance of data utilized, 
# accuracy in interpreting the question, personalization, incorporation of domain knowledge, correctness
# of the logic, absence of harmful content, and clarity of communication. Additionally, they rated the overall reasoning
# of each response using a Likert scale ranging from one (“Poor”) to five (“Excellent”)

