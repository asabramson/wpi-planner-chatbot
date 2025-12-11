# MY PROMPT

Hello, could I ask you for some advice on a good baseline architecture for fine-tuning an LLM? For context, I want to fine-tune an LLM to be an 'expert' on my university's (WPI) degree programs; essentially, it will play a similar role to an academic advisor. Students will be able to ask for advice on what courses to take based on their degree requirements and interests, when to take each course based on prerequisites and their graduation date, to summarize a course description, among other similar questions. I have access to the following data, which may be helpful to use:

1. The course titles, discipline (CS, MA, ECE, etc), course descriptions, and terms/times offered for every course.
2. A description of how WPI's curriculum works, including information on our term system, typical course workloads/normal practice for taking courses, how our project centers work/the effect they have on normal courses, how our BS/MS program works, among other important details that an academic advisor would be knowledgeable of.

How should I go about fine-tuning an LLM for this task? Please include a detailed step-by-step process that I can follow, thank you!

# CHATGPT RESPONSE

You’ll get *way* better results if you think of this as **“build an advisor system”** rather than “just fine-tune a model.”

For what you described (degree planning, prerequisites, course timing, BS/MS nuances), the sweet spot is usually:

> **Base LLM (general reasoning) + structured data + RAG + light fine-tuning**

Let’s walk through a concrete, step-by-step baseline architecture you can actually implement.

---

## 0. Big-picture architecture

Here’s the high-level design:

1. **Core LLM**: general-purpose model (e.g., GPT-style or open-source) for reasoning & natural language.
2. **Structured data store**:

   * Courses, prerequisites, offerings per term, degree requirements, policies.
   * Stored in a database (Postgres) or graph DB (Neo4j) or at least well-structured JSON.
3. **RAG layer**:

   * Converts user question → queries your data (SQL/graph/search).
   * Retrieves relevant courses, rules, policies, and feeds them to the LLM.
4. **Planning + validation logic**:

   * Some logic outside the model to:

     * enforce prerequisites,
     * ensure term availability,
     * check credit/requirement totals.
5. **(Optional) Fine-tuning**:

   * To make the model “speak like an advisor” and follow your institution-specific norms.

Most of the “correctness” will come from **good data modeling + retrieval**, not from fine-tuning alone.

---

## 1. Shape your data

You already have the most important pieces. Let’s shape them so a model/tool layer can use them.

### 1.1 Course catalog schema

Create a table or JSON schema roughly like:

```json
{
  "course_id": "CS 2102",
  "title": "Object-Oriented Design Concepts",
  "discipline": "CS",
  "description": "...",
  "terms_offered": ["A", "C"],        // WPI A–D term
  "credits": 1.0,                     // or equivalent
  "prerequisites": [
    {"type": "course", "id": "CS 1101"},
    {"type": "course", "id": "CS 2108"}
  ],
  "is_required_for": ["BS-CS", "BS-RBE"],
  "typical_year": "sophomore"
}
```

Store this in:

* A **relational DB** (Postgres, MySQL), or
* **Neo4j/graph DB** if you want fancy prerequisite path queries later.

### 1.2 Degree requirements

Represent degree programs as machine-readable rules:

```json
{
  "program_id": "BS-CS",
  "name": "Computer Science",
  "requirements": [
    {
      "id": "cs-core",
      "description": "CS Core",
      "type": "group_all",
      "courses": ["CS 1101", "CS 2102", "CS 2303", "CS 2223"]
    },
    {
      "id": "cs-3000-level",
      "description": "3000-level CS electives",
      "type": "min_credits",
      "credits_required": 3.0,
      "course_filter": {"discipline": "CS", "level_min": 3000}
    }
  ]
}
```

### 1.3 Policy / narrative content

For things like:

* term system explanation,
* normal course load per term,
* project center impact,
* BS/MS rules,

store as documents for retrieval:

```json
{
  "doc_id": "policy_term_system",
  "title": "WPI term system overview",
  "body": "WPI operates on a four-term academic calendar (A–D)...",
  "tags": ["policy", "terms", "overview"]
}
```

Later you’ll embed these and use them in RAG.

---

## 2. Decide: tools vs pure RAG vs both

For your use case, I’d recommend:

* **Tools/API calls for structured queries**, e.g.:

  * “find eligible CS courses next B term given I’ve taken X, Y”
* **RAG for unstructured explanation**, e.g.:

  * “how do project centers affect my course load?”

So:

1. The LLM interprets the user’s question.
2. It calls tools like:

   * `get_student_record(student_id)` (or takes structured input).
   * `get_degree_requirements(program_id)`.
   * `find_courses(filters...)`.
   * `build_candidate_plan(params...)`.
3. It uses the results (courses, schedules) to **explain** in natural language.

If you’re using an API like OpenAI’s with function calling, you’d define JSON schemas; if you’re using open-source, you can build a simple planner (manual “tool calling” pattern) around the model.

---

## 3. Retrieval layer (RAG)

This is for your **unstructured docs** (policies, explanations, maybe course descriptions).

### 3.1 Build a vector index

1. Chunk docs (e.g., 300–800 tokens).
2. Compute embeddings.
3. Store `(embedding, doc_id, metadata)` in a vector database (e.g., pgvector, Qdrant, Pinecone, Weaviate).

### 3.2 Query pattern

For each user query:

1. Embed the query.
2. Retrieve top *k* chunks (e.g., 5–10) from the index.
3. Add them to the model prompt under a **“context”** section.

Example system prompt snippet:

> You are an academic advisor bot for WPI. Use only the provided context about WPI’s curriculum, term system, and policies. If you are unsure or something is missing from the context, say so.

User prompt structure:

```text
[CONTEXT]
<top-k retrieved policy + course description snippets>

[USER QUESTION]
"I'm a CS major graduating in May 2027. I've finished CS 1101 and MA 1021.
What should I take in B term?"

[OPTIONAL STRUCTURED JSON]
{
  "student_program": "BS-CS",
  "expected_graduation_term": "D2027",
  "completed_courses": ["CS 1101", "MA 1021"]
}
```

---

## 4. Planning logic (course schedule construction)

Instead of expecting the LLM to perfectly enforce constraints, give it a helper layer.

### 4.1 Minimal baseline planner

Implement a small service that:

1. **Filters available courses** by term, program, and prerequisites:

   * Only courses whose `terms_offered` include the chosen term.
   * Only courses whose `prerequisites` ⊆ completed courses (or co-req rules).
2. **Classifies** them by requirement:

   * Required vs elective.
   * Fills missing requirement groups first.
3. **Constructs candidate term plans**:

   * Given a target per-term load (e.g., 3 courses A–D),
   * tries to assign courses across remaining terms.

Return to the LLM:

```json
{
  "candidate_plans": [
    {
      "plan_id": "plan_1",
      "timeline": [
        {
          "term": "B2025",
          "courses": ["CS 2102", "MA 1022", "HUA 1000-level elective"]
        },
        ...
      ],
      "notes": [
        "All prerequisites satisfied",
        "Leaves room in C-term for MQP planning"
      ]
    }
  ]
}
```

Then the LLM’s job is to:

* explain trade-offs,
* tailor based on interests (“more AI courses”, “fewer 8am classes”),
* highlight risks (“this plan assumes you can overload in C term”).

---

## 5. Where fine-tuning actually helps

You *do not* want to fine-tune the base model on all course data as plain text and hope it “memorizes” everything. That’s brittle and hard to update.

Instead, fine-tune (or at least use conversation examples in prompts) for:

1. **Tone & style**:

   * Empathetic, clear explanations.
   * Always suggests checking with a human advisor for final confirmation.
2. **Conversation patterns**:

   * Asking clarifying questions: “Are you double-majoring?”, “Do you want to study abroad?”
3. **Task format**:

   * Always responding with a structured section:

     * Summary,
     * Plan,
     * Checklist,
     * Follow-up questions.

### 5.1 Building the fine-tuning dataset

Create synthetic or semi-real conversation logs. Each training sample:

* **System + context**: “You’re WPI AdvisorBot…”
* **User message**: real or simulated query.
* **Assistant answer**: high-quality, manually crafted.

Example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are WPI AdvisorBot..."
    },
    {
      "role": "user",
      "content": "I'm a CS major planning to graduate in 2027..."
    },
    {
      "role": "assistant",
      "content": "Thanks for the details! Based on your plan..."
    }
  ]
}
```

Focus on:

* **Core scenarios**:

  * First-year planning,
  * Changing majors,
  * BS/MS interest,
  * MQP/IQP scheduling,
  * Overload risk conversations.
* **Edge cases**:

  * Missing prerequisites,
  * Course not offered in a given year,
  * Student behind on math or HUA,
  * Co-op / leave of absence.

Start with ~100–500 high-quality examples; quality > quantity.

---

## 6. Safety & guardrails (very important)

This is not just technical; you’re giving “advice” that affects graduation.

### 6.1 Hard constraints outside the model

* **NEVER** let the model be the single source of truth for:

  * degree completion,
  * graduation eligibility.
* Implement rule-based checks:

  * If `credits_remaining < 0` from model’s suggested plan → flag error.
  * If a required course is missing → flag & highlight.

### 6.2 LLM-side instructions

In your system prompt:

* “If you are not 100% sure about a policy, say so and recommend contacting a human advisor.”
* “Never guarantee that a student will graduate by a certain date; instead, present it as *likely* and suggest verification.”

Also instruct it to *never* fabricate courses. If uncertain, say:

> “I can’t find a course with that code in my data. Please verify in the official WPI course catalog.”

---

## 7. Evaluation loop

Before you let students touch it, do:

### 7.1 Static evals

Create a test set of ~50–100 scenarios:

* Inputs: student profile + question.
* Gold outputs: what a real advisor would say, or at least accept.

Check:

* **Constraint adherence**: no violating prerequisites, term offerings.
* **Coverage**: required courses not skipped.
* **Clarity**: student can actually follow the plan.

### 7.2 Ad-hoc advisor review

Have real WPI advisors (if you can) or senior students:

* Rate responses 1–5 for correctness + helpfulness.
* Tag serious errors (“dangerous”, “misleading”).

Use that feedback to:

* improve your planner logic,
* add new “training examples” to fine-tuning data,
* tweak prompts (e.g., be more conservative).

---

## 8. Concrete implementation roadmap

Here’s a possible step-by-step for you to follow:

1. **Data modeling (1–2 weeks)**

   * Build JSON/DB tables for:

     * courses,
     * degree requirements,
     * basic policies.
   * Write simple APIs: `list_courses`, `get_course`, `get_requirements`.

2. **Baseline RAG prototype (1 week)**

   * Build vector index for policy docs + course descriptions.
   * Implement: query → top-k context → LLM answer (no planning yet).
   * Use this for “what is X?” / “how does Y work?” questions.

3. **Planning logic (2–3 weeks)**

   * Implement:

     * prerequisite check,
     * term availability filter,
     * simple scheduler across A–D terms.
   * Wrap it in a function the LLM can call (or a pre-processing step).

4. **Advisor-style prompt tuning (few days)**

   * Craft a strong system prompt describing:

     * role,
     * limitations,
     * answer format.
   * Add a handful of in-context examples.

5. **Fine-tuning (optional, once base is solid)**

   * Prepare 100–500 conversation examples (JSONL).
   * Fine-tune the chosen base model.
   * Swap it into your pipeline (same RAG + tools).

6. **Testing & review**

   * Run static evaluation scenarios.
   * Have humans review.
   * Iterate on:

     * prompts,
     * planning rules,
     * what’s in your context window.

---

## 9. Practical tips and gotchas

* **Don’t hard-code today’s catalog into the model**:

  * Catalog changes annually. Keep data external and reloadable.
* **Entity normalization**:

  * Always use canonical IDs (“CS 2102”) internally, even if user types “cs2102” or “Object Oriented”.
* **Explain uncertainty explicitly**:

  * “Course offerings can change; check the current year’s catalog.”
* **Log everything**:

  * Store:

    * user input,
    * retrieved docs,
    * tools used,
    * final answer.
  * Makes debugging & improving the system way easier.

---

If you’d like, I can next:

* help you design **specific database schemas**, or
* sketch out **example prompts + tool definitions** for e.g. OpenAI function calling, or
* walk through one end-to-end example with a hypothetical WPI student and show how the system would handle it.