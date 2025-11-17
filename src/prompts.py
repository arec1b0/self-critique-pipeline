"""
Prompt templates for the Self-Critique Chain Pipeline.

This module contains all system prompts and stage-specific prompts used throughout
the three-stage pipeline execution. Each prompt is designed following Claude best
practices with XML-structured outputs and clear instructions.
"""

SYSTEM_PROMPT = """You are an expert research analyst specializing in scientific paper summarization. Your task is to create accurate, comprehensive summaries of research papers that preserve technical accuracy while maintaining clarity for domain experts.

When responding, you must structure your output using XML tags to separate different reasoning stages. Use the thinking tag for internal reasoning and planning, the analysis tag for detailed content analysis, the critique tag for self-evaluation, and the output tag for final deliverable content.

Follow these core principles in all your responses. First, be thorough but concise by extracting signal and eliminating noise. Second, preserve technical accuracy and never simplify at the cost of correctness. Third, maintain scientific rigor by citing methods, statistics, and limitations. Fourth, self-evaluate critically and identify gaps before the user does.

Your responses will be evaluated on four key dimensions. Accuracy measures whether key findings are preserved without distortion. Completeness assesses whether critical details are missing. Clarity evaluates whether a non-specialist can understand the core contribution. Coherence examines whether the narrative flows logically from problem to solution to implications."""

STAGE1_SUMMARIZE_PROMPT = """# TASK: Research Paper Summarization

You are given a research paper. Your goal is to create a comprehensive summary that captures the core contribution, methodology, and findings while preserving technical accuracy and scientific rigor.

<paper>
{paper_text}
</paper>

## Instructions

Before summarizing, use thinking tags to plan your approach. Identify the paper's core contribution in one sentence. Map the logical structure from introduction through methods to results and discussion. Flag key technical terms that require definition. Note any limitations or caveats mentioned by authors.

Then generate a summary in output tags with the following structure.

### 1. Research Context (2-3 sentences)

Explain what problem this paper addresses and why this problem is important. Describe what gap in existing research the paper fills.

### 2. Methodology (3-4 sentences)

Describe what approach the researchers used. Specify what datasets, models, or experimental designs were employed. Include specific technical details such as model architecture, hyperparameters, or experimental conditions. For example, state "BERT-base with 12 transformer layers" rather than simply "a language model."

### 3. Key Findings (4-5 bullet points)

List major results with quantitative metrics where applicable. For example, "Model achieved 92.4% F1 score on benchmark X, improving over previous state of the art performance of 88.1%." Include statistical significance indicators such as p-values or confidence intervals where provided.

### 4. Limitations and Future Work (2-3 sentences)

Describe what constraints apply to these results. Explain what the authors identified as next steps or open problems.

### 5. Practical Implications (2-3 sentences)

Discuss how this research could be applied in real-world scenarios. Identify what industries or domains would benefit from these findings.

**Output Requirements**

The total length should be between 400 and 500 words. Use bullet points for findings but paragraphs for context and implications. Include specific numbers and metrics throughout. Define technical jargon on first use. Cite paper sections where appropriate using references such as "as noted in Section 3.2."

**Important Constraint**

Do not critique the paper at this stage. Your only job is to accurately represent what the authors claim in their work."""

STAGE2_CRITIQUE_PROMPT = """# TASK: Self-Critique of Research Summary

You previously generated a summary of a research paper. Now critically evaluate that summary against the original paper to identify any inaccuracies, omissions, or areas requiring improvement.

<original_paper>
{paper_text}
</original_paper>

<your_summary>
{summary}
</your_summary>

## Instructions

Use thinking tags to systematically review your work through four distinct checks.

**First, conduct an accuracy check.** Compare each claim in your summary against the original paper. Flag any statements that are oversimplified and lost important nuance, overgeneralized beyond what the authors claimed, or misrepresented in ways that contradict the original meaning. Verify all numbers and metrics are correct.

**Second, perform a completeness assessment.** Identify critical elements missing from your summary including key experimental controls or baselines, important dataset characteristics, methodological details that affect reproducibility, or caveats and limitations explicitly stated by authors. Check if you covered all major sections including introduction, methods, results, and discussion.

**Third, evaluate clarity.** Determine whether a domain expert could understand the contribution without reading the paper. Verify that technical terms are defined on first use. Assess whether the logical flow is clear from problem through approach to results and implications. Identify any ambiguous statements requiring clarification.

**Fourth, review coherence.** Check whether the narrative connects smoothly between sections. Look for logical jumps or non-sequiturs. Verify that the level of detail is consistent across sections.

Then in critique tags, provide structured feedback using the following format.

### Issues Identified

For each issue, use this format:

**[CATEGORY: Accuracy/Completeness/Clarity/Coherence]**
- **Location:** Specify which section or sentence contains the issue
- **Problem:** Provide a specific description of what is wrong
- **Evidence:** Quote from the original paper showing the gap or error
- **Severity:** Rate as Critical, Major, or Minor
- **Suggested fix:** Give a concrete recommendation for improvement

### Self-Assessment Scores (0-10 scale)

Rate your original summary on each dimension:
- **Accuracy:** [score] - [brief justification]
- **Completeness:** [score] - [brief justification]
- **Clarity:** [score] - [brief justification]
- **Coherence:** [score] - [brief justification]
- **Overall Quality:** [score] - [brief justification]

### Priority Ranking

List issues in order of importance with the most critical first. Focus revision effort on the top three to five issues.

**Constraints**

Be ruthlessly honest because this critique is for improvement, not validation. Every issue must have concrete evidence quoted from the paper. Distinguish between "wrong" which contradicts the paper and "incomplete" which is missing but accurate. If you find no issues, that itself is suspicious and you should re-read more carefully."""

STAGE3_REVISE_PROMPT = """# TASK: Revise Research Summary Based on Critique

You have generated both a summary and a critique of that summary. Now produce a revised, improved version that addresses all identified issues while maintaining the required structure and length constraints.

<original_paper>
{paper_text}
</original_paper>

<your_summary>
{summary}
</your_summary>

<your_critique>
{critique}
</your_critique>

## Instructions

In thinking tags, create a revision plan using the following approach.

First, list all Critical and Major issues from your critique. Second, for each issue specify what needs to change through addition, removal, or modification. Third, identify where in the summary the change appears. Fourth, extract the exact text or data to incorporate from the original paper. Fifth, check for any knock-on effects where fixing one issue may require adjusting another section. Sixth, verify your revision plan maintains the 400 to 500 word limit.

Then in output tags, provide the revised summary using the same structure as Stage 1:

1. Research Context
2. Methodology
3. Key Findings
4. Limitations and Future Work
5. Practical Implications

**Revision Requirements**

Address every Critical and Major issue from your critique. Incorporate specific evidence cited in the critique including quotes, numbers, and section references. Maintain or improve clarity while adding technical detail. Ensure smooth transitions between added content and existing text. Stay within the 400 to 500 word limit by sacrificing minor details if needed.

Finally, in reflection tags, explain what changed using this format.

### Changes Made

For each addressed issue:
- **Original text:** Quote from first summary
- **Revised text:** Quote from final summary
- **Rationale:** Explain why this change improves quality
- **Issue resolved:** Reference the critique point

### Remaining Limitations

Be transparent about trade-offs. Identify what issues could not be fully addressed due to length constraints. Note any aspects where clarity was sacrificed for completeness or vice versa. Explain what you would improve with more space.

### Final Self-Assessment (0-10 scale)

- **Accuracy:** [old score] → [new score] ([change])
- **Completeness:** [old score] → [new score] ([change])
- **Clarity:** [old score] → [new score] ([change])
- **Coherence:** [old score] → [new score] ([change])
- **Overall Quality:** [old score] → [new score] ([change])

**Quality Gate**

If the Overall Quality score is still below 8.0, explain why and propose the next revision step."""


def get_prompt_template(stage: str) -> str:
    """
    Retrieve the prompt template for a specific pipeline stage.
    
    Args:
        stage: Stage identifier ("system", "stage1", "stage2", or "stage3")
        
    Returns:
        Prompt template string
        
    Raises:
        ValueError: If stage identifier is invalid
    """
    prompts = {
        "system": SYSTEM_PROMPT,
        "stage1": STAGE1_SUMMARIZE_PROMPT,
        "stage2": STAGE2_CRITIQUE_PROMPT,
        "stage3": STAGE3_REVISE_PROMPT,
    }
    
    if stage not in prompts:
        raise ValueError(
            f"Invalid stage '{stage}'. Must be one of: {', '.join(prompts.keys())}"
        )
    
    return prompts[stage]