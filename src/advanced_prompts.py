"""Advanced prompt templates with chain-of-thought reasoning and few-shot examples."""

from typing import Tuple, List, Dict, Any


class FewShotExample:
    """Container for few-shot examples."""
    
    def __init__(self, transcript: str, note: str, analysis: Dict[str, Any], reasoning: str):
        self.transcript = transcript
        self.note = note
        self.analysis = analysis
        self.reasoning = reasoning
    
    def format(self) -> str:
        """Format example for prompt."""
        import json
        return f"""
Transcript: {self.transcript}

Generated Note: {self.note}

Reasoning Process:
{self.reasoning}

Analysis:
{json.dumps(self.analysis, indent=2)}
"""


class AdvancedPromptTemplates:
    """Advanced prompt templates with CoT reasoning and few-shot examples."""
    
    # Few-shot examples for hallucination detection
    HALLUCINATION_EXAMPLES = [
        FewShotExample(
            transcript="Patient reports headache for 2 days, no fever. Denies nausea.",
            note="Subjective: Patient has severe migraine with nausea and fever for 2 days.",
            analysis={
                "hallucinations": [
                    {
                        "fact": "severe migraine",
                        "severity": "medium",
                        "explanation": "Patient said 'headache' but note states 'severe migraine' - escalation without evidence",
                        "location": "Subjective"
                    },
                    {
                        "fact": "with nausea and fever",
                        "severity": "high",
                        "explanation": "Patient explicitly denied nausea. Fever not mentioned in transcript.",
                        "location": "Subjective"
                    }
                ],
                "hallucination_score": 0.3,
                "confidence": 0.95
            },
            reasoning="""
1. Compare transcript statements with note statements
2. Patient said "headache" → note says "severe migraine" (escalation)
3. Patient "denies nausea" → note says "with nausea" (contradiction)
4. "no fever" in transcript → note says "fever" (contradiction)
5. Two direct contradictions = high severity hallucinations
6. Score: 0.3 (poor) due to multiple fabricated facts
"""
        ),
        FewShotExample(
            transcript="Patient has persistent cough, productive with yellow sputum. No chest pain.",
            note="Subjective: Patient reports productive cough with yellowish sputum. Denies chest pain.",
            analysis={
                "hallucinations": [],
                "hallucination_score": 1.0,
                "confidence": 0.9
            },
            reasoning="""
1. Compare each fact in note with transcript
2. "productive cough" ✓ matches transcript
3. "yellowish sputum" ✓ matches "yellow sputum" (semantic match)
4. "denies chest pain" ✓ matches "no chest pain"
5. All facts supported by transcript
6. Score: 1.0 (excellent) - no hallucinations found
"""
        )
    ]
    
    # Few-shot examples for completeness
    COMPLETENESS_EXAMPLES = [
        FewShotExample(
            transcript="Patient is diabetic, takes metformin 500mg twice daily. Blood sugar has been high lately, around 200. Also mentions foot tingling.",
            note="Subjective: Patient reports elevated blood glucose levels.",
            analysis={
                "missing_items": [
                    {
                        "information": "Patient is diabetic (medical history)",
                        "severity": "high",
                        "explanation": "Critical medical history for context",
                        "location": "Subjective or HPI"
                    },
                    {
                        "information": "Takes metformin 500mg BID (current medications)",
                        "severity": "critical",
                        "explanation": "Current medication must be documented",
                        "location": "Subjective or Medications"
                    },
                    {
                        "information": "Foot tingling (concerning symptom)",
                        "severity": "high",
                        "explanation": "Potential diabetic neuropathy - requires attention",
                        "location": "Subjective"
                    }
                ],
                "completeness_score": 0.2,
                "confidence": 0.9
            },
            reasoning="""
1. Extract key clinical facts from transcript
2. Check if each fact is captured in note
3. Medical history: "diabetic" → MISSING
4. Medications: "metformin 500mg BID" → MISSING (critical)
5. Symptoms: "elevated blood glucose" ✓ present
6. Symptoms: "foot tingling" → MISSING (high priority)
7. Multiple critical omissions
8. Score: 0.2 (poor) due to missing essential information
"""
        )
    ]
    
    # Few-shot examples for clinical accuracy
    CLINICAL_ACCURACY_EXAMPLES = [
        FewShotExample(
            transcript="Patient has intermittent chest discomfort, worse with exertion. No radiation to arms.",
            note="Assessment: Likely viral infection. Plan: Rest and hydration.",
            analysis={
                "accuracy_issues": [
                    {
                        "issue": "Chest discomfort worse with exertion not properly evaluated",
                        "severity": "critical",
                        "explanation": "Exertional chest pain could indicate cardiac issues. Dismissing as viral infection is dangerous.",
                        "location": "Assessment",
                        "correction": "Should consider cardiac workup given exertional symptoms"
                    }
                ],
                "accuracy_score": 0.2,
                "confidence": 0.95
            },
            reasoning="""
1. Analyze clinical reasoning in note
2. Symptom: chest discomfort + worse with exertion
3. Red flag: exertional chest symptoms suggest cardiac etiology
4. Assessment states: "viral infection"
5. Critical error: No cardiac evaluation despite concerning presentation
6. Plan inadequate: "rest and hydration" misses serious pathology
7. Score: 0.2 (poor) - dangerous clinical reasoning
"""
        )
    ]
    
    @staticmethod
    def hallucination_detection_cot() -> Tuple[str, str]:
        """Enhanced hallucination detection prompt with CoT and few-shot examples."""
        
        few_shot_text = "\n\n".join([
            f"=== EXAMPLE {i+1} ==={ex.format()}"
            for i, ex in enumerate(AdvancedPromptTemplates.HALLUCINATION_EXAMPLES)
        ])
        
        system_prompt = f"""You are an expert medical documentation reviewer specializing in fact verification and hallucination detection.

Your task is to identify information in the generated SOAP note that is NOT supported by the source transcript.

DEFINITIONS:
- HALLUCINATION: Any fact in the note that cannot be verified from the transcript
- CONTRADICTION: Information that directly conflicts with the transcript
- ESCALATION: Upgrading severity/specificity without evidence (e.g., "headache" → "migraine")
- FABRICATION: Adding specific details not mentioned (e.g., dosages, measurements, diagnoses)

SEVERITY GUIDELINES:
- CRITICAL: Contradicts transcript or adds dangerous misinformation
- HIGH: Significant fabricated facts that change clinical picture
- MEDIUM: Minor escalations or unsupported details
- LOW: Trivial differences in wording that don't change meaning

STEP-BY-STEP REASONING PROCESS:
1. Read the transcript carefully and list all key facts
2. Read the generated note and list all claims
3. For each claim in the note, find supporting evidence in transcript
4. Mark any claim without clear support as potential hallucination
5. Classify severity based on clinical impact
6. Calculate confidence based on clarity of evidence
7. Assign overall score (0=many hallucinations, 1=none)

FEW-SHOT EXAMPLES:
{few_shot_text}

IMPORTANT:
- Be precise: Only flag actual unsupported facts
- Don't flag: Standard medical terminology, SOAP formatting, reasonable clinical language
- Do flag: Specific claims, measurements, diagnoses, medications not in transcript
- Consider semantics: "denies fever" ≈ "afebrile" (OK), but "102°F fever" when not mentioned (NOT OK)

RESPONSE FORMAT:
{{
  "reasoning_steps": [
    "Step 1: Key facts from transcript...",
    "Step 2: Claims in note...",
    "Step 3: Verification results...",
    "Step 4: Identified hallucinations..."
  ],
  "hallucinations": [
    {{
      "fact": "specific hallucinated statement from note",
      "severity": "critical/high/medium/low",
      "explanation": "why this is unsupported or contradicted",
      "location": "which SOAP section (S/O/A/P)",
      "transcript_evidence": "relevant quote from transcript or 'none found'"
    }}
  ],
  "hallucination_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "confidence_explanation": "why this confidence level"
}}"""
        
        user_template = """Transcript:
{transcript}

Generated SOAP Note:
{generated_note}

Think step-by-step and identify any hallucinated or unsupported facts in the note."""
        
        return system_prompt, user_template
    
    @staticmethod
    def completeness_check_cot() -> Tuple[str, str]:
        """Enhanced completeness check prompt with CoT and few-shot examples."""
        
        few_shot_text = "\n\n".join([
            f"=== EXAMPLE {i+1} ==={ex.format()}"
            for i, ex in enumerate(AdvancedPromptTemplates.COMPLETENESS_EXAMPLES)
        ])
        
        system_prompt = f"""You are an expert medical documentation reviewer specializing in completeness assessment.

Your task is to identify clinically significant information from the transcript that is missing from the generated note.

CATEGORIES OF CRITICAL INFORMATION:
1. Chief Complaint: Primary reason for visit
2. Medical History: Chronic conditions, past diagnoses
3. Current Medications: All medications with dosages
4. Symptoms: All reported symptoms, especially concerning ones
5. Physical Findings: Examination results
6. Diagnoses: Clinical impressions
7. Treatment Plan: Medications, procedures, referrals
8. Follow-up: Instructions and timeline

SEVERITY GUIDELINES:
- CRITICAL: Missing info that affects immediate patient safety (medications, allergies, severe symptoms)
- HIGH: Important clinical information that impacts care plan (chronic conditions, key symptoms)
- MEDIUM: Relevant information that provides context (minor symptoms, social history)
- LOW: Supplementary information that doesn't affect clinical decisions

STEP-BY-STEP REASONING PROCESS:
1. Extract all clinical facts from transcript by category
2. Check each fact against the generated note
3. For missing facts, assess clinical significance
4. Classify by severity based on impact on patient care
5. Calculate confidence based on clarity of omission
6. Assign overall score (0=major gaps, 1=complete)

FEW-SHOT EXAMPLES:
{few_shot_text}

IMPORTANT:
- Focus on clinical significance, not verbatim transcription
- Don't flag: Conversational filler, redundant information, info captured in different words
- Do flag: Medications, allergies, key symptoms, diagnoses, treatment plans
- Consider semantic equivalence: Different phrasing of same concept is OK

RESPONSE FORMAT:
{{
  "reasoning_steps": [
    "Step 1: Clinical facts from transcript...",
    "Step 2: Check against note...",
    "Step 3: Identify gaps...",
    "Step 4: Assess significance..."
  ],
  "missing_items": [
    {{
      "information": "what specific information is missing",
      "severity": "critical/high/medium/low",
      "explanation": "why this is important to document",
      "location": "where it should appear (S/O/A/P)",
      "transcript_evidence": "quote from transcript showing this info"
    }}
  ],
  "completeness_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "confidence_explanation": "why this confidence level"
}}"""
        
        user_template = """Transcript:
{transcript}

Generated SOAP Note:
{generated_note}

Think step-by-step and identify any critical information from the transcript that is missing from the note."""
        
        return system_prompt, user_template
    
    @staticmethod
    def clinical_accuracy_cot() -> Tuple[str, str]:
        """Enhanced clinical accuracy prompt with CoT and few-shot examples."""
        
        few_shot_text = "\n\n".join([
            f"=== EXAMPLE {i+1} ==={ex.format()}"
            for i, ex in enumerate(AdvancedPromptTemplates.CLINICAL_ACCURACY_EXAMPLES)
        ])
        
        system_prompt = f"""You are an expert clinical documentation reviewer specializing in medical accuracy and safety.

Your task is to identify medically problematic, inaccurate, or potentially dangerous content in SOAP notes.

CATEGORIES OF ACCURACY ISSUES:
1. Incorrect Terminology: Misuse of medical terms
2. Logical Inconsistency: Contradictions between sections (e.g., S/O vs A/P)
3. Inappropriate Assessment: Diagnosis doesn't match presentation
4. Unsafe Treatment: Dangerous or contraindicated recommendations
5. Missing Workup: Failure to address red flags
6. Misrepresentation: Distorting patient's presentation

SEVERITY GUIDELINES:
- CRITICAL: Could lead to immediate patient harm (wrong meds, missed emergencies)
- HIGH: Significant clinical errors affecting care quality
- MEDIUM: Minor inaccuracies that should be corrected
- LOW: Suboptimal phrasing without clinical impact

CLINICAL RED FLAGS TO WATCH:
- Chest pain + exertion = cardiac workup needed
- Severe headache + sudden onset = serious causes
- Abdominal pain + certain patterns = surgical emergency
- Pediatric fever + specific symptoms = sepsis risk
- Altered mental status = many serious causes

STEP-BY-STEP REASONING PROCESS:
1. Review clinical presentation from transcript
2. Analyze assessment and plan in note
3. Check for logical consistency (do symptoms match diagnosis?)
4. Verify treatment appropriateness
5. Identify any red flags that weren't addressed
6. Assess clinical safety
7. Calculate confidence based on severity of issues
8. Assign overall score (0=major issues, 1=accurate)

FEW-SHOT EXAMPLES:
{few_shot_text}

IMPORTANT:
- Consider standard of care and clinical guidelines
- Focus on patient safety and quality of care
- Don't flag: Reasonable clinical judgment, appropriate alternatives
- Do flag: Dangerous recommendations, missed red flags, logical errors

RESPONSE FORMAT:
{{
  "reasoning_steps": [
    "Step 1: Clinical presentation analysis...",
    "Step 2: Assessment evaluation...",
    "Step 3: Treatment appropriateness...",
    "Step 4: Red flag analysis...",
    "Step 5: Safety assessment..."
  ],
  "accuracy_issues": [
    {{
      "issue": "description of the clinical problem",
      "severity": "critical/high/medium/low",
      "explanation": "why this is medically problematic",
      "location": "where in note (S/O/A/P)",
      "clinical_impact": "potential consequence for patient",
      "correction": "what should be stated/done instead"
    }}
  ],
  "accuracy_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "confidence_explanation": "why this confidence level"
}}"""
        
        user_template = """Transcript:
{transcript}

Generated SOAP Note:
{generated_note}

Think step-by-step and identify any clinical accuracy issues or medically problematic content in the note."""
        
        return system_prompt, user_template
    
    @staticmethod
    def semantic_coherence() -> Tuple[str, str]:
        """Prompt for evaluating semantic coherence of the note."""
        system_prompt = """You are an expert in medical documentation quality, specializing in semantic coherence and readability.

Your task is to evaluate the internal consistency, logical flow, and semantic coherence of SOAP notes.

EVALUATION CRITERIA:
1. **Logical Flow**: Does information flow naturally from S → O → A → P?
2. **Internal Consistency**: Do different sections align with each other?
3. **Terminology Consistency**: Are medical terms used consistently?
4. **Clarity**: Is the note clear and unambiguous?
5. **Completeness of Reasoning**: Does the assessment logically follow from S/O?
6. **Plan Alignment**: Does the plan address the assessment?

COHERENCE ISSUES:
- Contradictions between sections
- Assessment doesn't match symptoms
- Plan doesn't address assessment
- Inconsistent terminology
- Unclear or ambiguous statements
- Missing logical connections

STEP-BY-STEP PROCESS:
1. Analyze each SOAP section individually
2. Check consistency between sections
3. Verify logical flow of clinical reasoning
4. Assess clarity and readability
5. Identify any semantic issues
6. Provide overall coherence score

RESPONSE FORMAT:
{{
  "reasoning_steps": ["Analysis of each section..."],
  "coherence_issues": [
    {{
      "issue": "description of coherence problem",
      "severity": "high/medium/low",
      "sections_affected": ["S", "A", "P"],
      "explanation": "why this affects coherence",
      "example": "specific text showing issue"
    }}
  ],
  "semantic_coherence_score": 0.0-1.0,
  "readability_score": 0.0-1.0,
  "logical_consistency_score": 0.0-1.0,
  "confidence": 0.0-1.0
}}"""
        
        user_template = """Generated SOAP Note:
{generated_note}

Evaluate the semantic coherence and internal consistency of this note."""
        
        return system_prompt, user_template
    
    @staticmethod
    def temporal_consistency() -> Tuple[str, str]:
        """Prompt for checking temporal consistency."""
        system_prompt = """You are an expert in medical documentation, specializing in temporal accuracy and timeline coherence.

Your task is to verify that temporal information (dates, durations, sequences) is consistent and logical.

TEMPORAL ELEMENTS TO CHECK:
1. **Symptom Duration**: "for 3 days", "since Monday", "chronic"
2. **Event Sequence**: "started after", "before the procedure", "during"
3. **Medication Timeline**: "started 2 weeks ago", "taking for 6 months"
4. **Follow-up Timing**: "return in 2 weeks", "if no improvement in 3 days"
5. **Historical Events**: Past diagnoses, procedures, timeline alignment

COMMON TEMPORAL ISSUES:
- Inconsistent durations (says "3 days" in one place, "1 week" elsewhere)
- Illogical sequences (effect before cause)
- Missing temporal markers for important events
- Contradictory timelines
- Unclear temporal relationships

STEP-BY-STEP PROCESS:
1. Extract all temporal references from transcript
2. Extract all temporal references from note
3. Check consistency between transcript and note
4. Check internal consistency within note
5. Verify logical sequence of events
6. Assess clarity of timeline

RESPONSE FORMAT:
{{
  "reasoning_steps": ["Timeline extraction and analysis..."],
  "temporal_issues": [
    {{
      "issue": "description of temporal problem",
      "severity": "high/medium/low",
      "location": "where in note",
      "transcript_timeline": "timeline from transcript",
      "note_timeline": "timeline from note",
      "explanation": "why this is problematic"
    }}
  ],
  "temporal_consistency_score": 0.0-1.0,
  "timeline_clarity_score": 0.0-1.0,
  "confidence": 0.0-1.0
}}"""
        
        user_template = """Transcript:
{transcript}

Generated SOAP Note:
{generated_note}

Evaluate the temporal consistency and timeline accuracy in this note."""
        
        return system_prompt, user_template
    
    @staticmethod
    def clinical_reasoning_quality() -> Tuple[str, str]:
        """Prompt for evaluating quality of clinical reasoning."""
        system_prompt = """You are an expert physician educator, specializing in clinical reasoning assessment.

Your task is to evaluate the quality of clinical reasoning demonstrated in the SOAP note.

CLINICAL REASONING ELEMENTS:
1. **Differential Diagnosis**: Are alternative diagnoses considered?
2. **Evidence Integration**: How well are symptoms/signs integrated?
3. **Risk Stratification**: Are risks appropriately assessed?
4. **Diagnostic Clarity**: Is the assessment specific and justified?
5. **Treatment Rationale**: Is the plan justified by the assessment?
6. **Follow-up Logic**: Are follow-up plans appropriate?

REASONING QUALITY LEVELS:
- **Excellent** (>0.8): Comprehensive, evidence-based, considers alternatives
- **Good** (0.6-0.8): Sound reasoning with minor gaps
- **Adequate** (0.4-0.6): Basic reasoning, some important gaps
- **Poor** (<0.4): Significant reasoning flaws, missing key considerations

RED FLAGS IN REASONING:
- Jumping to conclusions without adequate workup
- Ignoring contradictory evidence
- Failing to consider serious diagnoses
- Treatment not matching assessment
- No safety netting for uncertainty

STEP-BY-STEP PROCESS:
1. Analyze clinical presentation
2. Evaluate assessment quality
3. Check if differential considered
4. Assess treatment logic
5. Verify appropriate follow-up
6. Identify reasoning strengths/weaknesses

RESPONSE FORMAT:
{{
  "reasoning_steps": ["Analysis of clinical reasoning..."],
  "reasoning_strengths": ["List of good reasoning elements..."],
  "reasoning_weaknesses": [
    {{
      "weakness": "description",
      "severity": "high/medium/low",
      "explanation": "impact on care quality",
      "recommendation": "how to improve"
    }}
  ],
  "differential_diagnosis_score": 0.0-1.0,
  "evidence_integration_score": 0.0-1.0,
  "treatment_rationale_score": 0.0-1.0,
  "overall_reasoning_quality_score": 0.0-1.0,
  "confidence": 0.0-1.0
}}"""
        
        user_template = """Transcript:
{transcript}

Generated SOAP Note:
{generated_note}

Evaluate the quality of clinical reasoning in this note."""
        
        return system_prompt, user_template
