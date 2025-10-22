"""Generate synthetic SOAP notes for testing."""

import json
import argparse
from pathlib import Path


SYNTHETIC_NOTES = [
    {
        "id": "synthetic_001",
        "transcript": """
        Patient is a 45-year-old male presenting with complaints of intermittent chest pain 
        for the past 2 weeks. Pain is described as pressure-like, located in the center of 
        the chest, radiating to the left arm. Occurs with exertion, relieved by rest. 
        Duration approximately 5-10 minutes per episode. No shortness of breath. 
        Patient has history of hypertension, currently on lisinopril 10mg daily. 
        Father had myocardial infarction at age 55. Patient smokes half pack per day for 20 years.
        
        Physical examination: BP 145/92, HR 78, regular rhythm. Heart sounds normal, 
        no murmurs. Lungs clear bilaterally. No peripheral edema.
        
        EKG shows normal sinus rhythm, no acute ST changes.
        """,
        "generated_note": """
        SUBJECTIVE:
        45-year-old male with 2 weeks of intermittent chest pain. Describes pressure-like 
        sensation in center of chest with radiation to left arm. Occurs with exertion, 
        relieved by rest, lasting 5-10 minutes. Denies SOB. PMH significant for HTN on lisinopril.
        Family history positive for MI in father at age 55. Current smoker, 10 pack-years.
        
        OBJECTIVE:
        Vital signs: BP 145/92, HR 78
        Cardiovascular: Regular rhythm, normal S1/S2, no murmurs
        Respiratory: Clear to auscultation bilaterally
        Extremities: No edema
        EKG: Normal sinus rhythm, no acute changes
        
        ASSESSMENT:
        Chest pain, likely angina pectoris. Rule out acute coronary syndrome.
        Hypertension, suboptimal control.
        
        PLAN:
        1. Order troponin levels, lipid panel
        2. Schedule stress test
        3. Continue lisinopril, may need to uptitrate
        4. Aspirin 81mg daily
        5. Smoking cessation counseling
        6. Follow-up in 1 week or sooner if symptoms worsen
        """,
        "reference_note": """
        SUBJECTIVE:
        45-year-old male with 2 weeks of intermittent chest pain. Describes pressure-like 
        sensation in center of chest with radiation to left arm. Occurs with exertion, 
        relieved by rest, lasting 5-10 minutes. Denies shortness of breath. Past medical 
        history significant for hypertension on lisinopril 10mg daily. Family history 
        positive for MI in father at age 55. Current smoker, 10 pack-years.
        
        OBJECTIVE:
        Vital signs: BP 145/92, HR 78, RR 16, Temp 98.6F
        Cardiovascular: Regular rate and rhythm, normal S1/S2, no murmurs, rubs, or gallops
        Respiratory: Clear to auscultation bilaterally, no wheezes or rales
        Extremities: No peripheral edema
        EKG: Normal sinus rhythm, no acute ST-T wave changes
        
        ASSESSMENT:
        1. Chest pain, concerning for stable angina pectoris, rule out ACS
        2. Hypertension, suboptimally controlled
        3. Tobacco use disorder
        
        PLAN:
        1. Laboratory: Troponin, lipid panel, CMP
        2. Cardiology consult for stress test
        3. Continue lisinopril 10mg daily, consider uptitration based on BP monitoring
        4. Start aspirin 81mg daily for cardiovascular protection
        5. Smoking cessation counseling, consider pharmacotherapy
        6. Return precautions discussed - return immediately if chest pain worsens or occurs at rest
        7. Follow-up appointment in 1 week
        """,
        "metadata": {"source": "synthetic", "scenario": "chest_pain_good"}
    },
    {
        "id": "synthetic_002",
        "transcript": """
        Patient is a 32-year-old female presenting with severe headache for 3 days. 
        Headache is bilateral, throbbing in nature, rated 8/10 in severity. 
        Associated with nausea but no vomiting. Photophobia present. 
        No fever, no neck stiffness. Has history of occasional headaches but this is worse.
        Takes ibuprofen with minimal relief. No recent trauma.
        
        Vital signs: BP 118/76, HR 72, Temp 98.4F
        Neurological exam: Alert and oriented x3, cranial nerves intact, 
        no focal deficits, no papilledema on fundoscopic exam.
        """,
        "generated_note": """
        SUBJECTIVE:
        32-year-old female with migraine headache for 3 days accompanied by aura and confusion.
        Patient reports severe pain with frequent vomiting and fever. Pain is worse in the morning.
        Patient has tried multiple medications without relief.
        
        OBJECTIVE:
        BP 118/76, HR 72, Temp 101.2F
        Neuro exam shows some confusion and difficulty with coordination.
        
        ASSESSMENT:
        Complicated migraine with neurological symptoms
        
        PLAN:
        CT head ordered. Start IV medications. Admit for observation.
        """,
        "reference_note": """
        SUBJECTIVE:
        32-year-old female with 3-day history of severe bilateral throbbing headache, 
        rated 8/10. Associated with nausea (no vomiting) and photophobia. 
        No fever, no neck stiffness. History of occasional headaches but this episode 
        is more severe. Tried ibuprofen with minimal relief. No recent head trauma.
        
        OBJECTIVE:
        Vital signs: BP 118/76, HR 72, Temp 98.4F
        Neurological: Alert and oriented x3, CN II-XII intact, no focal motor or 
        sensory deficits, no papilledema on fundoscopic examination, no meningismus
        
        ASSESSMENT:
        Severe headache, most consistent with migraine without aura
        
        PLAN:
        1. Sumatriptan 100mg PO for acute treatment
        2. Metoclopramide 10mg PO for nausea
        3. Recommend dark, quiet environment
        4. Follow-up in 48 hours if not improving
        5. Return immediately if fever, neck stiffness, or neurological symptoms develop
        6. Consider prophylactic therapy if headaches become more frequent
        """,
        "metadata": {"source": "synthetic", "scenario": "headache_hallucination"}
    },
    {
        "id": "synthetic_003",
        "transcript": """
        Patient is 68-year-old male with type 2 diabetes, hypertension, and 
        hyperlipidemia presenting for routine follow-up. Last seen 3 months ago.
        Blood sugars have been running 140-180 fasting. Takes metformin 1000mg twice daily,
        lisinopril 20mg daily, and atorvastatin 40mg at bedtime. Checking blood sugars 
        twice daily. Diet has been good. Exercising 30 minutes walking 5 days per week.
        No chest pain, no shortness of breath, no leg pain with walking.
        
        Vital signs: BP 132/84, HR 68, Weight 195 lbs (down 3 lbs from last visit)
        A1C today: 7.8% (previous 8.2%)
        LDL: 95 mg/dL
        Creatinine: 1.1 mg/dL
        Foot exam: No ulcers, monofilament intact bilaterally
        """,
        "generated_note": """
        SUBJECTIVE:
        68-year-old male with type 2 diabetes here for follow-up.
        
        OBJECTIVE:
        BP 132/84
        
        ASSESSMENT:
        Type 2 diabetes
        
        PLAN:
        Continue current medications. Follow-up in 3 months.
        """,
        "reference_note": """
        SUBJECTIVE:
        68-year-old male with type 2 diabetes mellitus, hypertension, and hyperlipidemia 
        presenting for routine follow-up. Last visit 3 months ago. Reports fasting blood 
        sugars 140-180 mg/dL. Medication adherent: metformin 1000mg BID, lisinopril 20mg 
        daily, atorvastatin 40mg QHS. Checks blood sugars twice daily. Following diabetic 
        diet. Exercise: walking 30 minutes 5 days/week. No chest pain, dyspnea, or 
        claudication symptoms.
        
        OBJECTIVE:
        Vital signs: BP 132/84, HR 68, Weight 195 lbs (down 3 lbs from previous)
        Labs:
        - A1C: 7.8% (improved from 8.2%)
        - LDL: 95 mg/dL
        - Creatinine: 1.1 mg/dL (stable)
        Foot examination: No ulcers, protective sensation intact bilaterally
        
        ASSESSMENT:
        1. Type 2 diabetes mellitus - improving control but not at goal
        2. Hypertension - well controlled
        3. Hyperlipidemia - at goal
        4. Overweight - improving with weight loss
        
        PLAN:
        1. Diabetes: Increase metformin to 1500mg BID, goal A1C <7.0%
        2. Continue lisinopril 20mg daily for BP and renal protection
        3. Continue atorvastatin 40mg QHS
        4. Continue current diet and exercise regimen
        5. Annual diabetic eye exam due - referral placed
        6. Repeat A1C, CMP, lipid panel in 3 months
        7. Continue home glucose monitoring
        8. Reviewed signs of hypoglycemia
        9. Follow-up visit in 3 months
        """,
        "metadata": {"source": "synthetic", "scenario": "diabetes_incomplete"}
    }
]


def generate_synthetic_dataset(num_notes: int = None, output_file: str = "data/synthetic_notes.json"):
    """Generate synthetic SOAP notes for testing."""
    if num_notes is None:
        notes = SYNTHETIC_NOTES
    else:
        # Repeat notes if needed
        notes = (SYNTHETIC_NOTES * ((num_notes // len(SYNTHETIC_NOTES)) + 1))[:num_notes]
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(notes, f, indent=2)
    
    print(f"Generated {len(notes)} synthetic SOAP notes")
    print(f"Saved to: {output_file}")
    
    # Print summary
    print("\nScenarios included:")
    scenarios = {}
    for note in notes:
        scenario = note['metadata']['scenario']
        scenarios[scenario] = scenarios.get(scenario, 0) + 1
    
    for scenario, count in scenarios.items():
        print(f"  - {scenario}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic SOAP notes")
    parser.add_argument("--num-notes", type=int, help="Number of notes to generate")
    parser.add_argument("--output", default="data/synthetic_notes.json", 
                       help="Output file path")
    
    args = parser.parse_args()
    generate_synthetic_dataset(args.num_notes, args.output)

