"""
Tuesday, June 9 - Afternoon Session
Actionable Insights Recommendation Engine (Solution Code)

What this script is:
A reference solution for actionable treatment mapping.

Goal of this script:
Map predicted conditions and confidence to agronomic recommendations, returning recapturing flags for low confidence.

Why we are doing it (Student Context):
AI predictions must translate to actions. Farmers need actionable treatment guidelines (e.g., chemical sprays or drainage changes) rather than just a model label, especially when model confidence is low.
"""

def generate_treatment_recommendation(predicted_class, confidence_score):
    """
    Translates model predictions and confidence into action guides.
    """
    # TODO 1 Solution: Low confidence handler
    if confidence_score < 0.60:
        return "Low confidence detection. Please retake the photo under better lighting, ensuring focus is on the leaf."

    # TODO 2 Solution: Crop class recommendation mappings
    if predicted_class == 'Soybeans':
        return "Crop is healthy. No treatment needed. Maintain standard watering and weed control."
        
    elif predicted_class == 'FrogEyeLeafSpot':
        return ("Fungal spots (Frog Eye) detected. Monitor spot density. "
                "If spots cover >10% of the leaf canopy, apply a strobilurin fungicide to prevent yield loss.")
        
    elif predicted_class == 'SuddenDeathSyndrome':
        return ("Sudden Death Syndrome (soil fungus Fusarium virguliforme) detected. "
                "In-season chemical sprays are ineffective. Improve field drainage, and plan to "
                "rotate crops and plant resistant seed varieties for next season.")
        
    elif predicted_class == 'DicambaDamage':
        return ("Chemical herbicide drift (Dicamba) detected. "
                "Document weather/wind records, inspect spray buffer zones, and coordinate with neighboring farms.")
        
    elif predicted_class in ['InsectDamage', 'GenericFeeding']:
        return ("Active insect foliage feeding detected. Deploy sticky traps to monitor pest populations. "
                "Apply organic neem oil or targeted pesticide if feeding thresholds are exceeded.")
        
    else:
        return "Unknown condition. Contact local agricultural extension office for physical tissue testing."

def main():
    # Test cases to verify the recommendation engine
    test_cases = [
        ("Soybeans", 0.95),             # Healthy case
        ("FrogEyeLeafSpot", 0.88),      # High-confidence FLS
        ("SuddenDeathSyndrome", 0.74),   # High-confidence SDS
        ("DicambaDamage", 0.92),        # High-confidence Dicamba
        ("InsectDamage", 0.81),         # High-confidence Insects
        ("GenericFeeding", 0.45),       # Low-confidence case
    ]
    
    print("=========================================")
    print("Soybean Decision Support Engine Testing:")
    print("=========================================")
    for cls, conf in test_cases:
        rec = generate_treatment_recommendation(cls, conf)
        print(f"\nModel Prediction: {cls} ({conf*100:.1f}%)")
        print(f"Recommendation: {rec}")
    print("\n=========================================")

if __name__ == "__main__":
    main()
