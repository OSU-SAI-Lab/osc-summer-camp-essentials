"""
Tuesday, June 9 - Afternoon Session
Actionable Insights Recommendation Engine (Starter Code)

What this script is:
A decision support recommendation engine script.

Goal of this script:
Map model disease predictions and confidence scores to agronomic recommendations, and handle low-confidence detections.

Why we are doing it (Student Context):
AI predictions must translate to actions. Farmers need actionable treatment guidelines (e.g., chemical sprays or drainage changes) rather than just a model label, especially when model confidence is low.
"""

def generate_treatment_recommendation(predicted_class, confidence_score):
    """
    Translates model predictions and confidence into action guides.
    
    Args:
        predicted_class (str): Class name predicted by the model
        confidence_score (float): Confidence probability value between 0 and 1
        
    Returns:
        recommendation (str): Agronomic recommendation message
    """
    # TODO 1: Implement low confidence handling.
    # If the confidence score is below 0.60, return a recommendation asking the
    # student or farmer to capture a clearer, more focused image.
    
    if confidence_score < 0.60:
        return "Low confidence detection. Please retake the photo under better lighting, ensuring focus is on the leaf."

    # TODO 2: Map each crop class to the appropriate agronomic recommendation
    # Classes:
    #   - 'Soybeans' (Healthy)
    #   - 'FrogEyeLeafSpot' (FLS)
    #   - 'SuddenDeathSyndrome' (SDS)
    #   - 'DicambaDamage' (Herbicide drift)
    #   - 'InsectDamage' or 'GenericFeeding' (Pest feeding)
    
    if predicted_class == 'Soybeans':
        # TODO: Return healthy crop recommendation
        
        pass
        
    elif predicted_class == 'FrogEyeLeafSpot':
        # TODO: Return FLS treatment recommendation
        
        pass
        
    elif predicted_class == 'SuddenDeathSyndrome':
        # TODO: Return SDS management recommendation (Fusarium virguliforme)
        
        pass
        
    elif predicted_class == 'DicambaDamage':
        # TODO: Return herbicide drift response recommendation
        
        pass
        
    elif predicted_class in ['InsectDamage', 'GenericFeeding']:
        # TODO: Return insect feeding treatment recommendation
        
        pass
        
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
