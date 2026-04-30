
class EscalationEngine:
    def __init__(self):
        # Thresholds can be tuned based on university policy
        self.level_1_threshold = 0.50
        self.level_2_threshold = 0.75
        self.level_3_threshold = 0.90

    def evaluate_student(self, risk_score, shap_reasons=None, consecutive_high_risk_semesters=1):
        """
        Determines the intervention level and prescriptive action.
        Integrates SHAP reasons to provide a 'Prescriptive Recommendation'.
        """
        
        # 1. Determine Level
        if risk_score >= self.level_3_threshold or (risk_score >= self.level_2_threshold and consecutive_high_risk_semesters >= 3):
            level, default_action = "Level 3 (Academic VP)", "Strategic financial aid review & mandatory program re-evaluation."
        elif risk_score >= self.level_2_threshold or (risk_score >= self.level_1_threshold and consecutive_high_risk_semesters >= 2):
            level, default_action = "Level 2 (Dean/Counselor)", "Psychological counseling & academic foundation plan."
        elif risk_score >= self.level_1_threshold:
            level, default_action = "Level 1 (Advisor)", "Automated email intervention & 1-on-1 meeting request."
        else:
            return "Level 0 (Standard)", "Maintain standard academic monitoring."

        # 2. Add Prescriptive Detail (The Recommendation Engine)
        if shap_reasons:
            top_driver = shap_reasons[0]['metric']
            
            # Map specific drivers to specific "Prescriptions"
            prescriptions = {
                'average_score': "Enrollment in 'Academic Foundation' bridge course.",
                'total_lms_clicks': "Mandatory technical orientation for Virtual Learning Environment (VLE).",
                'avg_submission_day': "Time-management workshop & personalized submission calendar.",
                'imd_band': "Financial aid eligibility review & student support grant consultation.",
                'studied_credits': "Course-load re-balancing; consider reducing credits next semester."
            }
            
            specific_action = prescriptions.get(top_driver, default_action)
            return level, specific_action

        return level, default_action

if __name__ == '__main__':
    # Test the logic
    engine = EscalationEngine()
    print(f"Risk 0.6, 1 sem: {engine.evaluate_student(0.6, 1)}")
    print(f"Risk 0.6, 2 sem: {engine.evaluate_student(0.6, 2)}")
    print(f"Risk 0.95, 1 sem: {engine.evaluate_student(0.95, 1)}")
