
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_pdf():
    file_path = "docs/Project_Research_Summary_Detailed.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = styles['Title']
    story.append(Paragraph("Research Summary: Proactive Educational Governance", title_style))
    story.append(Spacer(1, 12))

    # 1. Purpose and Objective
    story.append(Paragraph("1. Purpose and Objective of the Research", styles['Heading2']))
    story.append(Paragraph(
        "The objective is to move from reactive 'post-mortem' student analysis to a proactive 'governance' framework. "
        "The system aims to predict dropout risk early enough for meaningful intervention, explain the risk drivers "
        "to human advisors, and automate a multi-tiered response protocol based on the severity and persistence of risk.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 12))

    # 2. Data Collected for Training
    story.append(Paragraph("2. Data Collection & Preprocessing", styles['Heading2']))
    story.append(Paragraph("<b>Source of Data:</b> Open University Learning Analytics Dataset (OULAD).", styles['BodyText']))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("<b>Data Structure & Relationships:</b>", styles['BodyText']))
    data = [
        ['Table', 'Fields', 'Description'],
        ['studentInfo', 'Gender, Region, IMD, Education, Age', 'Primary Demographic Metadata'],
        ['studentVle', 'sum_clicks, date', 'Engagement logs (LMS interaction)'],
        ['studentAssessment', 'score, date_submitted', 'Academic Performance metrics']
    ]
    t = Table(data, colWidths=[100, 250, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Cleaning Process:</b>", styles['BodyText']))
    story.append(Paragraph(
        "• Aggregation: Millions of raw VLE clickstreams were aggregated into student-level engagement features.<br/>"
        "• Missing Values: Handled socio-economic gaps (IMD Band) using 'Missing' category encoding to maintain bias awareness.<br/>"
        "• Latency Engineering: Calculated average submission dates as a proxy for student procrastination behavior.<br/>"
        "• Filtering: Removed non-predictive administrative IDs to ensure the model focuses on behavioral patterns.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 12))

    # 3. AI Models & Deployment
    story.append(Paragraph("3. AI Models: Training & Justification", styles['Heading2']))
    story.append(Paragraph("<b>Models Used & Deployed:</b> XGBoost (Predictor), SHAP (Explainer), CTGAN (Synthetic Generator).", styles['BodyText']))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Justification:</b>", styles['BodyText']))
    story.append(Paragraph(
        "• XGBoost: The gold standard for tabular data, providing superior handling of non-linear student behavior patterns.<br/>"
        "• SHAP: Essential for 'Human-in-the-Loop' trust; it translates mathematical weights into readable 'Intervention Cards'.<br/>"
        "• CTGAN: Allows for the creation of 'Digital Twins' to protect real student privacy while enabling robust system testing.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Training Process:</b>", styles['BodyText']))
    story.append(Paragraph(
        "• Split: 80% Training / 20% Testing.<br/>"
        "• Optimization: GridSearchCV with 3-fold cross-validation used to tune 54 hyperparameter combinations.<br/>"
        "• Fairness: Audited with AIF360 to ensure zero disparate impact across demographics.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 12))

    # 4. Results
    story.append(Paragraph("4. Model Results", styles['Heading2']))
    results = [['Metric', 'Score'], ['Accuracy', '89.65%'], ['F1-Score', '0.8973'], ['ROC-AUC', '0.9609']]
    t2 = Table(results, colWidths=[150, 150])
    t2.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.darkblue), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('GRID', (0, 0), (-1, -1), 0.5, colors.black)]))
    story.append(t2)
    story.append(Spacer(1, 12))

    # 5. Technology Stack
    story.append(Paragraph("5. Technology Stack", styles['Heading2']))
    story.append(Paragraph("Python, Flask, Streamlit, XGBoost, SHAP, CTGAN, AIF360, Pandas, Scikit-Learn.", styles['BodyText']))
    story.append(Spacer(1, 12))

    # 6. Limitations & Future Work
    story.append(Paragraph("6. Limitations & Future Work", styles['Heading2']))
    story.append(Paragraph(
        "• Limitations: Current escalation thresholds are static; requires real-world institutional piloting for threshold tuning.<br/>"
        "• Future Work: Direct LMS (Canvas/Moodle) integration; expansion of the prescriptive Knowledge Graph for automated support.",
        styles['BodyText']
    ))

    doc.build(story)
    print(f"PDF generated successfully at {file_path}")

if __name__ == '__main__':
    generate_pdf()
