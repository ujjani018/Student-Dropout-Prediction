from flask import Flask, render_template, request, send_file
import joblib
import pandas as pd
import os
import io

from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# -----------------------------
# Load trained model safely
# -----------------------------

model_path = os.path.join("model", "student_dropout_model.pkl")

try:
    model = joblib.load(model_path)
    print("✅ Model Loaded Successfully")

except Exception as e:
    model = None
    print("❌ Model Loading Error:", e)


# -----------------------------
# Store latest report
# -----------------------------

latest_report = {}


# -----------------------------
# Home Page
# -----------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        confidence=None,
        advice=None,
        suggestions=None
    )
@app.route("/predict", methods=["POST"])
def predict():

    global latest_report

    try:

        if model is None:
            return render_template(
                "index.html",
                prediction="❌ Model not loaded.",
                status="danger"
            )

        # -----------------------------
        # Get Input
        # -----------------------------

        attendance = float(request.form["attendance"])
        cgpa = float(request.form["cgpa"])
        backlogs = int(request.form["backlogs"])
        study_hours = float(request.form["study_hours"])

        # -----------------------------
        # Validation
        # -----------------------------

        if attendance < 0 or attendance > 100:
            raise ValueError("Attendance should be between 0-100")

        if cgpa < 0 or cgpa > 10:
            raise ValueError("CGPA should be between 0-10")

        if backlogs < 0:
            raise ValueError("Backlogs cannot be negative")

        if study_hours < 0 or study_hours > 24:
            raise ValueError("Study Hours should be between 0-24")

        # -----------------------------
        # Input Data
        # -----------------------------

        input_data = pd.DataFrame(
            [[attendance, cgpa, backlogs, study_hours]],
            columns=[
                "Attendance",
                "CGPA",
                "Backlogs",
                "StudyHours"
            ]
        )

        # -----------------------------
        # Prediction
        # -----------------------------

        prediction = model.predict(input_data)[0]

        confidence = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            if prediction == 1:
                confidence = round(probabilities[1] * 100, 2)
            else:
                confidence = round(probabilities[0] * 100, 2)

        # -----------------------------
        # Result
        # -----------------------------

        if confidence is not None:

            if confidence >= 70:
                result = "⚠️ High Dropout Risk"
                status = "danger"
                advice = "Student needs immediate academic support."

            elif confidence >= 40:
                result = "⚠️ Medium Dropout Risk"
                status = "warning"
                advice = "Student should be monitored regularly."

            else:
                result = "✅ Low Dropout Risk"
                status = "success"
                advice = "Student performance looks stable."

        else:

            if prediction == 1:
                result = "⚠️ High Dropout Risk"
                status = "danger"
                advice = "Student requires academic support."

            else:
                result = "✅ Low Dropout Risk"
                status = "success"
                advice = "Student performance is stable."

        # -----------------------------
        # Suggestions
        # -----------------------------

        suggestions = []

        if attendance < 75:
            suggestions.append("✔ Improve attendance above 75%")

        if cgpa < 6:
            suggestions.append("✔ Improve CGPA by focusing on academics")

        if backlogs > 0:
            suggestions.append("✔ Clear pending backlogs")

        if study_hours < 2:
            suggestions.append("✔ Increase study hours to at least 2-3 hours daily")

        if len(suggestions) == 0:
            suggestions.append("🎉 Excellent performance. Keep it up!")

        # -----------------------------
        # Save Report
        # -----------------------------

        latest_report = {
            "Attendance": attendance,
            "CGPA": cgpa,
            "Backlogs": backlogs,
            "Study Hours": study_hours,
            "Prediction": result,
            "Confidence": confidence,
            "Advice": advice,
            "Date": datetime.now().strftime("%d-%m-%Y %I:%M %p")
        }

        # -----------------------------
        # Return
        # -----------------------------

        return render_template(
            "index.html",
            prediction=result,
            status=status,
            confidence=confidence,
            advice=advice,
            suggestions=suggestions,
            attendance=attendance,
            cgpa=cgpa,
            backlogs=backlogs,
            study_hours=study_hours,
            time=latest_report["Date"]
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction=f"❌ Error: {str(e)}",
            status="danger"
        )
    # -----------------------------
# Download PDF Report
# -----------------------------

@app.route("/download_report")
def download_report():

    global latest_report

    if not latest_report:
        return "No report available. Please generate a prediction first."

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph("<b>Student Dropout Prediction Report</b>", styles["Title"])
    )

    elements.append(
        Paragraph("<br/>", styles["Normal"])
    )

    for key, value in latest_report.items():
        elements.append(
            Paragraph(f"<b>{key}:</b> {value}", styles["Normal"])
        )

    doc.build(elements)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Student_Dropout_Report.pdf",
        mimetype="application/pdf"
    )
# -----------------------------
# Run Application
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)