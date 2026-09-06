from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from blip_vqa import answer_question
from agent import detect_task
from grounding import create_grounding_evidence
from change_analysis import create_change_map
from optical_sar import analyze_optical_sar

import os
import html


app = FastAPI(title="SatQuery AI")


# ---------------------------------------------------------
# UPLOAD FOLDER
# ---------------------------------------------------------

os.makedirs("uploads", exist_ok=True)

app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads"
)


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <!DOCTYPE html>
    <html>
    <head>

        <title>SatQuery AI</title>

        <style>

            body {
                font-family: Arial, sans-serif;
                background: #f4f7fb;
                margin: 0;
                padding: 0;
            }

            .header {
                background: #0b1f3a;
                color: white;
                padding: 30px;
                text-align: center;
            }

            .header h1 {
                margin: 0;
                font-size: 32px;
            }

            .header p {
                margin-top: 8px;
                color: #c9d6e8;
            }

            .container {
                max-width: 900px;
                margin: 40px auto;
                padding: 0 25px;
            }

            .card {
                background: white;
                padding: 35px;
                border-radius: 16px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            }

            h2 {
                margin-top: 0;
                color: #0b1f3a;
            }

            label {
                display: block;
                margin-top: 20px;
                margin-bottom: 8px;
                font-weight: bold;
            }

            input[type="file"],
            textarea {
                width: 100%;
                box-sizing: border-box;
                padding: 12px;
                border: 1px solid #ccd5e0;
                border-radius: 8px;
                font-size: 15px;
            }

            textarea {
                min-height: 100px;
                resize: vertical;
            }

            button {
                margin-top: 25px;
                width: 100%;
                padding: 14px;
                background: #0b1f3a;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 17px;
                cursor: pointer;
            }

            button:hover {
                background: #173b6d;
            }

            .info {
                background: #eef5ff;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }

        </style>

    </head>

    <body>

        <div class="header">

            <h1>SatQuery AI</h1>

            <p>
                Interactive Vision-Language Assistant for
                Multimodal Remote Sensing Image Analysis
            </p>

        </div>


        <div class="container">

            <div class="card">

                <h2>Satellite Image Analysis</h2>

                <div class="info">

                    Ask a question about your satellite image
                    in natural language.

                    <br><br>

                    Examples:

                    <br>
                    • What can you see in this image?
                    <br>
                    • Where is the water body?
                    <br>
                    • What changed between these two satellite images?
                    <br>
                    • Use optical and SAR together to analyze this scene.

                </div>


                <form action="/analyze" method="post" enctype="multipart/form-data">

                    <label>
                        Satellite Image
                    </label>

                    <input
                        type="file"
                        name="image"
                        accept=".png,.jpg,.jpeg,.tif,.tiff"
                        required
                    >


                    <label>
                        Second Satellite Image
                        (for Change Analysis)
                    </label>

                    <input
                        type="file"
                        name="image2"
                        accept=".png,.jpg,.jpeg,.tif,.tiff"
                    >


                    <label>
                        SAR Image
                        (for Optical + SAR Analysis)
                    </label>

                    <input
                        type="file"
                        name="sar_image"
                        accept=".png,.jpg,.jpeg,.tif,.tiff"
                    >


                    <label>
                        Ask your question
                    </label>

                    <textarea
                        name="question"
                        placeholder="Example: What can you see in this satellite image?"
                        required
                    ></textarea>


                    <button type="submit">
                        Analyze Satellite Image
                    </button>

                </form>

            </div>

        </div>

    </body>
    </html>
    """


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "application": "SatQuery AI"
    }


# ---------------------------------------------------------
# ANALYSIS ENDPOINT
# ---------------------------------------------------------

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(

    image: UploadFile = File(...),

    image2: UploadFile = File(None),

    sar_image: UploadFile = File(None),

    question: str = Form(...)

):

    # -----------------------------------------------------
    # SAVE MAIN IMAGE
    # -----------------------------------------------------

    image_path = "uploads/main_image_" + image.filename

    image_content = await image.read()

    with open(image_path, "wb") as f:
        f.write(image_content)


    # -----------------------------------------------------
    # SAVE SECOND IMAGE
    # -----------------------------------------------------

    image2_path = None

    if image2 and image2.filename:

        image2_path = "uploads/second_image_" + image2.filename

        image2_content = await image2.read()

        with open(image2_path, "wb") as f:
            f.write(image2_content)


    # -----------------------------------------------------
    # SAVE SAR IMAGE
    # -----------------------------------------------------

    sar_image_path = None

    if sar_image and sar_image.filename:

        sar_image_path = "uploads/sar_" + sar_image.filename

        sar_content = await sar_image.read()

        with open(sar_image_path, "wb") as f:
            f.write(sar_content)


    # -----------------------------------------------------
    # AGENT TASK DETECTION
    # -----------------------------------------------------

    task = detect_task(question)

    print("Detected task:", task)


    # -----------------------------------------------------
    # RESULT VARIABLES
    # -----------------------------------------------------

    answer = None

    evidence_path = None

    change_percentage = None

    optical_sar_result = None


    # -----------------------------------------------------
    # VQA
    # -----------------------------------------------------

    if task == "vqa":

        print("Running BLIP VQA...")

        answer = answer_question(
            image_path,
            question
        )

        print("VQA Answer:", answer)


    # -----------------------------------------------------
    # GROUNDING
    # -----------------------------------------------------

    elif task == "grounding":

        print("Running grounding evidence...")

        evidence_path = "uploads/grounding_evidence.jpg"

        create_grounding_evidence(
            image_path,
            evidence_path,
            question
        )

        answer = (
            "A visual grounding region has been generated "
            "for the requested object or area."
        )


    # -----------------------------------------------------
    # CHANGE ANALYSIS
    # -----------------------------------------------------

    elif task == "change_analysis":

        print("Running change analysis...")

        if image2_path:

            evidence_path = "uploads/change_map.jpg"

            evidence_path, change_percentage = create_change_map(
                image_path,
                image2_path,
                evidence_path
            )

            answer = (
                f"Visual change detected: "
                f"{change_percentage}% of pixels "
                f"differed between the two images."
            )

        else:

            answer = (
                "Please upload a second satellite image "
                "to perform change analysis."
            )


    # -----------------------------------------------------
    # OPTICAL + SAR
    # -----------------------------------------------------

    elif task == "optical_sar":

        print("Running Optical + SAR analysis...")

        if sar_image_path:

            optical_sar_result = analyze_optical_sar(
                image_path,
                sar_image_path
            )

            answer = optical_sar_result

        else:

            answer = (
                "Please upload a SAR image together with "
                "the optical satellite image."
            )


    # -----------------------------------------------------
    # IF SAR IS PROVIDED WITH ANOTHER TASK
    # -----------------------------------------------------

    if sar_image_path and task != "optical_sar":

        print("SAR image detected.")

        if task == "change_analysis":

            answer = (
                answer
                + "<br><br>"
                + "<b>Note:</b> A SAR image was also uploaded. "
                "For a complete multimodal Optical + SAR analysis, "
                "use a question such as: "
                "<i>Use optical and SAR together to analyze this scene.</i>"
            )


    # -----------------------------------------------------
    # ESCAPE TEXT SAFELY
    # -----------------------------------------------------

    safe_question = html.escape(question)

    if answer is None:

        answer = "Analysis completed."

    safe_answer = str(answer)


    # -----------------------------------------------------
    # EVIDENCE IMAGE HTML
    # -----------------------------------------------------

    evidence_html = ""

    if evidence_path:

        evidence_filename = os.path.basename(evidence_path)

        evidence_html = f"""
        <div class="evidence">

            <h3>Visual Evidence</h3>

            <img
                src="/uploads/{html.escape(evidence_filename)}"
                alt="Analysis Evidence"
            >

        </div>
        """


    # -----------------------------------------------------
    # EXECUTION SUMMARY
    # -----------------------------------------------------

    execution_summary = f"""
    <div class="summary">

        <h3>Execution Summary</h3>

        <p>
            <b>User Query:</b>
            {safe_question}
        </p>

        <p>
            <b>Detected Task:</b>
            {html.escape(str(task))}
        </p>

        <p>
            <b>Primary Image:</b>
            {html.escape(image.filename)}
        </p>

    """

    if image2_path:

        execution_summary += f"""
        <p>
            <b>Second Image:</b>
            {html.escape(image2.filename)}
        </p>
        """

    if sar_image_path:

        execution_summary += f"""
        <p>
            <b>SAR Image:</b>
            {html.escape(sar_image.filename)}
        </p>
        """

    execution_summary += """
    </div>
    """


    # -----------------------------------------------------
    # RESULT PAGE
    # -----------------------------------------------------

    return f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>SatQuery AI - Analysis Result</title>

        <style>

            body {{
                font-family: Arial, sans-serif;
                background: #f4f7fb;
                margin: 0;
                padding: 0;
            }}

            .header {{
                background: #0b1f3a;
                color: white;
                padding: 25px;
                text-align: center;
            }}

            .container {{
                max-width: 900px;
                margin: 35px auto;
                padding: 0 25px;
            }}

            .card {{
                background: white;
                padding: 30px;
                border-radius: 16px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.08);
                margin-bottom: 25px;
            }}

            h2 {{
                color: #0b1f3a;
                margin-top: 0;
            }}

            h3 {{
                color: #173b6d;
            }}

            .answer {{
                background: #eef7ee;
                padding: 20px;
                border-radius: 10px;
                font-size: 18px;
                line-height: 1.6;
            }}

            .summary {{
                background: #f5f7fa;
                padding: 20px;
                border-radius: 10px;
                line-height: 1.6;
            }}

            .evidence {{
                margin-top: 25px;
            }}

            .evidence img {{
                max-width: 100%;
                border-radius: 10px;
                border: 1px solid #ddd;
            }}

            .back {{
                display: inline-block;
                margin-top: 20px;
                padding: 12px 20px;
                background: #0b1f3a;
                color: white;
                text-decoration: none;
                border-radius: 8px;
            }}

        </style>

    </head>


    <body>

        <div class="header">

            <h1>SatQuery AI</h1>

            <p>Satellite Analysis Result</p>

        </div>


        <div class="container">


            <div class="card">

                <h2>AI Answer</h2>

                <div class="answer">

                    {safe_answer}

                </div>

            </div>


            <div class="card">

                {execution_summary}

                {evidence_html}

            </div>


            <div class="card">

                <a class="back" href="/">
                    ← Analyze Another Image
                </a>

            </div>


        </div>

    </body>

    </html>
    """