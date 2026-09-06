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
                margin: 0;
                font-family: Arial, sans-serif;
                background: #f4f7fb;
                color: #172033;
            }

            .header {
                background: #0b1f3a;
                color: white;
                padding: 22px 50px;
            }

            .header h1 {
                margin: 0;
                font-size: 32px;
            }

            .header p {
                margin: 6px 0 0;
                color: #c9d6e8;
            }

            .container {
                max-width: 900px;
                margin: 45px auto;
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
                margin-top: 25px;
                margin-bottom: 8px;
                font-weight: bold;
            }

            input[type="file"],
            input[type="text"] {
                width: 100%;
                box-sizing: border-box;
                padding: 13px;
                border: 1px solid #ccd5e0;
                border-radius: 8px;
                font-size: 15px;
            }

            button {
                margin-top: 28px;
                width: 100%;
                padding: 14px;
                border: none;
                border-radius: 8px;
                background: #1769aa;
                color: white;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
            }

            button:hover {
                background: #12588e;
            }

            .info {
                margin-top: 25px;
                padding: 15px;
                background: #eef5fb;
                border-radius: 8px;
                font-size: 14px;
            }

            .footer {
                text-align: center;
                margin-top: 30px;
                color: #718096;
                font-size: 13px;
            }

        </style>

    </head>


    <body>

        <div class="header">

            <h1>🛰️ SatQuery AI</h1>

            <p>
                Interactive Vision-Language Assistant for Remote Sensing
            </p>

        </div>


        <div class="container">

            <div class="card">

                <h2>Analyze Satellite Imagery</h2>

                <p>
                    Upload a satellite image and ask a question
                    using natural language.
                </p>


                <form
                    action="/analyze"
                    method="post"
                    enctype="multipart/form-data"
                >


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
                        Your Question
                    </label>

                    <input
                        type="text"
                        name="question"
                        placeholder="Example: What can you see in this image?"
                        required
                    >


                    <button type="submit">
                        🔍 Analyze Image
                    </button>

                </form>


                <div class="info">

                    <b>Supported:</b>

                    Optical / multispectral / SAR image formats
                    can be integrated into the full SatQuery AI pipeline.

                </div>

            </div>


            <div class="footer">

                SatQuery AI • SIH 2026 • Remote Sensing Intelligence

            </div>

        </div>

    </body>

    </html>
    """


# ---------------------------------------------------------
# ANALYZE
# ---------------------------------------------------------

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(

    image: UploadFile = File(...),

    image2: UploadFile = File(None),

    sar_image: UploadFile = File(None),

    question: str = Form(...)

):

    os.makedirs("uploads", exist_ok=True)


    # -----------------------------------------------------
    # SAVE MAIN OPTICAL IMAGE
    # -----------------------------------------------------

    safe_filename = os.path.basename(image.filename)

    image_path = os.path.join(
        "uploads",
        safe_filename
    )


    with open(image_path, "wb") as f:

        content = await image.read()

        f.write(content)


    # -----------------------------------------------------
    # SAVE SECOND IMAGE
    # -----------------------------------------------------

    image2_path = None


    if image2 and image2.filename:

        safe_filename2 = os.path.basename(
            image2.filename
        )

        image2_path = os.path.join(
            "uploads",
            safe_filename2
        )


        with open(image2_path, "wb") as f:

            content2 = await image2.read()

            f.write(content2)


    # -----------------------------------------------------
    # SAVE SAR IMAGE
    # -----------------------------------------------------

    sar_image_path = None


    if sar_image and sar_image.filename:

        safe_sar_filename = os.path.basename(
            sar_image.filename
        )

        sar_image_path = os.path.join(
            "uploads",
            safe_sar_filename
        )


        with open(sar_image_path, "wb") as f:

            sar_content = await sar_image.read()

            f.write(sar_content)


    # -----------------------------------------------------
    # AGENT TASK DETECTION
    # -----------------------------------------------------

    task = detect_task(question)

    print("Detected task:", task)


    # -----------------------------------------------------
    # VQA
    # -----------------------------------------------------

    if task == "vqa":

        print("Running BLIP VQA...")


        answer = answer_question(

            image_path,

            question

        )


    # -----------------------------------------------------
    # GROUNDING
    # -----------------------------------------------------

    elif task == "grounding":

        print("Running grounding evidence...")


        evidence_path = (
            "uploads/grounding_evidence.jpg"
        )


        create_grounding_evidence(

            image_path,

            evidence_path,

            question

        )


        answer = (
            "The relevant region has been "
            "highlighted in the image below."
        )


    # -----------------------------------------------------
    # CHANGE ANALYSIS
    # -----------------------------------------------------

    elif task == "change_analysis":

        print("Running change analysis...")


        if image2_path:

            change_map_path = (
                "uploads/change_map.jpg"
            )


            (
                change_map_path,
                change_percentage
            ) = create_change_map(

                image_path,

                image2_path,

                change_map_path

            )


            answer = (

                f"Visual change detected: "
                f"{change_percentage}% between "
                f"the two satellite images."

            )


        else:

            answer = (
                "Please upload a second satellite "
                "image for change analysis."
            )


    # -----------------------------------------------------
    # OPTICAL + SAR
    # -----------------------------------------------------

    elif task == "optical_sar":

        print(
            "Running Optical + SAR analysis..."
        )


        if sar_image_path:

            answer = analyze_optical_sar(

                image_path,

                sar_image_path

            )


        else:

            answer = (
                "Please upload a SAR image for "
                "Optical + SAR analysis."
            )


    # -----------------------------------------------------
    # OTHER TASK
    # -----------------------------------------------------

    else:

        answer = (

            f"The SatQuery Agent detected "
            f"this as '{task}'."

        )


    print("AI Answer:", answer)


    # -----------------------------------------------------
    # SAFE HTML VALUES
    # -----------------------------------------------------

    safe_question = html.escape(
        question
    )


    safe_answer = html.escape(
        str(answer)
    )


    safe_filename = html.escape(
        safe_filename
    )


    safe_sar_filename = ""


    if sar_image_path:

        safe_sar_filename = html.escape(

            os.path.basename(
                sar_image_path
            )

        )


    # -----------------------------------------------------
    # EVIDENCE HTML
    # -----------------------------------------------------

    if task == "grounding":

        evidence_html = """

        <h2>
            🔎 Grounding Evidence
        </h2>

        <img
            src="/uploads/grounding_evidence.jpg"
            alt="Grounding evidence"
            style="
                max-width:100%;
                max-height:500px;
                border-radius:12px;
                margin:20px 0;
            "
        >

        """


    elif (
        task == "change_analysis"
        and image2_path
    ):

        evidence_html = """

        <h2>
            🔄 Change Analysis Evidence
        </h2>

        <img
            src="/uploads/change_map.jpg"
            alt="Change analysis map"
            style="
                max-width:100%;
                max-height:500px;
                border-radius:12px;
                margin:20px 0;
            "
        >

        """


    else:

        evidence_html = ""


    # -----------------------------------------------------
    # SAR IMAGE HTML
    # -----------------------------------------------------

    if safe_sar_filename:

        sar_html = f"""

        <h3>
            📡 SAR Image
        </h3>

        <img
            src="/uploads/{safe_sar_filename}"
            alt="SAR image"
            style="
                max-width:100%;
                max-height:500px;
                border-radius:12px;
                margin:20px 0;
            "
        >

        """


    else:

        sar_html = ""


    # -----------------------------------------------------
    # RESULT PAGE
    # -----------------------------------------------------

    return f"""

    <!DOCTYPE html>

    <html>

    <head>

        <title>
            SatQuery AI Result
        </title>


        <style>

            body {{
                margin: 0;
                font-family: Arial, sans-serif;
                background: #f4f7fb;
                color: #172033;
            }}


            .header {{
                background: #0b1f3a;
                color: white;
                padding: 22px 50px;
            }}


            .header h1 {{
                margin: 0;
                font-size: 32px;
            }}


            .container {{
                max-width: 900px;
                margin: 45px auto;
                padding: 0 25px;
            }}


            .card {{
                background: white;
                padding: 35px;
                border-radius: 16px;
                box-shadow:
                    0 8px 25px
                    rgba(0,0,0,0.08);
            }}


            .question {{
                background: #f4f7fb;
                padding: 18px;
                border-radius: 8px;
                margin: 15px 0;
            }}


            .answer {{
                background: #eef7ee;
                border-left:
                    5px solid #3c8d40;
                padding: 20px;
                border-radius: 8px;
                font-size: 18px;
                white-space: pre-line;
            }}


            img {{
                display: block;
            }}


            a {{
                display: inline-block;
                margin-top: 25px;
                color: #1769aa;
                text-decoration: none;
                font-weight: bold;
            }}

        </style>

    </head>


    <body>


        <div class="header">

            <h1>
                🛰️ SatQuery AI
            </h1>

        </div>


        <div class="container">


            <div class="card">


                <h2>
                    Analysis Result
                </h2>


                <p>

                    <b>
                        Optical / Main Satellite Image:
                    </b>

                    {safe_filename}

                </p>


                <img
                    src="/uploads/{safe_filename}"
                    alt="Uploaded optical satellite image"
                    style="
                        max-width:100%;
                        max-height:500px;
                        border-radius:12px;
                        margin:20px 0;
                    "
                >


                {sar_html}


                {evidence_html}


                <h3>
                    Your Question
                </h3>


                <div class="question">

                    {safe_question}

                </div>


                <h3>
                    🤖 AI Answer
                </h3>


                <div class="answer">

                    {safe_answer}

                </div>


                <a href="/">

                    ← Analyze another image

                </a>


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

        "status": "running",

        "project": "SatQuery AI"

    }