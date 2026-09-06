from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>SatQuery AI</title>

        <meta name="viewport" content="width=device-width, initial-scale=1">

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
                padding: 25px;
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
                padding: 0 20px;
            }

            .card {
                background: white;
                padding: 35px;
                border-radius: 16px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            }

            h2 {
                color: #0b1f3a;
            }

            label {
                display: block;
                margin-top: 22px;
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
                    Upload satellite image(s) and ask a question
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


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "application": "SatQuery AI"
    }


# ============================================================
# ANALYZE
# ============================================================

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(

    image: UploadFile = File(...),

    image2: UploadFile = File(None),

    sar_image: UploadFile = File(None),

    question: str = Form(...)

):

    os.makedirs("uploads", exist_ok=True)


    # --------------------------------------------------------
    # SAVE PRIMARY IMAGE
    # --------------------------------------------------------

    safe_filename = os.path.basename(image.filename)

    image_path = os.path.join(
        "uploads",
        safe_filename
    )

    content = await image.read()

    with open(image_path, "wb") as f:
        f.write(content)


    # --------------------------------------------------------
    # SAVE SECOND IMAGE
    # --------------------------------------------------------

    image2_path = None

    if image2 and image2.filename:

        safe_filename2 = os.path.basename(
            image2.filename
        )

        image2_path = os.path.join(
            "uploads",
            safe_filename2
        )

        content2 = await image2.read()

        with open(image2_path, "wb") as f:
            f.write(content2)


    # --------------------------------------------------------
    # SAVE SAR IMAGE
    # --------------------------------------------------------

    sar_image_path = None

    if sar_image and sar_image.filename:

        safe_sar_filename = os.path.basename(
            sar_image.filename
        )

        sar_image_path = os.path.join(
            "uploads",
            safe_sar_filename
        )

        sar_content = await sar_image.read()

        with open(sar_image_path, "wb") as f:
            f.write(sar_content)


    # --------------------------------------------------------
    # DETECT TASK
    # --------------------------------------------------------

    task = detect_task(question)

    print("Detected task:", task)


    # --------------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------------

    answer = ""

    evidence_path = None


    # ========================================================
    # VQA
    # ========================================================

    if task == "vqa":

        print("Running lightweight VQA demo...")

        answer = (
            "This satellite image contains visible land features. "
            "For full remote-sensing VQA, an RS-specific "
            "vision-language model can be connected in the "
            "cloud version."
        )


    # ========================================================
    # GROUNDING
    # ========================================================

    elif task == "grounding":

        print("Running grounding evidence...")

        evidence_path = os.path.join(
            "uploads",
            "grounding_evidence.jpg"
        )

        create_grounding_evidence(

            image_path,

            evidence_path,

            question

        )

        answer = (
            "A visual grounding region has been generated "
            "for the requested object or area."
        )


    # ========================================================
    # CHANGE ANALYSIS
    # ========================================================

    elif task == "change_analysis":

        print("Running change analysis...")

        if image2_path:

            evidence_path = os.path.join(
                "uploads",
                "change_map.jpg"
            )

            evidence_path, change_percentage = create_change_map(

                image_path,

                image2_path,

                evidence_path

            )

            answer = (
                f"Visual change detected: "
                f"{change_percentage}% between "
                f"the two satellite images."
            )

        else:

            answer = (
                "Please upload a second satellite image "
                "for change analysis."
            )


    # ========================================================
    # OPTICAL + SAR
    # ========================================================

    elif task == "optical_sar":

        print("Running Optical + SAR analysis...")

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


    # ========================================================
    # UNKNOWN TASK
    # ========================================================

    else:

        answer = (
            f"The SatQuery Agent detected this as '{task}'."
        )


    # --------------------------------------------------------
    # ESCAPE USER CONTENT
    # --------------------------------------------------------

    safe_question = html.escape(question)

    safe_answer = html.escape(
        str(answer)
    )

    safe_filename = html.escape(
        os.path.basename(image_path)
    )


    # --------------------------------------------------------
    # EXECUTION SUMMARY
    # --------------------------------------------------------

    execution_summary = f"""

        <div class="summary">

            <h2>Execution Summary</h2>

            <p>
                <b>User Query:</b>
                {safe_question}
            </p>

            <p>
                <b>Detected Task:</b>
                {html.escape(task)}
            </p>

            <p>
                <b>Primary Image:</b>
                {safe_filename}
            </p>

    """


    if image2_path:

        execution_summary += f"""

            <p>
                <b>Second Image:</b>
                {html.escape(
                    os.path.basename(image2_path)
                )}
            </p>

        """


    if sar_image_path:

        execution_summary += f"""

            <p>
                <b>SAR Image:</b>
                {html.escape(
                    os.path.basename(sar_image_path)
                )}
            </p>

        """


    execution_summary += """

        </div>

    """


    # --------------------------------------------------------
    # VISUAL EVIDENCE
    # --------------------------------------------------------

    evidence_html = ""


    if evidence_path:

        evidence_filename = html.escape(
            os.path.basename(evidence_path)
        )


        if task == "grounding":

            evidence_title = "🔎 Grounding Evidence"

        elif task == "change_analysis":

            evidence_title = "🔄 Change Analysis Evidence"

        else:

            evidence_title = "Visual Evidence"


        evidence_html = f"""

        <div class="evidence">

            <h2>{evidence_title}</h2>

            <img

                src="/uploads/{evidence_filename}"

                alt="Analysis Evidence"

            >

        </div>

        """


    # ========================================================
    # RESULT PAGE
    # ========================================================

    return f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>
            SatQuery AI - Analysis Result
        </title>

        <meta
            name="viewport"
            content="width=device-width, initial-scale=1"
        >


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

                padding: 22px;

                text-align: center;

            }}


            .header h1 {{

                margin: 0;

                font-size: 30px;

            }}


            .header p {{

                margin-top: 6px;

                color: #c9d6e8;

            }}


            .container {{

                max-width: 1000px;

                margin: 35px auto;

                padding: 0 20px;

            }}


            .card {{

                background: white;

                padding: 30px;

                border-radius: 16px;

                box-shadow:
                    0 8px 25px rgba(0,0,0,0.08);

            }}


            .answer {{

                background: #eef7ee;

                border-left: 5px solid #3c8d40;

                padding: 20px;

                border-radius: 8px;

                font-size: 18px;

                line-height: 1.6;

            }}


            .summary {{

                background: #f4f7fb;

                padding: 20px;

                border-radius: 10px;

                margin-top: 25px;

            }}


            .summary h2 {{

                color: #0b1f3a;

                margin-top: 0;

            }}


            .image-section {{

                margin-top: 25px;

            }}


            .image-section img {{

                display: block;

                width: 100%;

                max-width: 800px;

                max-height: 600px;

                object-fit: contain;

                margin-top: 15px;

                border-radius: 12px;

                border: 1px solid #d8dee8;

            }}


            .evidence {{

                margin-top: 30px;

            }}


            .evidence h2 {{

                color: #0b1f3a;

            }}


            .evidence img {{

                display: block;

                width: 100%;

                max-width: 800px;

                max-height: 600px;

                object-fit: contain;

                margin-top: 15px;

                border-radius: 12px;

                border: 1px solid #d8dee8;

            }}


            .answer-section {{

                margin-top: 30px;

            }}


            .back-button {{

                display: inline-block;

                margin-top: 30px;

                padding: 12px 18px;

                background: #0b1f3a;

                color: white;

                text-decoration: none;

                border-radius: 8px;

            }}


            .back-button:hover {{

                background: #1769aa;

            }}

        </style>

    </head>


    <body>


        <div class="header">

            <h1>🛰️ SatQuery AI</h1>

            <p>
                Satellite Analysis Result
            </p>

        </div>


        <div class="container">


            <div class="card">


                <!-- UPLOADED IMAGE -->

                <div class="image-section">

                    <h2>
                        Uploaded Satellite Image
                    </h2>

                    <img

                        src="/uploads/{safe_filename}"

                        alt="Uploaded Satellite Image"

                    >

                </div>


                <!-- ANSWER -->

                <div class="answer-section">

                    <h2>
                        AI Answer
                    </h2>

                    <div class="answer">

                        {safe_answer}

                    </div>

                </div>


                <!-- EXECUTION SUMMARY -->

                {execution_summary}


                <!-- VISUAL EVIDENCE -->

                {evidence_html}


                <a
                    href="/"
                    class="back-button"
                >
                    ← Analyze Another Image
                </a>


            </div>

        </div>


    </body>

    </html>
    """