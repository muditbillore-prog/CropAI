import os
import tempfile
from datetime import date, datetime, timedelta

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -------------------- CONFIG --------------------
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "best.pt"
)

st.set_page_config(
    page_title="CropAI | Intelligent Farm Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_rice_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Rice model not found: {MODEL_PATH}")
    return YOLO(MODEL_PATH)


def calculate_maturity(planting_date, crop_duration):
    """Calculate a planning estimate using planting date and crop duration."""
    duration_days = int(crop_duration)
    maturity_date = planting_date + timedelta(days=duration_days)
    today = date.today()
    elapsed = (today - planting_date).days
    remaining = (maturity_date - today).days
    progress = min(1.0, max(0.0, elapsed / duration_days))

    if progress < 0.15:
        stage = "Early establishment"
    elif progress < 0.60:
        stage = "Vegetative growth"
    elif progress < 0.80:
        stage = "Reproductive growth"
    elif progress < 1.0:
        stage = "Ripening / grain filling"
    else:
        stage = "Expected maturity window reached"

    return {
        "duration_days": duration_days,
        "maturity_date": maturity_date,
        "remaining": remaining,
        "progress": progress,
        "stage": stage,
    }


# -------------------- STYLING --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#0b100e;--panel:#131b17;--border:#29372e;--green:#b5f36a;--muted:#96a79a;--white:#f1f6f0}
.stApp{background:var(--bg);color:var(--white);font-family:'DM Sans',sans-serif}
h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.6px}
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1440px}
[data-testid="stSidebar"]{background:#101713;border-right:1px solid var(--border)}
.brand{font:800 25px 'Manrope',sans-serif;color:var(--white);letter-spacing:-1px}
.brand span{color:var(--green)}
.eyebrow{color:var(--green);font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}
.hero{background:linear-gradient(120deg,#19271e,#111a15 65%,#1b2a1c);border:1px solid #344536;border-radius:20px;padding:30px;margin:12px 0 25px}
.hero h1{font-size:clamp(29px,3.4vw,43px);line-height:1.15;margin:12px 0}
.hero p{color:#b4c4b7;max-width:660px;line-height:1.7}
.section-title{font:700 19px 'Manrope',sans-serif;margin:25px 0 5px}
.section-subtitle{color:var(--muted);font-size:13px;margin-bottom:16px}
.metric-card,.panel{background:var(--panel);border:1px solid var(--border);border-radius:15px;padding:19px}
.metric-card{min-height:112px}
.metric-label{font-size:12px;color:var(--muted);margin-bottom:10px}
.metric-value{font:700 22px 'Manrope',sans-serif}
.metric-note{color:var(--green);font-size:11px;margin-top:7px}
.panel{border-radius:17px;padding:22px;margin-bottom:14px}
.panel-title{font:700 16px 'Manrope',sans-serif;margin-bottom:5px}
.panel-desc{color:var(--muted);font-size:12px;line-height:1.6;margin-bottom:12px}
.pill{display:inline-block;background:#233722;border:1px solid #40583b;color:var(--green);border-radius:30px;padding:5px 10px;font-size:10px;font-weight:700;letter-spacing:.7px}
div[data-testid="stFileUploader"]{background:#111a15;border:1px dashed #536b53;border-radius:13px;padding:12px}
.stButton>button{background:var(--green);color:#12200e;border:0;border-radius:10px;min-height:45px;font-weight:700;width:100%}
.stButton>button:hover{background:#d0ff98;color:#12200e;border:0}
div[data-baseweb="select"]>div,div[data-testid="stNumberInput"] input{background:#111a15;border-color:var(--border);border-radius:9px}
label,[data-testid="stWidgetLabel"] p{color:#d5e0d6!important;font-size:13px!important}
hr{border-color:var(--border)} footer,#MainMenu{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# -------------------- NAVIGATION --------------------
# Apply a pending page selection BEFORE creating the radio widget
if "pending_page_choice" in st.session_state:
    st.session_state["page_choice"] = st.session_state.pop(
        "pending_page_choice"
    )

with st.sidebar:
    st.markdown(
        '<div class="brand">🌿 Crop<span>AI</span></div>',
        unsafe_allow_html=True
    )
    st.caption("INTELLIGENT FARM ANALYTICS")
    st.divider()

    pages = [
        "Farm Overview",
        "Video Analysis",
        "Rice Variety AI",
        "About CropAI"
    ]

    page = st.radio("WORKSPACE", pages, key="page_choice")

    st.divider()
    st.markdown("**PROJECT STATUS**")
    st.markdown(
        '<span class="pill">PROTOTYPE · AI CLASSIFICATION</span>',
        unsafe_allow_html=True
    )
    st.caption(
        "Rice variety classification and preliminary maturity estimation."
    )
    st.divider()
    st.caption("CROPAI • AGRICULTURAL INTELLIGENCE")


# -------------------- FARM OVERVIEW --------------------

if page == "Farm Overview":
    st.markdown(
        '<div class="eyebrow">AGRICULTURAL INTELLIGENCE PLATFORM</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero">
        <div class="pill">SMART FARMING · AI WORKSPACE</div>
        <h1>Understand your farm.<br>Make better decisions.</h1>
        <p>Analyse agricultural videos, explore rice varieties and estimate crop
        maturity using farm information — all in one workspace.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Workspace overview</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">Your analysis environment at a glance.</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(3)

    cards = [
        ("01 / INPUT", "Video analysis", "Drone & ground-level footage"),
        ("02 / AI MODEL", "Rice classification", "Top-5 variety predictions"),
        ("03 / FARM PLANNING", "Maturity estimate", "Timeline from planting date"),
    ]

    for col, (label, value, note) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-note">{note}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">Start an analysis</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="panel">'
        '<div class="panel-title">Rice variety and maturity workspace</div>'
        '<div class="panel-desc">Upload a farm video or rice image, classify '
        'likely varieties, and estimate a maturity timeline.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    with a:
        if st.button(
            "Open Rice Variety AI →",
            key="home_open_rice_ai"
        ):
            st.session_state["pending_page_choice"] = "Rice Variety AI"
            st.rerun()

    with b:
        if st.button(
            "Open Video Analysis →",
            key="home_open_video_analysis"
        ):
            st.session_state["pending_page_choice"] = "Video Analysis"
            st.rerun()


# -------------------- VIDEO ANALYSIS --------------------
elif page == "Video Analysis":
    st.markdown('<div class="eyebrow">ANALYSIS WORKSPACE / 01</div>', unsafe_allow_html=True)
    st.title("Farm Video Analysis")
    st.markdown('<div class="section-subtitle">Classify sampled video frames and estimate crop maturity from your planting timeline.</div>', unsafe_allow_html=True)

    left, right = st.columns([0.9, 1.35], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Farm information</div><div class="panel-desc">Add the farm details and planting timeline for this recording.</div></div>', unsafe_allow_html=True)
        farm_area = st.number_input("Farm area (hectares)", min_value=0.01, value=1.0, step=0.5, key="video_farm_area")
        video_type = st.selectbox("Recording method", ["Drone / aerial video", "Ground-level video"], key="video_recording_method")
        notes = st.text_input("Farm name or reference (optional)", placeholder="e.g. Farm A / Field 01", key="video_farm_reference")
        planting_date_video = st.date_input(
            "Planting / transplanting date",
            value=date.today(),
            max_value=date.today(),
            key="video_planting_date",
        )
        crop_duration_video = st.number_input(
            "Expected crop duration (days)",
            min_value=70,
            max_value=180,
            value=120,
            step=5,
            key="video_crop_duration",
        )
        st.caption("Maturity is estimated from the date and duration you enter; 120 days is a default, not a guarantee.")

    with right:
        st.markdown('<div class="panel"><div class="panel-title">Upload farm footage</div><div class="panel-desc">Upload an MP4, AVI, MOV or MKV video. The model classifies sampled whole frames, not individual plants.</div></div>', unsafe_allow_html=True)
        uploaded_video = st.file_uploader("Choose farm footage", type=["mp4", "avi", "mov", "mkv"], key="farm_video_upload")
        if uploaded_video:
            st.video(uploaded_video)
            st.caption(f"{uploaded_video.name} · {uploaded_video.size / (1024 * 1024):.2f} MB")
        sample_count = st.slider("Number of frames to analyse", 3, 10, 5, help="Frames are sampled at evenly spaced timestamps.", key="video_ai_sample_count")
        run_video = st.button("Analyse Video & Estimate Maturity →", type="primary", disabled=uploaded_video is None, key="inspect_and_run_video")

    if uploaded_video is not None and run_video:
        temp_path = None
        cap = None
        try:
            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(uploaded_video.getvalue())
                temp_path = temp_file.name

            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                st.error("Unable to read this video. Please try another video.")
            else:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0

                st.markdown("---")
                st.markdown('<div class="eyebrow">VIDEO ANALYSIS RESULTS</div>', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Video duration", f"{duration:.1f} sec")
                m2.metric("Resolution", f"{width} × {height}")
                m3.metric("Farm area", f"{farm_area:.2f} ha")

                ok, first_frame = cap.read()
                preview_col, detail_col = st.columns([1.3, 0.9], gap="large")
                with preview_col:
                    st.markdown('<div class="panel"><div class="panel-title">Video frame preview</div><div class="panel-desc">First readable frame from uploaded footage.</div></div>', unsafe_allow_html=True)
                    if ok:
                        st.image(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    else:
                        st.warning("Could not read a preview frame.")
                with detail_col:
                    st.markdown('<div class="panel"><div class="panel-title">Recording details</div><div class="panel-desc">Uploaded recording summary.</div></div>', unsafe_allow_html=True)
                    st.write("**File name:**", uploaded_video.name)
                    st.write("**Recording method:**", video_type)
                    st.write("**Farm reference:**", notes or "Not provided")
                    st.write("**Frame rate:**", f"{fps:.2f} FPS" if fps > 0 else "Unavailable")

                maturity = calculate_maturity(planting_date_video, crop_duration_video)
                st.markdown("---")
                st.markdown("### 📅 Estimated Crop Maturity")
                st.caption("This is calculated from your planting/transplanting date and expected crop duration, not detected from the video.")
                maturity_col1, maturity_col2, maturity_col3 = st.columns(3)
                maturity_col1.metric("Estimated maturity date", maturity["maturity_date"].strftime("%d %b %Y"))
                maturity_col2.metric("Days remaining", max(0, maturity["remaining"]))
                maturity_col3.metric("Estimated growth stage", maturity["stage"])
                st.progress(maturity["progress"], text=f"Estimated timeline elapsed: {maturity['progress'] * 100:.1f}%")
                if maturity["remaining"] < 0:
                    st.info("The estimated maturity date has passed. Check actual field conditions before harvest.")
                elif maturity["remaining"] == 0:
                    st.info("The estimated maturity date is today. Check actual crop conditions before harvest.")

                metadata = pd.DataFrame([{
                    "File name": uploaded_video.name,
                    "Recording method": video_type,
                    "Farm reference": notes or "",
                    "Farm area (hectares)": farm_area,
                    "Frame count": frame_count,
                    "FPS": fps,
                    "Duration (seconds)": round(duration, 3),
                    "Width": width,
                    "Height": height,
                    "Planting / transplanting date": planting_date_video.isoformat(),
                    "Expected crop duration (days)": maturity["duration_days"],
                    "Estimated maturity date": maturity["maturity_date"].isoformat(),
                    "Days remaining": max(0, maturity["remaining"]),
                    "Estimated growth stage": maturity["stage"],
                    "Timeline elapsed (%)": round(maturity["progress"] * 100, 1),
                }])

                st.markdown("---")
                st.markdown('<div class="eyebrow">VIDEO AI / RICE CLASSIFICATION</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Rice Variety Predictions</div>', unsafe_allow_html=True)
                st.caption("The model classifies sampled whole frames. Aerial footage may not show the close-up features the model learned.")

                timestamps = [duration * i / sample_count for i in range(sample_count)] if duration > 0 and frame_count > 0 else []
                predictions, previews = [], []
                progress_bar = st.progress(0, text="Preparing video frames...")
                model = load_rice_model()

                for index, timestamp in enumerate(timestamps):
                    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                    frame_ok, sampled_frame = cap.read()
                    if frame_ok:
                        rgb = cv2.cvtColor(sampled_frame, cv2.COLOR_BGR2RGB)
                        try:
                            result = model.predict(source=Image.fromarray(rgb), imgsz=224, device="cpu", verbose=False)[0]
                            probs = result.probs
                            top_ids = probs.top5
                            names = model.names
                            best_id = top_ids[0]
                            best_name = names[best_id]
                            confidence = float(probs.data[best_id]) * 100
                            top_five = [{"variety": names[cid], "confidence": float(probs.data[cid]) * 100} for cid in top_ids]
                            predictions.append({
                                "Frame number": int(timestamp * fps) if fps > 0 else index,
                                "Timestamp (seconds)": round(timestamp, 2),
                                "Predicted variety": best_name,
                                "Confidence (%)": round(confidence, 4),
                            })
                            previews.append({"timestamp": timestamp, "image": rgb, "variety": best_name, "confidence": confidence, "top_five": top_five})
                        except Exception as prediction_error:
                            st.warning(f"Could not classify frame at {timestamp:.1f}s: {prediction_error}")
                    progress_bar.progress((index + 1) / max(1, len(timestamps)), text=f"Analysing frame {index + 1} of {len(timestamps)}...")
                progress_bar.empty()

                if predictions:
                    df = pd.DataFrame(predictions)
                    st.session_state["cropai_video_predictions"] = predictions
                    st.session_state["cropai_video_previews"] = previews
                    a, b = st.columns(2)
                    a.metric("Frames analysed", len(predictions))
                    b.metric("Most frequent prediction", df["Predicted variety"].value_counts().idxmax())
                    st.markdown("### Sampled frame predictions")
                    for item in previews:
                        with st.expander(f"{item['timestamp']:.1f}s — {item['variety']} ({item['confidence']:.2f}%)"):
                            st.image(item["image"], caption=f"Sampled at {item['timestamp']:.2f} seconds", use_container_width=True)
                            st.write(f"**Predicted variety:** {item['variety']}")
                            st.progress(max(0.0, min(item["confidence"] / 100, 1.0)), text=f"{item['confidence']:.2f}% model confidence")
                            st.markdown("**Top 5 varieties for this frame**")
                            for rank, option in enumerate(item["top_five"], start=1):
                                st.write(f"{rank}. {option['variety']} — {option['confidence']:.2f}%")
                    st.markdown("### Frame-by-frame summary")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.caption("Confidence is a model score, not a guarantee of correctness. The model does not detect individual plants or crop health.")
                    combined_report = df.copy()
                    for key, value in {
                        "Video file": uploaded_video.name,
                        "Recording method": video_type,
                        "Farm reference": notes or "",
                        "Farm area (hectares)": farm_area,
                        "Planting / transplanting date": planting_date_video.isoformat(),
                        "Expected crop duration (days)": maturity["duration_days"],
                        "Estimated maturity date": maturity["maturity_date"].isoformat(),
                        "Days remaining": max(0, maturity["remaining"]),
                        "Estimated growth stage": maturity["stage"],
                        "Timeline elapsed (%)": round(maturity["progress"] * 100, 1),
                    }.items():
                        combined_report[key] = value
                    st.download_button(
                        "📥 Download Video & Maturity Report (CSV)",
                        combined_report.to_csv(index=False).encode("utf-8"),
                        "cropai_video_maturity_report.csv",
                        "text/csv",
                        key="download_video_ai_predictions",
                    )
                else:
                    st.warning("Video metadata was read, but no sampled frames could be classified. Try a different video.")
                    st.download_button(
                        "📥 Download Video Inspection & Maturity Report (CSV)",
                        metadata.to_csv(index=False).encode("utf-8"),
                        "cropai_video_maturity_report.csv",
                        "text/csv",
                        key="download_video_inspection_report",
                    )

        except Exception as error:
            st.error(f"Video inspection or AI analysis failed: {error}")
        finally:
            if cap is not None:
                cap.release()
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


# -------------------- RICE VARIETY AI --------------------
elif page == "Rice Variety AI":
    st.markdown('<div class="eyebrow">AI CROP CLASSIFICATION</div>', unsafe_allow_html=True)
    st.title("Rice Variety AI")
    st.markdown("Identify a likely rice variety and estimate a crop maturity timeline.")

    uploaded_image = st.file_uploader("Upload a rice image", type=["jpg", "jpeg", "png", "webp"], key="rice_variety_image")
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded rice image", use_container_width=True)
            st.markdown("### 🌱 Crop Growth & Maturity")
            col1, col2 = st.columns(2)
            with col1:
                planting_date = st.date_input("Planting / transplanting date", value=date.today(), max_value=date.today(), key="rice_planting_date")
            with col2:
                crop_duration = st.number_input("Expected crop duration (days)", min_value=70, max_value=180, value=120, step=5, key="rice_crop_duration")
            st.caption("Maturity depends on the planting date and crop duration you enter; 120 days is only a default.")
            if st.button("Predict Rice Variety", type="primary", key="predict_rice_image"):
                with st.spinner("Analysing image with CropAI..."):
                    model = load_rice_model()
                    result = model.predict(source=image, imgsz=224, device="cpu", verbose=False)[0]
                probs, top_ids, names = result.probs, result.probs.top5, model.names
                best_id = top_ids[0]
                best_name = names[best_id]
                best_confidence = float(probs.data[best_id]) * 100
                now = datetime.now().astimezone()
                timestamp_text = now.strftime("%d %B %Y, %I:%M:%S %p %Z")
                timestamp_iso = now.isoformat(timespec="seconds")
                st.markdown("### Prediction Results")
                st.caption(f"🕒 Prediction generated on: {timestamp_text}")
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#123c2b,#1c5b3c);padding:24px;border-radius:16px;border:1px solid #347653;margin:12px 0 24px">
                <div style="color:#a7e5bd;font-size:13px;font-weight:600;letter-spacing:1px">TOP PREDICTION</div>
                <div style="color:white;font-size:28px;font-weight:700;margin:8px 0">{best_name}</div>
                <div style="color:#d7f5df;font-size:16px">Model confidence: {best_confidence:.2f}%</div></div>
                """, unsafe_allow_html=True)

                report_rows = []
                st.markdown("#### Top 5 possible varieties")
                for rank, class_id in enumerate(top_ids, start=1):
                    variety = names[class_id]
                    confidence = float(probs.data[class_id]) * 100
                    st.markdown(f"**{rank}. {variety}**")
                    st.progress(max(0.0, min(confidence / 100, 1.0)), text=f"{confidence:.2f}% confidence")
                    report_rows.append({
                        "Rank": rank, "Image filename": uploaded_image.name,
                        "Predicted variety": variety, "Confidence (%)": round(confidence, 4),
                        "Prediction date and time": timestamp_iso,
                    })
                st.caption("Confidence scores are model estimates, not guaranteed real-world accuracy.")

                maturity = calculate_maturity(planting_date, crop_duration)
                st.markdown("---")
                st.markdown("### 📅 Estimated Crop Maturity")
                m1, m2, m3 = st.columns(3)
                m1.metric("Estimated maturity date", maturity["maturity_date"].strftime("%d %b %Y"))
                m2.metric("Days remaining", max(0, maturity["remaining"]))
                m3.metric("Estimated growth stage", maturity["stage"])
                st.progress(maturity["progress"], text=f"Estimated timeline elapsed: {maturity['progress'] * 100:.1f}%")
                if maturity["remaining"] < 0:
                    st.info("The estimated maturity date has passed. Check actual field conditions; this calculation cannot confirm harvest readiness.")
                elif maturity["remaining"] == 0:
                    st.info("The estimated maturity date is today. Check actual crop conditions before harvest.")
                st.caption("Growth stage is inferred from elapsed time, NOT detected from the image. This is a planning estimate, not verified agronomic advice.")

                for row in report_rows:
                    row.update({
                        "Planting date": planting_date.isoformat(),
                        "Expected crop duration (days)": maturity["duration_days"],
                        "Estimated maturity date": maturity["maturity_date"].isoformat(),
                        "Days remaining": max(0, maturity["remaining"]),
                        "Estimated growth stage": maturity["stage"],
                        "Timeline elapsed (%)": round(maturity["progress"] * 100, 1),
                    })
                report = pd.DataFrame(report_rows)
                st.download_button("📥 Download Prediction & Maturity Report (CSV)", report.to_csv(index=False).encode("utf-8"), "cropai_rice_maturity_report.csv", "text/csv", key="download_rice_report")
        except Exception as error:
            st.error(f"Rice image analysis error: {error}")


# -------------------- ABOUT CROPAI --------------------
elif page == "About CropAI":
    st.markdown('<div class="eyebrow">ABOUT THE PLATFORM</div>', unsafe_allow_html=True)
    st.title("Built for smarter agriculture")
    st.markdown("""
    <div class="hero"><h1>From farm footage to useful insights.</h1>
    <p>CropAI is a developing computer-vision project for rice variety classification,
    farm video inspection and preliminary crop maturity estimation.</p></div>
    """, unsafe_allow_html=True)
    st.markdown("### Development roadmap")
    st.markdown("""
    - **Phase 1:** Farm video upload, inspection and reporting.
    - **Phase 2:** Rice variety classification using a trained model.
    - **Phase 3:** Planting-date-based maturity estimation.
    - **Phase 4:** Image-based growth-stage recognition and field validation.
    """)
    st.warning("CropAI is a prototype. Maturity dates and growth stages are estimates, not verified agronomic advice.")
