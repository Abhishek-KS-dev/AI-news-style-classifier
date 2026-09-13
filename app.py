import gradio as gr
import spaces
from transformers import pipeline

# Load a model trained on Fact Checking & Claim Verification (FEVER dataset)
pipe = pipeline(
    "text-classification",
    model="DhavalTaunk/fever-roberta-large"
)

@spaces.GPU
def fake_news_detector(text):
    if not text.strip():
        return "Please enter a statement to analyze."

    results = pipe(text)
    best = results[0]

    # FEVER models map to SUPPORTS (True), REFUTES (False/Fake), or NOT ENOUGH INFO
    label_raw = best["label"].upper()
    
    if "SUPPORTS" in label_raw or "LABEL_0" in label_raw:
        final_label = "REAL / FACTUALLY SUPPORTED"
    elif "REFUTES" in label_raw or "LABEL_1" in label_raw:
        final_label = "FAKE / REFUTED"
    else:
        final_label = "UNVERIFIED / NOT ENOUGH INFO"

    confidence = round(best["score"] * 100, 2)

    return (
        f"Prediction: {final_label}\n"
        f"Confidence: {confidence} %\n"
        "Note: This result is based on claim verification patterns, not live web search."
    )

interface = gr.Interface(
    fn=fake_news_detector,
    inputs=gr.Textbox(lines=5, placeholder="Enter a news statement here..."),
    outputs="text",
    title="AI Fact & Claim Verification System",
    description="This system checks whether short claims resemble verified facts or false statements.",
    examples=[
        ["Elon Musk founded SpaceX."],
        ["Scientists confirm the moon will fall to Earth next week."],
        ["Water boils at 100 degrees Celsius."]
    ],
    cache_examples=False
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
  
