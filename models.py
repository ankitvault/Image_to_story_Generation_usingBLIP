import os
import torch
from pathlib import Path
from PIL import Image
from transformers import (
    AutoTokenizer, BlipImageProcessor, BlipProcessor, BlipForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration
)

BASE_DIR = Path(__file__).resolve().parent

class AIPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Loading BLIP...")
        local_blip = BASE_DIR / "k7owntrained"
        if local_blip.exists():
            blip_path = str(local_blip)
        else:
            blip_path = "Salesforce/blip-image-captioning-large"
            
        print(f"Loading BLIP from: {blip_path}")
        self.blip_processor = BlipProcessor(
            image_processor=BlipImageProcessor.from_pretrained(blip_path),
            tokenizer=AutoTokenizer.from_pretrained(blip_path)
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_path)
        self.blip_model.eval()
        self.blip_model.to(self.device)
        print("BLIP loaded")
        
        print("Loading FLAN-T5...")
        flan_t5_id = "google/flan-t5-large"
        self.t5_tokenizer = T5Tokenizer.from_pretrained(flan_t5_id)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(flan_t5_id)
        self.t5_model.eval()
        self.t5_model.to(self.device)
        print("FLAN-T5 loaded")

    def generate_caption(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.blip_model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                repetition_penalty=1.2
            )
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def generate_story(self, captions: list[str]) -> str:
        if not captions:
            raise ValueError("Captions list must not be empty.")

        # Sanitize captions
        captions = [str(c).strip() for c in captions if str(c).strip()]
        if not captions:
            raise ValueError("All captions are empty or whitespace after sanitization.")

        scenes = "\n".join([f"- Scene: {c}" for c in captions])
        prompt = f"""Task: Write a detailed, creative, and engaging narrative story based on the following scenes.
Rules:
- Expand the scenes into a flowing, descriptive story. Add atmospheric details to make it cinematic.
- Write at least 4 to 5 sentences.
- Focus ONLY on the subjects mentioned in the captions. Do NOT introduce any new characters, genders (no women or children), or unrelated events.
- Build a logical sequence of events based on the chronological order of the scenes.

Scenes:
{scenes}

Story:"""

        try:
            inputs = self.t5_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {e}")

        try:
            with torch.no_grad():
                out = self.t5_model.generate(
                    **inputs,
                    max_length=250,
                    min_length=60,
                    temperature=0.6,
                    top_p=0.92,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    do_sample=True
                )
        except Exception as e:
            raise RuntimeError(f"Story generation failed: {e}")

        try:
            story = self.t5_tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"Decoding failed: {e}")

        if not story or not story.strip():
            raise ValueError("Generated story is empty. Try different captions.")

        return story.strip()

    def summarize_story(self, story: str) -> str:
        prompt = f"Summarize the following story in one or two sentences:\n\n{story}"
        
        inputs = self.t5_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.t5_model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        summary = self.t5_tokenizer.decode(out[0], skip_special_tokens=True)
        return summary

# Singleton pipeline instance
pipeline = AIPipeline()